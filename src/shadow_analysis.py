import time
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import pybdshadow

try:
    from .digital_surface_model import DigitalSurfaceModel
except ImportError:
    from digital_surface_model import DigitalSurfaceModel

class ShadowCalculation:
    """
    Class for calculating shadow areas based on a Digital Surface Model (DSM), Digital Ground Model (DGM), building geometries, and Local Climate Zones (LCZ).
    """
    def __init__(self, dsm: DigitalSurfaceModel, dgm: DigitalSurfaceModel, gdf_buildings: gpd.GeoDataFrame, lcz: gpd.GeoDataFrame, set_height: bool=True):
        self.crs = "EPSG:4326"
        self.dsm = dsm
        self.dgm = dgm
        self.gdf_buildings = gdf_buildings
        self.lcz = lcz.to_crs(self.crs)
        self.store = {}

        self._preprocess_buildings(set_height)

    def _preprocess_buildings(self, set_height: bool):
        """
        Preprocesses the building geometries by adding height information from the DSM and DGM, and dissolving buildings by height to speed up shadow calculation.

        Args:
            set_height (bool): Whether to set the height of the buildings based on the DSM and DGM. 

        Returns:
            None: Modification of the buildings in place.
        """
        gdf_buildings = self.gdf_buildings.to_crs("EPSG:25832")
        gdf_buildings["x"] = gdf_buildings.geometry.centroid.x
        gdf_buildings["y"] = gdf_buildings.geometry.centroid.y
        if set_height:
            gdf_buildings = gdf_buildings[gdf_buildings["geometry"] != shapely.Polygon()].copy() # keep only buildings with non-empty geometries
            gdf_buildings = gdf_buildings.drop(columns=["height"], errors="ignore")  # remove OSM height strings to avoid dtype conflicts
            gdf_buildings = self.dsm.get_altitude_for_dataframe(gdf_buildings, x_col="x", y_col="y", altitude_col="height")
            gdf_buildings = self.dgm.get_altitude_for_dataframe(gdf_buildings, x_col="x", y_col="y", altitude_col="ground_height")
            gdf_buildings["height"] = gdf_buildings["height"] - gdf_buildings["ground_height"]
            gdf_buildings = gdf_buildings.drop(columns=["ground_height"])
            gdf_buildings["height"] = gdf_buildings["height"].clip(lower=10) # set lowest building height to 10m
            gdf_buildings["height"] = gdf_buildings["height"].astype("float").round(1)
        gdf_buildings = gdf_buildings.dissolve(by="height").reset_index().to_crs(self.crs) # dissolve buildings by height to speed up shadow calculation, as buildings with the same height have the same shadow

        self.gdf_buildings = pybdshadow.bd_preprocess(gdf_buildings)
        buildings_25832 = self.gdf_buildings.to_crs("EPSG:25832").buffer(0).values
        self._buildings_geoms_25832 = buildings_25832
        self._buildings_tree_25832 = shapely.STRtree(buildings_25832)
        lcz_a = self.lcz[self.lcz["lcz"] == "A"].to_crs("EPSG:25832")
        self._lcz_a_geoms_25832 = lcz_a.geometry.buffer(0).values if len(lcz_a) > 0 else np.array([], dtype=object)


    def _get_gdf_shadows(self, datetime: pd.Timestamp, use_store: bool=True) -> gpd.GeoDataFrame:
        """
        Retrieves the shadow geometries for a given datetime. If use_store is True, it will check if the shadow geometries for the given datetime are already calculated and stored in the store dictionary, and return them if available. If not available or if use_store is False, it will calculate the shadow geometries using the pybdshadow library, store them in the store dictionary, and return them.

        Args:
            datetime (pd.Timestamp): The datetime for which to retrieve the shadow geometries.
            use_store (bool): Whether to use the store to cache shadow geometries for different datetimes. Defaults to True.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the shadow geometries for the given datetime.
        """
        if use_store and datetime in self.store:
            return self.store[datetime]
        
        gdf_shadows = pybdshadow.bdshadow_sunlight(self.gdf_buildings, datetime, roof=False, include_building=False)
        self.store[datetime] = gdf_shadows
        return gdf_shadows


    def _strtree_difference(self, buffers_arr: np.ndarray, geoms: np.ndarray, tree: shapely.STRtree) -> np.ndarray:
        """
        Subtract geoms from each buffer using a spatial index. Avoids a city-wide union.
        
        Args:
            buffers_arr (np.ndarray): An array of geometries representing the buffers from which to subtract the geoms.
            geoms (np.ndarray): An array of geometries to be subtracted from the buffers.
            tree (shapely.STRtree): A spatial index built on the geoms to efficiently query which geoms intersect with each buffer.

        Returns:
            np.ndarray: An array of geometries representing the result of subtracting the geoms from each buffer.
        """
        buf_idx, geom_idx = tree.query(buffers_arr, predicate="intersects")
        result = buffers_arr.copy()
        if len(buf_idx) == 0:
            return result
        order = np.argsort(buf_idx, kind="stable")
        sorted_buf, sorted_geom = buf_idx[order], geom_idx[order]
        unique_bufs, starts = np.unique(sorted_buf, return_index=True)
        ends = np.append(starts[1:], len(sorted_buf))
        for bi, s, e in zip(unique_bufs, starts, ends):
            local_union = shapely.union_all(geoms[sorted_geom[s:e]])
            result[bi] = shapely.difference(result[bi], local_union)
        return result

    def _strtree_intersection_area(self, buffers_arr: np.ndarray, geoms: np.ndarray, tree: shapely.STRtree) -> np.ndarray:
        """
        Compute intersection area of each buffer with geoms using a spatial index.
        
        Args:
            buffers_arr (np.ndarray): An array of geometries representing the buffers for which to calculate the intersection area with the geoms.
            geoms (np.ndarray): An array of geometries with which to calculate the intersection area for each buffer.
            tree (shapely.STRtree): A spatial index built on the geoms to efficiently query which geoms intersect with each buffer.

        Returns:
            np.ndarray: An array of float values representing the intersection area of each buffer with the geoms.
        """
        buf_idx, geom_idx = tree.query(buffers_arr, predicate="intersects")
        overlap_areas = np.zeros(len(buffers_arr))
        if len(buf_idx) == 0:
            return overlap_areas
        order = np.argsort(buf_idx, kind="stable")
        sorted_buf, sorted_geom = buf_idx[order], geom_idx[order]
        unique_bufs, starts = np.unique(sorted_buf, return_index=True)
        ends = np.append(starts[1:], len(sorted_buf))
        for bi, s, e in zip(unique_bufs, starts, ends):
            local_union = shapely.union_all(geoms[sorted_geom[s:e]])
            overlap_areas[bi] = shapely.area(shapely.intersection(buffers_arr[bi], local_union))
        return overlap_areas

    def calculate_street_shadow_coverage_percentage_batch(self, streets: gpd.GeoSeries, datetime: pd.Timestamp, max_street_width: int=10, use_store: bool=True) -> pd.Series:
        """
        Calculates the percentage of each street that is covered by shadows for a given datetime.

        Args:
            streets (gpd.GeoSeries): A GeoSeries containing the geometries of the streets for which to calculate the shadow coverage percentage.
            datetime (pd.Timestamp): The datetime for which to calculate the shadow coverage percentage.
            max_street_width (int): The maximum width of the street buffers to consider for shadow coverage calculation. Defaults to 10.
            use_store (bool): Whether to use the store to cache shadow geometries for different datetimes. Defaults to True.

        Returns:
            pd.Series: A Series containing the percentage of each street that is covered by shadows for the given datetime.
        """
        start_time = time.time()
        gdf_shadows = self._get_gdf_shadows(datetime, use_store=use_store)
        print(f"Shadow geometries retrieved in {(time.time() - start_time):.2f} seconds.")

        shadow_tree_key = ("shadow_tree", datetime)
        if shadow_tree_key not in self.store:
            start_time = time.time()
            shadow_geoms = gdf_shadows.to_crs("EPSG:25832").geometry.buffer(0).values
            if len(self._lcz_a_geoms_25832) > 0:
                shadow_geoms = np.concatenate([shadow_geoms, self._lcz_a_geoms_25832])
            self.store[shadow_tree_key] = (shadow_geoms, shapely.STRtree(shadow_geoms))
            print(f"Shadow tree built in {(time.time() - start_time):.2f} seconds.")
        shadow_geoms, shadow_tree = self.store[shadow_tree_key]

        buffers_arr = streets.to_crs("EPSG:25832").buffer(max_street_width / 2, cap_style=2, join_style=2).buffer(0).values

        start_time = time.time()
        buffers_arr = self._strtree_difference(buffers_arr, self._buildings_geoms_25832, self._buildings_tree_25832)
        print(f"Street buffers subtracted by buildings in {(time.time() - start_time):.2f} seconds.")

        start_time = time.time()
        overlap_areas = self._strtree_intersection_area(buffers_arr, shadow_geoms, shadow_tree)
        print(f"Street buffers overlap area calculated in {(time.time() - start_time):.2f} seconds.")

        start_time = time.time()
        buffer_areas = shapely.area(buffers_arr)
        percentage_in_shadow = np.divide(overlap_areas, buffer_areas, out=np.zeros(len(buffers_arr)), where=buffer_areas > 0)
        print(f"Street buffers percentage in shadow calculated in {(time.time() - start_time):.2f} seconds.")

        return pd.Series(percentage_in_shadow, index=streets.index)


    def calculate_solar_exposure_index(self, streets: gpd.GeoSeries, datetime: pd.Timestamp, max_street_width: int=10, use_store: bool=True) -> pd.Series:
        """
        Calculates a solar exposure index for each street, which is defined as 1 minus the percentage of the street that is covered by shadows for a given datetime.
        
        Args:
            streets (gpd.GeoSeries): A GeoSeries containing the geometries of the streets for which to calculate the solar exposure index.
            datetime (pd.Timestamp): The datetime for which to calculate the solar exposure index.
            max_street_width (int): The maximum width of the street buffers to consider for shadow coverage calculation. Defaults to 10.
            use_store (bool): Whether to use the store to cache shadow geometries for different datetimes. Defaults to True.

        Returns:
            pd.Series: A Series containing the solar exposure index for each street.
        """
        percentage_in_shadow = self.calculate_street_shadow_coverage_percentage_batch(streets, datetime, max_street_width=max_street_width, use_store=use_store)
        solar_exposure_index = 1 - percentage_in_shadow
        return solar_exposure_index
