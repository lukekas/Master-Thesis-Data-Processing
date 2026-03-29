import os
import rasterio
import shapely
import geopandas as gpd
import numpy as np
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm

try:
    from . import config
except ImportError:
    import config

class DigitalSurfaceModel:
    """
    Class for handling Digital Surface Model (DSM) data. It allows retrieving altitude information for given coordinates by mapping them to the corresponding DSM files. The class is initialized with the path to the directory containing the DSM files, and it sets up an index to efficiently find the relevant DSM file for any given point. The altitude data can be retrieved for a DataFrame containing coordinates, and the results are added as a new column in the DataFrame.
    """
    def __init__(self, dsm_path: str):
        self._dsm_path = dsm_path
        self._set_up_index()

    def _set_up_index(self):
        """
        Sets up an index of the DSM files by reading the metadata of each file in the specified directory. 

        Returns:
            None: The function does not return anything, but it initializes the _dsm_index attribute with a GeoDataFrame containing the file names, their corresponding geometries (bounding boxes), and the CRS information.
        """
        result = []

        for file in os.listdir(self._dsm_path):
            if file.endswith(".tif"):
                filepath = os.path.join(self._dsm_path, file)
                with rasterio.open(filepath) as src:
                    result.append({
                        "file": file,
                        "crs": src.crs,
                        "bounds": src.bounds,
                        "geometry": shapely.geometry.box(*src.bounds)
                    })
                    self._raster_crs = src.crs

        self._dsm_index = gpd.GeoDataFrame(result, geometry="geometry", crs=self._raster_crs)


    def _get_dsm_file(self, x: float, y: float, on_error="raise") -> str:
        """
        Function to retrieve the DSM file that contains the given coordinates (x, y).

        Args:
            x (float): The x-coordinate (longitude) of the point for which to find the corresponding DSM file.
            y (float): The y-coordinate (latitude) of the point for which to find the corresponding DSM file.
            on_error (str, optional): Determines the behavior when no DSM file contains the given point. Defaults to "raise".

        Returns:
            str: The path to the DSM file that contains the given coordinates or None if no file is found.
        """
        point = shapely.geometry.Point(x, y)
        matched = self._dsm_index[self._dsm_index.contains(point)]

        if len(matched) == 0:
            if on_error == "empty":
                return ""
            raise ValueError(f"No DSM file contains the given point: ({x}, {y})")

        dsm_file = matched.iloc[0]["file"]
        return os.path.join(self._dsm_path, dsm_file)

    @staticmethod
    def _round_coordinates(vals: np.array) -> np.array:
        """Rounds the given values to the nearest 0.1, ensuring that the last digit is odd.
        
        Args:   
            vals (np.array): An array of float values representing coordinates to be rounded.

        Returns:
            np.array: An array of float values representing the rounded coordinates.
        """
        vals = np.round(vals, 1)
        vals = np.where(vals * 10 % 2 == 0, vals + 0.1, vals) # last number is even
        vals = np.where(vals * 10 % 10 == 0, vals + 0.1, vals) # last number is 0
        return vals      

    def _get_altitude_data_by_tif_filename(self, filename: str, group: pd.DataFrame, x_col: str="x", y_col: str="y", altitude_col: str="altitude") -> pd.DataFrame:
        """
        Retrieves altitude data for a group of coordinates from a specified DSM file and adds it to the group DataFrame.

        Args:
            filename (str): The path to the DSM file from which to retrieve altitude data.
            group (pd.DataFrame): A DataFrame containing the coordinates for which to retrieve altitude data.
            x_col (str, optional): The name of the column in the group DataFrame that contains the x-coordinates. Defaults to "x".
            y_col (str, optional): The name of the column in the group DataFrame that contains the y-coordinates. Defaults to "y".
            altitude_col (str, optional): The name of the column to be added to the group DataFrame that will contain the retrieved altitude data. Defaults to "altitude".

        Returns:
            pd.DataFrame: The input group DataFrame with an additional column containing the retrieved altitude data.
        """
        if filename == "":
            return None
        
        with rasterio.open(filename) as src:
            band = src.read(1)
            rows, cols = rasterio.transform.rowcol(src.transform, group[x_col].values, group[y_col].values)
            group[altitude_col] = band[rows, cols]

        return group
    
    def get_altitude_for_dataframe(self, df: pd.DataFrame, x_col: str, y_col: str, altitude_col: str) -> pd.DataFrame:
        """
        Retrieve altitude data for the coordinates specified in the input DataFrame and add it as a new column.

        Args:
            df (pd.DataFrame): A DataFrame containing the coordinates for which to retrieve altitude data.
            x_col (str): The name of the column in the input DataFrame that contains the x-coordinates.
            y_col (str): The name of the column in the input DataFrame that contains the y-coordinates.
            altitude_col (str): The name of the column to be added to the input DataFrame that will contain the retrieved altitude data.
        
        Returns:
            pd.DataFrame: The input DataFrame with an additional column containing the retrieved altitude data.
        """
        old_crs = df.crs if hasattr(df, "crs") else config.default_epsg
        gpd_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x_col], df[y_col]), crs=old_crs).to_crs(self._raster_crs)
        
        gpd_df[x_col] = gpd_df.geometry.x
        gpd_df[y_col] = gpd_df.geometry.y

        start_time = time.time()
        df[altitude_col] = pd.Series(np.nan, index=df.index, dtype="float64")
        gpd_df[x_col] = self._round_coordinates(gpd_df[x_col])
        gpd_df[y_col] = self._round_coordinates(gpd_df[y_col])
        print(f"Rounding done in {(time.time() - start_time):.2f} seconds.")
        print("Mapping DSM files.")
        gpd_df["filename"] = gpd_df.apply(lambda row: self._get_dsm_file(row[x_col], row[y_col], on_error="empty"), axis=1)

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self._get_altitude_data_by_tif_filename, filename, group, x_col, y_col, altitude_col): (filename, group) for filename, group in gpd_df.groupby("filename")}
            
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Getting node altitudes per tif tile"):
                current_group = future.result()
                if current_group is not None:
                    df.loc[current_group.index, altitude_col] = current_group[altitude_col]
        return df
