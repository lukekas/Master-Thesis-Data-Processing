from typing import Literal
import networkx as nx
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
import geopandas as gpd
import shapely
import osmnx as ox
from enum import Enum
import pickle
import time
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape

try:
    from .digital_surface_model import DigitalSurfaceModel
    from .shadow_analysis import ShadowCalculation
    from . import config
    from .config import DayTimeNames, DayTimes
except ImportError:
    from digital_surface_model import DigitalSurfaceModel
    from shadow_analysis import ShadowCalculation
    import config
    from config import DayTimeNames, DayTimes


class EdgeType(Enum):
    OSM_WALK = "walk"
    OSM_BIKE = "bike"
    OSM_TO_STATION = "osm_to_station"
    PUBLIC_TRANSPORTATION = "public_transportation"


class TransportationMode(Enum):
    WALK = "walk"
    BIKE = "bike"
    WALK_PT = "walk_pt"


class Attributes(Enum):
    ALTITUDE = "altitude"
    INCLINATION_PERCENTAGE = "inclination_percentage"
    DURATION_TIME = "time"
    TRANSPORTATION_TYPE = "transportation_type"
    CHANGE_PT_TIME = "change_pt_time"

    @staticmethod
    def duration_time(
        time_of_day: config.DayTimeNames, vulnerability_group: int
    ) -> str:
        return f"{Attributes.DURATION_TIME.value}_{time_of_day.value}_vuln{vulnerability_group}"

    @staticmethod
    def solar_exposure(daytime) -> str:
        if isinstance(daytime, config.DayTimeNames):
            return f"solar_exposure_{daytime.value}"
        else:
            return f"solar_exposure_{daytime}"


class Graph:
    """
    Class representing a graph model for a specific location and transportation mode, enriched with topographical information such as altitude and inclination. The graph is created using OSMnx to retrieve the walkable or bikeable network for the specified location, and it can be enriched with altitude data from a DigitalSurfaceModel. The class also includes methods for loading public transportation stops from GTFS data and adding edges between the walkable network and the public transportation stations.
    """

    def __init__(
        self,
        location: str,
        dsm: DigitalSurfaceModel,
        transportation_mode: Literal[
            TransportationMode.WALK,
            TransportationMode.BIKE,
            TransportationMode.WALK_PT,
        ],
        lcz: gpd.GeoDataFrame,
        include_solar_exposure_index: bool = True,
        include_inclination: bool = True,
        dgm: DigitalSurfaceModel = None,
    ):
        self.location = location
        self.dsm = dsm
        self.transportation_mode = transportation_mode
        self.lcz = lcz
        self.include_solar_exposure_index = include_solar_exposure_index
        self.include_inclination = include_inclination
        if include_solar_exposure_index and dgm is None:
            raise ValueError(
                "If solar exposure index is included, a DigitalSurfaceModel for ground heights (dgm) must be provided."
            )
        self.dgm = dgm

        self.__create_new_graph(location, transportation_mode)

    def enrich_graph_with_altitude_and_time(self) -> None:
        """
        Adds topographical information to the given graph using the provided DigitalSurfaceModel.
        Appends mostly elevation data to nodes and calculates inclination and duration_time for edges.

        Args:
            None

        Returns:
            None
        """
        # add elevation data to nodes
        df_nodes = pd.DataFrame(
            [{"id": node, **data} for node, data in self.G.nodes(data=True)]
        )
        df_nodes_with_altitude = self.dsm.get_altitude_for_dataframe(
            df_nodes, x_col="x", y_col="y", altitude_col=Attributes.ALTITUDE.value
        )
        updating = {
            row["id"]: row[Attributes.ALTITUDE.value]
            for _, row in df_nodes_with_altitude.iterrows()
        }
        nx.set_node_attributes(self.G, updating, name=Attributes.ALTITUDE.value)

        # add inclination and duration_time to edges
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.get_duration_time_by_edge,
                    edge,
                    self.G.nodes[edge[0]][Attributes.ALTITUDE.value],
                    self.G.nodes[edge[1]][Attributes.ALTITUDE.value],
                    self.include_solar_exposure_index,
                    self.include_inclination,
                ): edge
                for edge in list(self.G.edges(data=True, keys=True))
            }
            results = []

            for future in tqdm.tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Setting time duration attributes on edges",
            ):
                results.append(future.result())

        updates = {edge: attrs for edge, attrs in results}
        nx.set_edge_attributes(self.G, updates)

    def _load_stops(
        self, gtfs_path: str, bounding_box: shapely.Polygon
    ) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Loads public transportation stops from GTFS data, filters them by the given bounding box, and converts them to a GeoDataFrame.

        Args:
            gtfs_path (str): The file path to the GTFS data directory.
            bounding_box (shapely.Polygon): A polygon defining the area of interest for filtering stops.

        Returns:
            tuple[pd.DataFrame, gpd.GeoDataFrame]: A tuple containing a DataFrame of stops and a GeoDataFrame of stops with geometries.
        """
        df_stops = pd.read_csv(f"{gtfs_path}/stops.txt")
        df_stops["stop_id"] = df_stops["stop_id"].astype(str)
        gdf_stops = gpd.GeoDataFrame(
            df_stops,
            geometry=gpd.points_from_xy(df_stops.stop_lon, df_stops.stop_lat),
            crs=config.wgs_84,
        )
        gdf_stops = gdf_stops[gdf_stops.within(bounding_box)].copy()
        gdf_stops = gdf_stops.to_crs(config.default_epsg)
        gdf_stops["x"] = gdf_stops["geometry"].x
        gdf_stops["y"] = gdf_stops["geometry"].y
        gdf_stops = gdf_stops.drop(columns=["stop_lon", "stop_lat"])
        df_stops = pd.DataFrame(gdf_stops)

        return df_stops, gdf_stops

    def _load_stop_times(self, gtfs_path: str, df_stops: pd.DataFrame) -> pd.DataFrame:
        """
        Loads stop times from GTFS data, filters them to only include stops that are in the provided DataFrame of stops, and processes the arrival and departure times.

        Args:
            gtfs_path (str): The file path to the GTFS data directory.
            df_stops (pd.DataFrame): A DataFrame containing the stops that are in the area of interest, used to filter the stop times.

        Returns:
            pd.DataFrame: A DataFrame containing the stop times for the stops in the area of interest, with processed arrival and departure times.
        """
        df_stop_times = pd.read_csv(f"{gtfs_path}/stop_times.txt", low_memory=False)
        df_stop_times["stop_id"] = df_stop_times["stop_id"].astype(str)
        df_stop_times = df_stop_times[df_stop_times["arrival_time"].notna()].copy()
        df_stop_times["arrival_time"] = df_stop_times["arrival_time"].apply(
            lambda x: x if int(str(x)[:2]) < 24 else "00" + x[2:]
        )
        df_stop_times["departure_time"] = df_stop_times["departure_time"].apply(
            lambda x: x if int(str(x)[:2]) < 24 else "00" + x[2:]
        )
        df_stop_times["arrival_time"] = pd.to_datetime(
            df_stop_times["arrival_time"], format="%H:%M:%S"
        )
        df_stop_times["departure_time"] = pd.to_datetime(
            df_stop_times["departure_time"], format="%H:%M:%S"
        )
        df_stop_times = df_stop_times.merge(
            df_stops[["stop_id"]], on="stop_id"
        )  # inner join to only keep stop times for stops that are actually in the area of interest

        return df_stop_times

    def _get_frequency_for_stops(
        self,
        df_stops: pd.DataFrame,
        df_stop_times: pd.DataFrame,
        df_trips: pd.DataFrame,
        df_routes: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculates the frequency in seconds of public transportation for each stop based on the stop times, trips, and routes data from GTFS.

        Args:
            df_stops (pd.DataFrame): A DataFrame containing the stops in the area of interest.
            df_stop_times (pd.DataFrame): A DataFrame containing the stop times for the stops in the area of interest.
            df_trips (pd.DataFrame): A DataFrame containing the trips data from GTFS.
            df_routes (pd.DataFrame): A DataFrame containing the routes data from GTFS.

        Returns:
            pd.Series: A Series containing the frequency of public transportation for each stop, indexed by stop_id.
        """
        df_stops_stop_times = pd.merge(
            df_stops,
            df_stop_times[["trip_id", "stop_id", "arrival_time", "departure_time"]],
            on="stop_id",
        )
        df_stops_with_route_id = pd.merge(
            df_stops_stop_times, df_trips[["route_id", "trip_id"]], on="trip_id"
        )
        df_stops_with_short_name = pd.merge(
            df_stops_with_route_id,
            df_routes[["route_id", "route_short_name"]],
            on="route_id",
        )

        def calculate_frequency(df: pd.DataFrame) -> float:
            """
            Calculates the frequency in seconds for public transportation stops based on the departure times in the provided DataFrame.

            Args:
                df (pd.DataFrame): A DataFrame containing the departure times.

            Returns:
                float: The frequency in seconds of public transportation for the stops in the provided DataFrame.
            """
            df = df.sort_values(by="departure_time")
            df["previous_departure"] = df["departure_time"].shift()
            df["frequency"] = np.where(
                (
                    df["previous_departure"].notna()
                    & df["departure_time"].notna()
                    & (df["departure_time"] > df["previous_departure"])
                ),
                (df["departure_time"] - df["previous_departure"]).dt.total_seconds(),
                None,
            )
            mode = df["frequency"].mode()
            if len(mode) == 0:
                return None

            return mode.min()  # smallest of the most often occurring frequencies if there are multiple modes, otherwise the only mode

        df_frequency_with_line = df_stops_with_short_name.groupby(
            ["route_short_name", "stop_id"]
        ).apply(calculate_frequency, include_groups=False)
        series_frequency_by_station = df_frequency_with_line.groupby("stop_id").apply(
            lambda x: x.mode().min() if len(x.mode()) > 0 else None
        )
        df_frequency_by_station = pd.DataFrame(
            series_frequency_by_station, columns=["frequency"]
        )

        return df_frequency_by_station

    def _generate_edges_to_stations(
        self, gdf_virtual_nodes: gpd.GeoDataFrame
    ) -> list[tuple[int, int, dict]]:
        """
        Generates new edges between the projected points on the walkable edges and the public transportation stations, including the duration time attribute based on the distance from the stop to the projected point.

        Args:
            gdf_virtual_nodes (gpd.GeoDataFrame): A GeoDataFrame containing the virtual nodes created by projecting the public transportation stations onto the nearest walkable edges, with columns for the original node ids (u and v), the geometry of the edge, the geometry of the station, and the distance from the station to the projected point.

        Returns:
            list[tuple[int, int, dict]]: A list of tuples representing the new edges to be added to the graph, where each tuple contains the source node id, the target node id, and a dictionary of edge attributes including length, geometry, transportation type, and duration time.
        """
        new_edges = []

        # add new edges from u to stop and from stop to v via projected point on edge
        for _, row in gdf_virtual_nodes.iterrows():
            changing_time = (
                row["frequency"] / 2 if not pd.isna(row["frequency"]) else 0
            )  # halve the change time for average waiting time
            linestrings_around_geometry = shapely.ops.split(
                row["geometry_edge"], row["geometry"].buffer(1e-9)
            ).geoms  # split linestring of edge at the nearest point to the public transportation station

            linestring_u_new_node = linestrings_around_geometry[0]
            linestring_u_station_node = shapely.LineString(
                list(linestring_u_new_node.coords) + [row["geometry_station"].coords[0]]
            )  # generate linestring from u to the public transportation station via nearest point on edge
            new_edges.append(
                (
                    row["u"],
                    row["stop_id"],
                    {
                        "length": linestring_u_station_node.length,
                        "geometry": linestring_u_station_node,
                        Attributes.TRANSPORTATION_TYPE.value: EdgeType.OSM_TO_STATION.value,
                        Attributes.CHANGE_PT_TIME.value: changing_time,
                    },
                )
            )
            new_edges.append(
                (
                    row["stop_id"],
                    row["u"],
                    {
                        "length": linestring_u_station_node.length,
                        "geometry": linestring_u_station_node,
                        Attributes.TRANSPORTATION_TYPE.value: EdgeType.OSM_TO_STATION.value,
                    },
                )  # no change time since it is not needed when leaving buses
            )

            linestring_v_new_node = linestrings_around_geometry[-1]
            linestring_v_station_node = shapely.LineString(
                [row["geometry_station"].coords[0]] + list(linestring_v_new_node.coords)
            )  # generate linestring from the public transportation station to v via nearest point on edge
            new_edges.append(
                (
                    row["v"],
                    row["stop_id"],
                    {
                        "length": linestring_v_station_node.length,
                        "geometry": linestring_v_station_node,
                        Attributes.TRANSPORTATION_TYPE.value: EdgeType.OSM_TO_STATION.value,
                        Attributes.CHANGE_PT_TIME.value: changing_time,
                    },
                )
            )
            new_edges.append(
                (
                    row["stop_id"],
                    row["v"],
                    {
                        "length": linestring_v_station_node.length,
                        "geometry": linestring_v_station_node,
                        Attributes.TRANSPORTATION_TYPE.value: EdgeType.OSM_TO_STATION.value,
                    },
                )  # no change time since it is not needed when leaving buses
            )

        return new_edges

    def _get_virtual_nodes_for_stops(
        self, gdf_stops: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        For each public transportation stop, finds the nearest walkable edge in the graph, projects the stop onto that edge to create a new virtual node, and calculates the distance from the stop to the projected point on the edge.

        Args:
            gdf_stops (gpd.GeoDataFrame): A GeoDataFrame containing the public transportation stops with their geometries.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the virtual nodes created by projecting the public transportation stations onto the nearest walkable edges, with columns for the original node ids (u and v), the geometry of the edge, the geometry of the station, and the distance from the station to the projected point.
        """
        # add edges between stops and nearest walkable edge
        # Idea: For each stop, find nearest edge, project point onto edge, create new node at projected point, add two new edges (from u to new point and from new point to v) into two edges
        _, edges = ox.graph_to_gdfs(self.G, nodes=True, edges=True)
        gdf_edges = edges.to_crs(config.default_epsg).copy().reset_index()
        gdf_edges["geometry_edge"] = gdf_edges["geometry"].copy()

        gdf_virtual_nodes = gdf_stops.to_crs(config.default_epsg).sjoin_nearest(
            gdf_edges[["geometry", "geometry_edge", "u", "v", "key"]], how="left"
        )  # spatial join to find nearest walkable edge (way) for each stop

        gdf_virtual_nodes["geometry_station"] = gdf_virtual_nodes["geometry"].copy()
        gdf_virtual_nodes["geometry"] = gdf_virtual_nodes.apply(
            lambda row: row["geometry_edge"].interpolate(
                row["geometry_edge"].project(row["geometry_station"])
            ),
            axis=1,
        )  # project stop point onto nearest walkable edge (way)
        gdf_virtual_nodes["distance_stop_new_node"] = gdf_virtual_nodes.apply(
            lambda row: row["geometry"].distance(row["geometry_station"]), axis=1
        )  # calculate length from stop to projected point on walkable edge

        return gdf_virtual_nodes

    def _get_pt_edges(self, df_stop_times: pd.DataFrame) -> list[tuple[int, int, dict]]:
        """
        Generates edges for real public transportation trips between stops based on the stop times data from GTFS. For each pair of consecutive stops in the same trip, an edge is created with the duration time calculated as the difference between the arrival time at the next stop and the departure time from the current stop.

        Args:
            df_stop_times (pd.DataFrame): A DataFrame containing the stop times for the stops in the area of interest, with processed arrival and departure times.

        Returns:
            list[tuple[int, int, dict]]: A list of tuples representing the edges for real public transportation trips between stops, where each tuple contains the source stop_id, the target stop_id, and a dictionary of edge attributes including duration time and transportation type.
        """
        # add edge for real public transportation trips between stops
        df_pt_edges = df_stop_times.copy()

        df_pt_edges["next_trip_id"] = df_pt_edges["trip_id"].shift(
            -1
        )  # add column with the trip id of the next row
        df_pt_edges["next_trip_id_fits"] = (
            df_pt_edges["trip_id"] == df_pt_edges["next_trip_id"]
        )  # add column which flags whether the next stop belongs to the same trip

        df_pt_edges["next_stop_id"] = df_pt_edges["stop_id"].shift(
            -1
        )  # add column with next stop id
        df_pt_edges["next_arrival_time"] = df_pt_edges["arrival_time"].shift(
            -1
        )  # add column with next arrival time

        df_pt_edges.loc[
            df_pt_edges["next_trip_id_fits"], Attributes.DURATION_TIME.value
        ] = (
            (
                df_pt_edges["next_arrival_time"] - df_pt_edges["departure_time"]
            ).dt.total_seconds()
            / 60
        )  # add column "duration_time" which is the number of minutes between the two stops

        df_pt_edges = df_pt_edges[
            ["stop_id", "next_stop_id", Attributes.DURATION_TIME.value]
        ]  # only keep necessary rows

        df_pt_edges = df_pt_edges[
            df_pt_edges[Attributes.DURATION_TIME.value].notna()
        ]  # only keep rows where "duration_time" is a valid value
        df_pt_edges[Attributes.TRANSPORTATION_TYPE.value] = (
            EdgeType.PUBLIC_TRANSPORTATION.value
        )
        df_pt_edges = df_pt_edges.drop_duplicates(
            ["stop_id", "next_stop_id", Attributes.DURATION_TIME.value]
        )
        df_pt_edges[Attributes.DURATION_TIME.value] = np.where(
            df_pt_edges[Attributes.DURATION_TIME.value] == 0,
            0.5,
            df_pt_edges[Attributes.DURATION_TIME.value],
        )  # replace 0 minute trips with 0.5 minute trips to avoid issues in routing
        df_pt_edges = df_pt_edges[
            df_pt_edges[Attributes.DURATION_TIME.value] > 0
        ].copy()  # only keep edges with positive duration time
        return [
            (row["stop_id"], row["next_stop_id"], dict(row))
            for _, row in df_pt_edges.iterrows()
        ]

    def add_public_transportation_information(
        self, gtfs_path: str, bounding_box: shapely.Polygon
    ) -> None:
        """
        Adds public transportation information to the graph by loading GTFS data, calculating the frequency of public transportation for each stop, and adding edges between stops and the nearest walkable edge in the graph, as well as edges for real public transportation trips between stops.

        Args:
            gtfs_path (str): The file path to the GTFS data directory.
            bounding_box (shapely.Polygon): A polygon defining the area of interest for filtering stops.

        Returns:
            None
        """
        df_stops, gdf_stops = self._load_stops(gtfs_path, bounding_box)  # load stops
        df_stop_times = self._load_stop_times(gtfs_path, df_stops)  # load stop times
        df_routes = pd.read_csv(f"{gtfs_path}/routes.txt")  # load routes
        df_trips = pd.read_csv(f"{gtfs_path}/trips.txt")  # load trips

        df_freq = self._get_frequency_for_stops(
            gdf_stops, df_stop_times, df_trips, df_routes
        )
        gdf_stops = gdf_stops.join(
            df_freq, on="stop_id", how="left"
        )  # add frequency information to gdf_stops

        self.G.add_nodes_from(
            map(
                lambda stop: (stop["stop_id"], stop),
                gdf_stops.to_dict(orient="records"),
            )
        )  # add stops to graph as nodes

        gdf_virtual_nodes = self._get_virtual_nodes_for_stops(
            gdf_stops
        )  # get virtual nodes for stops by projecting them onto the nearest walkable edge
        new_edges = self._generate_edges_to_stations(gdf_virtual_nodes)
        self.G.add_edges_from(new_edges)

        pt_edges = self._get_pt_edges(df_stop_times)
        self.G.add_edges_from(pt_edges)

    def store(self, filepath: str) -> None:
        """
        Stores the graph as a pickle file at the given filepath.

        Args:
            filepath (str): The path where the graph should be stored.

        Returns:
            None
        """
        with open(filepath, "wb") as file:
            pickle.dump(self.G, file)

    def __create_new_graph(
        self,
        location: str,
        transportation_mode: Literal[
            TransportationMode.WALK,
            TransportationMode.BIKE,
            TransportationMode.WALK_PT,
        ],
    ) -> nx.MultiDiGraph:
        """
        Creates a new graph for the given location and transportation mode. The graph is simplified and projected to the default EPSG code. Motorway edges are removed, and if the transportation mode is public transportation, additional information from GTFS data is added.

        Args:
            location (str): The location for which the graph should be created.
            transportation_mode (TransportationMode): The transportation mode for which the graph should be created.

        Returns:
            nx.MultiDiGraph: The created graph.
        """
        network_type = (
            TransportationMode.BIKE.value
            if transportation_mode == TransportationMode.BIKE
            else TransportationMode.WALK.value
        )
        self.G = ox.graph_from_place(location, simplify=True)
        nx.set_edge_attributes(
            self.G, network_type, name=Attributes.TRANSPORTATION_TYPE.value
        )
        self.G = ox.project_graph(self.G, to_crs=config.default_epsg)

        self.__remove_motorway_edges()

        if transportation_mode == TransportationMode.WALK_PT:
            self.add_public_transportation_information(
                gtfs_path=config.gtfs_path,
                bounding_box=ox.geocode_to_gdf(location).iloc[0]["geometry"],
            )

        if self.include_solar_exposure_index:
            self.add_solar_exposure_index()
        self.enrich_graph_with_altitude_and_time()

        self._add_node_coordinates()
        self.G = ox.project_graph(
            self.G, to_crs=config.wgs_84
        )  # for compatibility with Vienna approach

    @staticmethod
    def get_duration_time_by_edge(
        edge: tuple,
        altitude_u: float,
        altitude_v: float,
        include_solar_exposure_index: bool,
        include_inclination: bool = True,
    ) -> tuple:
        """
        Calculates the duration time for a given edge based on the transportation type, length, and inclination. For walking and biking edges, the duration time is calculated using a speed model that takes into account the inclination percentage.

        Args:
            edge (tuple): A tuple representing the edge, containing the source node id, target node id, edge key, and a dictionary of edge attributes.
            altitude_u (float): The altitude of the source node.
            altitude_v (float): The altitude of the target node.
            include_solar_exposure_index (bool): A flag indicating whether to include the solar exposure index in the duration time calculation.
            include_inclination (bool): A flag indicating whether to include the inclination in the duration time calculation.

        Returns:
            tuple: A tuple containing the edge (source node id, target node id, edge key) and a dictionary of new attributes including inclination percentage and duration time (in minutes), as well as duration time adjusted for solar exposure if wanted.
        """
        u, v, edge_key, data = edge

        if (
            data.get(Attributes.TRANSPORTATION_TYPE.value)
            == EdgeType.PUBLIC_TRANSPORTATION.value
        ):
            return (
                u,
                v,
                edge_key,
            ), {}  # duration time is already set for public transportation edges

        if (
            altitude_u is None
            or altitude_v is None
            or np.isnan(altitude_u)
            or np.isnan(altitude_v)
        ):
            altitude_difference = 0  # fallback if no altitude data is available
        else:
            altitude_difference = altitude_v - altitude_u

        inclination_percentage = altitude_difference / data["length"]

        duration_time = None

        # walking speed model based on inclination
        if (
            data.get(Attributes.TRANSPORTATION_TYPE.value)
            == EdgeType.OSM_TO_STATION.value
            or data.get(Attributes.TRANSPORTATION_TYPE.value) == EdgeType.OSM_WALK.value
        ):
            walking_speed = config.base_walking_speed
            if include_inclination:
                if inclination_percentage < -0.15:
                    walking_speed += 0.08  # in m/s
                elif inclination_percentage < -0.10:
                    walking_speed += 0.07
                elif inclination_percentage < -0.05:
                    walking_speed += 0.06
                elif inclination_percentage < 0:
                    walking_speed += 0
                elif inclination_percentage < 0.05:
                    walking_speed += 0
                elif inclination_percentage < 0.10:
                    walking_speed -= 0.05
                elif inclination_percentage < 0.15:
                    walking_speed -= 0.14
                else:
                    walking_speed -= 0.24

            duration_time = data["length"] / walking_speed

        # biking speed model based on inclination
        elif data.get(Attributes.TRANSPORTATION_TYPE.value) == EdgeType.OSM_BIKE.value:
            biking_speed = 6.04  # base speed in m/s for flat terrain
            if include_inclination:
                if inclination_percentage < 0:
                    biking_speed += 0.24 * (  # 0.86 km/h in m/s
                        inclination_percentage * -100
                    )  # base speed adjusted for downhill
                else:
                    biking_speed -= 0.31 * (  # 1.14 km/h in m/s
                        inclination_percentage * 100
                    )  # base speed adjusted for uphill

                biking_speed = max(
                    biking_speed, 1.10
                )  # fallback to minimum walking speed

            duration_time = data["length"] / biking_speed

        new_attributes = {
            Attributes.INCLINATION_PERCENTAGE.value: inclination_percentage
        }
        if duration_time:
            duration_time += data.get(
                Attributes.CHANGE_PT_TIME.value, 0
            )  # add change time for public transportation in seconds
            new_attributes[Attributes.DURATION_TIME.value] = (
                duration_time / 60
            )  # time in minutes

            if include_solar_exposure_index:
                for time_of_day in [
                    config.DayTimeNames.MORNING,
                    config.DayTimeNames.NOON,
                    config.DayTimeNames.AFTERNOON,
                    config.DayTimeNames.EVENING,
                ]:
                    for vulnerability_group in config.vulnerability_groups:  # 1 to 5
                        solar_exposure_index = data[
                            Attributes.solar_exposure(time_of_day)
                        ]

                        new_attributes[
                            Attributes.DURATION_TIME.duration_time(
                                time_of_day, vulnerability_group
                            )
                        ] = (
                            (1 + (solar_exposure_index * vulnerability_group))
                            * duration_time
                        ) / 60  # time in minutes

        return (u, v, edge_key), new_attributes

    def _add_node_coordinates(self):
        """
        Adds the geometry of the nodes to the graph as node attributes. The geometry is created as a Point from the x and y coordinates of the nodes.

        Args:
            None

        Returns:
            None
        """
        geometries = gpd.GeoSeries(
            [shapely.Point(data["x"], data["y"]) for _, data in self.G.nodes(data=True)]
        ).set_crs(config.default_epsg)
        geoms = geometries.to_crs(
            config.wgs_84
        )  # for compatibility with Vienna approach
        updating = {
            node: {"geom": geom, "geometry": geometry}
            for node, geom, geometry in zip(self.G.nodes, geoms, geometries)
        }
        nx.set_node_attributes(self.G, updating)

    def __remove_motorway_edges(self):
        """
        Removes edges from the graph that are classified as motorways, as they are not suitable for walking or biking. After removing these edges, any nodes that become isolated (degree 0) are also removed.

        Args:
            None
        Returns:
            None
        """
        edges_to_remove = [
            (u, v, key)
            for u, v, key, data in self.G.edges(data=True, keys=True)
            if "motorway" in str(data.get("highway"))
        ]
        self.G.remove_edges_from(edges_to_remove)
        nodes_to_remove = [
            node for node, degree in dict(self.G.degree()).items() if degree == 0
        ]
        self.G.remove_nodes_from(nodes_to_remove)

        # remove all but the largest weakly connected component (all orphans are removed)
        largest_nodes = max(nx.weakly_connected_components(self.G), key=len)
        self.G = self.G.subgraph(largest_nodes).copy()

    def add_solar_exposure_index(self):
        """
        Adds solar exposure index to edges in the graph based on the DigitalSurfaceModel, DigitalGroundModel, building geometries, and Local Climate Zones. The solar exposure index is calculated for each edge at different times of the day (morning, noon, afternoon, evening) and for different vulnerability groups (1 to 5), and it is added as an edge attribute.

        Args:
            None
        Returns:
            None
        """
        gdf_edges = ox.graph_to_gdfs(self.G, nodes=False, edges=True)
        gdf_edges = gdf_edges[
            gdf_edges[Attributes.TRANSPORTATION_TYPE.value].isin(
                [
                    EdgeType.OSM_WALK.value,
                    EdgeType.OSM_BIKE.value,
                    EdgeType.OSM_TO_STATION.value,
                ]
            )
        ].copy()
        gdf_buildings = ox.features_from_place(
            self.location, tags={"building": True}
        ).to_crs(config.default_epsg)
        shadow_calculation = ShadowCalculation(
            self.dsm, self.dgm, gdf_buildings, self.lcz
        )

        times = DayTimes.as_list()
        time_names = DayTimeNames.as_list()

        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    shadow_calculation.calculate_solar_exposure_index,
                    gdf_edges["geometry"].to_crs(config.default_epsg),
                    pd.to_datetime(time_str)
                    .tz_localize(config.local_timezone)
                    .tz_convert(config.global_timezone),
                ): time_name
                for time_str, time_name in zip(times, time_names)
            }
            for future in tqdm.tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Calculating shadows for times",
            ):
                gdf_edges[Attributes.solar_exposure(futures[future])] = future.result()

        new_attributes = {
            edge_idx: {
                Attributes.solar_exposure(time_name): row[
                    Attributes.solar_exposure(time_name)
                ]
                for time_name in DayTimeNames.as_list()
            }
            for edge_idx, row in gdf_edges.iterrows()
        }

        nx.set_edge_attributes(self.G, new_attributes)


def create_lcz_gpd(lcz_path: str, location: str) -> gpd.GeoDataFrame:
    """
    Creates a GeoDataFrame containing the Local Climate Zone (LCZ) information for a given location. The LCZ information is extracted from a raster file, where each pixel value corresponds to an LCZ class. The function reads the raster file, masks it to the area of interest defined by the location, and converts the valid pixels into geometries with their corresponding LCZ classes.

    Args:
        lcz_path (str): The file path to the raster file containing the LCZ information.
        location (str): The location for which the LCZ information should be extracted.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the geometries and LCZ classes for the area of interest.
    """
    with rasterio.open(lcz_path) as src:
        polygon = ox.geocode_to_gdf(location).iloc[0]["geometry"]
        out_image, out_transform = mask(src, [polygon], crop=True)
        band = out_image[0]

        if src.nodata is not None:
            valid_mask = band != src.nodata
        else:
            valid_mask = ~np.isnan(band)

        records = []
        for geom, value in shapes(band, mask=valid_mask, transform=out_transform):
            records.append({"geometry": shape(geom), "lcz_class": int(value)})

        gdf_lcz = gpd.GeoDataFrame(records, crs=src.crs)
        gdf_lcz["lcz"] = gdf_lcz["lcz_class"].map(
            {
                1: "1",
                2: "2",
                3: "3",
                4: "4",
                5: "5",
                6: "6",
                7: "7",
                8: "8",
                9: "9",
                10: "10",
                11: "A",
                12: "B",
                13: "C",
                14: "D",
                15: "E",
                16: "F",
                17: "G",
            }
        )

        return gdf_lcz[["geometry", "lcz"]].to_crs(config.default_epsg)


if __name__ == "__main__":
    dsm = DigitalSurfaceModel(dsm_path=config.dsm_path)
    dgm = DigitalSurfaceModel(dsm_path=config.dgm_path)
    lcz = create_lcz_gpd(config.lcz_path, config.location)


    # creating the walking graph
    start_time = time.time()
    graph_walk = Graph(
        config.location,
        dsm,
        TransportationMode.WALK,
        lcz,
        include_solar_exposure_index=True,
        include_inclination=True,
        dgm=dgm,
    )
    graph_walk.store(config.graph_walk_pickle_path)
    walk_time = time.time() - start_time
    print(f"graph_walk done in {(walk_time):.2f} seconds.")

    # creating the cycling graph
    start_time = time.time()
    graph_bike = Graph(
        config.location,
        dsm,
        TransportationMode.BIKE,
        lcz,
        include_solar_exposure_index=True,
        include_inclination=True,
        dgm=dgm,
    )
    graph_bike.store(config.graph_bike_pickle_path)
    bike_time = time.time() - start_time
    print(f"graph_bike done in {(bike_time):.2f} seconds.")

    # creating the walking + public transportation graph
    start_time = time.time()
    graph_pt = Graph(
        config.location,
        dsm,
        TransportationMode.WALK_PT,
        lcz,
        include_solar_exposure_index=True,
        include_inclination=True,
        dgm=dgm,
    )
    graph_pt.store(config.graph_walk_pt_pickle_path)
    pt_time = time.time() - start_time
    print(f"graph_pt done in {(pt_time):.2f} seconds.")

    print("All times: ")
    print(f"graph_walk: {(walk_time):.2f} seconds.")
    print(f"graph_bike: {(bike_time):.2f} seconds.")
    print(f"graph_pt: {(pt_time):.2f} seconds.")
