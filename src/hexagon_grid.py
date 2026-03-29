import osmnx as ox
import geopandas as gpd
import pandas as pd
import math
import shapely
import pickle
import numpy as np
import networkx as nx
import tqdm
from concurrent.futures import ProcessPoolExecutor
try:
    from .config import vulnerability_groups, number_of_closest_pois
    from . import config
except ImportError:
    from config import vulnerability_groups, number_of_closest_pois
    import config

daytimes = config.DayTimeNames.as_list()

_graphs = {}
_gdf_pois = None


def initialize_worker(
    graphs: dict[str, nx.MultiDiGraph], gdf_pois: gpd.GeoDataFrame
) -> None:
    global _graphs, _gdf_pois
    _graphs = graphs
    _gdf_pois = gdf_pois


def build_average_time_column_names(transportation_mode: str) -> list[str]:
    """
    Build all average-time column names for one transportation mode.

    Args:
        transportation_mode (str): The transportation mode for which to build the column names (e.g. "walk", "bike", "walk_PT").

    Returns:
        list[str]: A list of column names in the format "average_time_{transportation_mode}_{n}" or "average_time_{transportation_mode}_{daytime}_vuln{vulnerability_group}_{n}".
    """
    return [
        f"average_time_{transportation_mode}_{n}"
        for n in number_of_closest_pois
    ] + [
        f"average_time_{transportation_mode}_{daytime}_vuln{vulnerability_group}_{n}"
        for daytime in daytimes
        for vulnerability_group in vulnerability_groups
        for n in number_of_closest_pois
    ]


# function to create a hexagonal grid, from Vienna paper
def create_hexagonal_grid(
    polygon: shapely.Polygon, side_length: float = 200, crs: str = config.default_epsg
) -> gpd.GeoDataFrame:
    """
    Create a hexagonal grid inside a given polygon with specified side length.

    Parameters:
        polygon (shapely.geometry.Polygon): input polygon.
        side_length (float): length of each hexagon side in meters.

    Returns:
        GeoDataFrame: hexagonal grid clipped to the polygon.
    """
    # spacing between hexagon centers
    dx = math.sqrt(3) * side_length  # horizontal distance between centers
    dy = 1.5 * side_length  # vertical distance between centers

    # bounding box of the polygon
    minx, miny, maxx, maxy = polygon.bounds

    # generate hexagon centers
    hex_centers = []
    y = miny
    row = 0
    while y <= maxy + dy:
        x = minx
        if row % 2 == 1:  # offset odd rows
            x += (math.sqrt(3) * side_length) / 2

        while x <= maxx + dx:
            hex_centers.append(shapely.Point(x, y))
            x += dx
        y += dy
        row += 1

    # create hexagons around each center
    hexagons = []
    for center in hex_centers:
        hexagon = shapely.Polygon(
            [
                (
                    center.x + side_length * math.cos(angle),
                    center.y + side_length * math.sin(angle),
                )
                for angle in [math.pi / 6 + i * math.pi / 3 for i in range(6)]
            ]
        )
        hexagons.append(hexagon)

    # convert to GeoDataFrame
    grid = gpd.GeoDataFrame(geometry=hexagons, crs=crs)
    grid["centroids"] = hex_centers

    # clip the grid to the input polygon
    clipped_grid = grid[grid.intersects(polygon)].copy()
    clipped_grid.reset_index(drop=True, inplace=True)

    return clipped_grid


def compute_poi_network_distances(hexagon_row: pd.Series) -> dict:
    """
    Computes the average network distance from the given hexagon to the closest POIs per category for each transportation mode, daytime and vulnerability group.

    Args:
        hexagon_row (pd.Series): A row from the hexagon GeoDataFrame containing the graph node for each transportation mode.

    Returns:
        dict: A dictionary with keys in the format "average_time_{transportation_mode}_{daytime}_vuln{vulnerability_group}_{n}" and values as dictionaries with category as key and average time to the closest POIs in that category as value.
    """
    if not _graphs or _gdf_pois is None:
        raise RuntimeError("Worker process was not initialized with graphs and POIs")

    result = {}
    for transportation_mode in _graphs.keys():
        column_name_graph_node = f"graph_node_{transportation_mode}"
        source_node = hexagon_row[column_name_graph_node]

        # calculate network distances in one batch using single_source_dijkstra
        for vulnerability_group in vulnerability_groups:
            for daytime in daytimes + [None]:  # include None for overall average times
                if daytime is None and vulnerability_group != 1:
                    continue  # skip redundant calculations for overall average times

                # calculate network distances from hexagon to all target nodes
                weight_column = (
                    "time"
                    if daytime is None
                    else f"time_{daytime}_vuln{vulnerability_group}"
                )
                network_distances = nx.single_source_dijkstra_path_length(
                    _graphs[transportation_mode],
                    source=source_node,
                    weight=weight_column,
                    cutoff=None,
                )
                df_network_distances = (
                    pd.Series(network_distances, name="distance")
                    .rename_axis("node")
                    .reset_index()
                )
                df_network_distances = df_network_distances.merge(
                    _gdf_pois[[column_name_graph_node, "category"]],
                    left_on="node",
                    right_on=column_name_graph_node,
                    how="inner",
                )

                # calculate average time per category of nodes with 1, 2 and 20 smallest distances
                for n in number_of_closest_pois:
                    average_times = (
                        df_network_distances.groupby("category")["distance"]
                        .nsmallest(n)
                        .groupby(level=0)
                        .mean()
                    )

                    if daytime is None:
                        result[f"average_time_{transportation_mode}_{n}"] = (
                            average_times.to_dict()
                        )
                    else:
                        result[
                            f"average_time_{transportation_mode}_{daytime}_vuln{vulnerability_group}_{n}"
                        ] = average_times.to_dict()

    return result


def materialize_time_columns(G: nx.MultiDiGraph) -> None:
    """
    Materializes the time columns for each daytime and vulnerability group in the graph by copying the base "time" column to the specific time columns if they do not already exist. This ensures that there is a time value for each edge for all combinations of daytime and vulnerability group, which is necessary for calculating network distances based on different time attributes.

    Args:
        G (nx.MultiDiGraph): The graph for which to materialize the time columns.

    Returns:
        None, but modifies the graph in place by adding the necessary time columns.
    """
    edges = G.edges(data=True, keys=True)
    for daytime in daytimes:
        for vulnerability_group in vulnerability_groups:
            time_column_name = config.time_edge_attribute_column_name(daytime, vulnerability_group)
            time_temp = {
                (u, v, key): (
                    d["time"] if time_column_name not in d else d[time_column_name]
                )
                for u, v, key, d in edges
            }
            nx.set_edge_attributes(G, time_temp, time_column_name)


def calculate_census_data_per_hexagon(gdf_hexagon: gpd.GeoDataFrame, files: list[dict]) -> None:
    """
    Creates the raster parameters for the evaluation.

    Args:
        gdf_hexagon: GeoDataFrame with hexagonal grid
        files: list of dicts with keys "name", "path", "input_column_name" and "crs" for the csv files containing the zensus data

    Returns:
        None, but stores the calculated values for evaluation in parquet files
    """
    result_column = config.evaluation_result_column
    for file in tqdm.tqdm(files, desc="Processing census files", total=len(files)):
        crs = file["crs"]
        gdf_hexagon = gdf_hexagon.to_crs(crs)
        boundary = gdf_hexagon.union_all().buffer(100)

        # convert csv file to GeoDataFrame
        df = pd.read_csv(file["path"], sep=file.get("separator", ";"))
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x_mp_100m"], df["y_mp_100m"]), crs=crs)
        del df

        gdf = gdf[gdf.within(boundary)].copy() # remove all geometries not being in Bamberg
        gdf["geometry"] = gdf["geometry"].buffer(50, cap_style=3) # create sqaures from the centroid
        source_values = gdf[file["input_column_name"]].astype(str)
        source_values = source_values.str.replace(",", ".", regex=False)
        source_values = source_values.str.replace("–", "", regex=False)
        gdf[result_column] = pd.to_numeric(source_values, errors="coerce") # convert column to number
        gdf = gdf[gdf[result_column].notna()].copy() # remove all geometries with no value for the result column, as they do not contribute to the weighted average

        gdf["square_geometry"] = gdf["geometry"].copy(deep=True) # copy geometry to another column for preserving it after spatial join
        gdf_joined = gdf_hexagon[["geometry"]].copy(deep=True)
        gdf_joined["index"] = gdf_joined.index
        gdf_result = gdf_hexagon[["geometry"]].copy(deep=True)
        gdf_joined = gdf_joined.sjoin(gdf[[result_column, "geometry", "square_geometry"]], how="left", predicate="intersects") # spatially join to get all sqares from the raster that belong to a hexagon
        gdf_joined["overlapping_area"] = gdf_joined["geometry"].intersection(gdf_joined["square_geometry"]).area # calculate how much the square covers the hexagon for weighted average
        gdf_joined["weighted_value"] = gdf_joined[result_column] * gdf_joined["overlapping_area"] # calculate the weighted value for each square
        grouped_sum = gdf_joined.groupby("index", as_index=False)[["weighted_value", "overlapping_area"]].sum()
        grouped_sum[result_column] = grouped_sum["weighted_value"].div(grouped_sum["overlapping_area"].where(grouped_sum["overlapping_area"] > 0, np.nan)) # calculate the weighted average for each hexagon
        gdf_result = gdf_result.reset_index(drop=False) # reset index to have the index as a column for merging
        gdf_result = gdf_result.merge(grouped_sum[[result_column, "index"]], on="index") # add the calculated value to the hexagon geodataframe
        gdf_result.to_parquet(config.evaluation_raster_path(file["name"])) # store the calculated values for evaluation in parquet file


def import_graphs(graph_paths: dict[str, str]) -> dict[str, nx.MultiDiGraph]:
    """
    Imports the graphs for the different transportation modes from the specified paths and projects them to the default CRS.

    Args:
        graph_paths (dict): Dictionary with transportation mode as key and path to the graph pickle file as value

    Returns:
        dict: Dictionary with transportation mode as key and the imported and projected graph as value
    """
    graphs = {}
    for transportation_mode, path in graph_paths.items():
        with open(path, "rb") as f:
            G = pickle.load(f)
        G = ox.project_graph(G, to_crs=config.default_epsg)
        graphs[transportation_mode] = G
    return graphs

def import_pois(path: str) -> gpd.GeoDataFrame:
    """
    Imports the POIs from the specified path and converts non-point geometries to centroids.

    Args:
        path (str): The file path to the POIs parquet file.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the POIs with geometries as centroids for non-point features.
    """
    gdf_pois = gpd.read_parquet(path)
    gdf_pois["geometry"] = np.where(
        gdf_pois.type != "Point", gdf_pois["geometry"].centroid, gdf_pois["geometry"]
    )  # convert non-point geometries to centroids
    return gdf_pois


if __name__ == "__main__":
    # get boundary of Bamberg
    boundary = ox.geocode_to_gdf(config.location).to_crs(config.default_epsg).iloc[0]["geometry"]

    # separate bamberg in a hexagon grid
    gdf_hexagon_grid = create_hexagonal_grid(
        boundary
    )
    gdf_hexagon_grid["index"] = gdf_hexagon_grid.index

    # import POIs
    gdf_pois = import_pois(config.osm_features_parquet_path)

    # import graphs
    _graphs = import_graphs({
        "bike": config.graph_bike_pickle_path,
        "walk": config.graph_walk_pickle_path,
        "walk_PT": config.graph_walk_pt_pickle_path,
    })

    for transportation_mode in _graphs.keys():
        # find nearest graph nodes to hexagon centroids
        nodes_x = [point.x for point in gdf_hexagon_grid["centroids"]]
        nodes_y = [point.y for point in gdf_hexagon_grid["centroids"]]
        
        used_graph = _graphs[transportation_mode] if transportation_mode != "walk_PT" else _graphs["walk"] # for walk_PT, use walk graph since bus stations have waiting times
        gdf_hexagon_grid[f"graph_node_{transportation_mode}"] = ox.nearest_nodes(
            used_graph, nodes_x, nodes_y
        )
        gdf_pois[f"graph_node_{transportation_mode}"] = ox.nearest_nodes(
            used_graph, gdf_pois["geometry"].x, gdf_pois["geometry"].y
        )

        materialize_time_columns(_graphs[transportation_mode])

        # initialize columns for average times
        new_columns = {
            column_name: None
            for column_name in build_average_time_column_names(transportation_mode)
        }

        gdf_hexagon_grid = pd.concat(
            [gdf_hexagon_grid, pd.DataFrame(new_columns, index=gdf_hexagon_grid.index)],
            axis=1,
        )

    _gdf_pois = gdf_pois # set global gdf_pois for access in the compute_poi_network_distances function
    with ProcessPoolExecutor(
        initializer=initialize_worker,
        initargs=(_graphs, _gdf_pois),
    ) as executor:
        futures = {
            idx: executor.submit(compute_poi_network_distances, hexagon_row) # for a selected hexagon, find the 20 closest POIs per category based on network distance
            for idx, hexagon_row in gdf_hexagon_grid.iterrows()
        }  # parallelization for selected hexagons
        for idx, future in tqdm.tqdm(
            futures.items(), total=len(futures), desc="Processing hexagons"
        ):
            intermediary_result = future.result()
            for key, value in intermediary_result.items():
                gdf_hexagon_grid.at[idx, key] = value

    # store graph
    columns = (
        ["geometry"]
        + [
            column_name
            for mode in _graphs.keys()
            for column_name in build_average_time_column_names(mode)
        ]
    )
    gdf_hexagon_grid = gdf_hexagon_grid[columns].copy() # filter to only keep the columns needed for visualization and evaluation

    gdf_hexagon_grid.to_parquet(config.hexagon_grid_output_parquet_path)
    gdf_hexagon_grid.to_crs(config.wgs_84).to_file(
        config.hexagon_grid_output_geojson_path, driver="GeoJSON"
    )

    calculate_census_data_per_hexagon(gdf_hexagon_grid, config.evaluation_census_files) # create evaluation data for the hexagons based on census data
