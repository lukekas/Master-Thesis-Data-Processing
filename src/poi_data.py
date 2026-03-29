import osmnx as ox
import pandas as pd
import numpy as np
from tqdm import tqdm
import geopandas as gpd
import os

try:
    import config
except ModuleNotFoundError:
    from . import config

def get_osm_features_by_category(location: str, categories: list[str], input_dir: str=config.osm_tags_path, output_crs: str=config.default_epsg) -> gpd.GeoDataFrame:
    """
    Downloads OpenStreetMap (OSM) features for a specified location and categories, and returns them as a GeoDataFrame. 

    Args:
        location (str): The location for which to download OSM features (e.g., "Bamberg, Germany").
        categories (list[str]): A list of categories for which to download OSM features (e.g., ["healthcare", "services", "transport"]).
        input_dir (str): The directory where the input CSV files containing OSM tags for each category are located. Defaults to config.osm_tags_path.
        output_crs (str): The coordinate reference system to which the downloaded OSM features should be transformed. Defaults to config.default_epsg.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the downloaded OSM features for the specified location and categories.
    """
    all_gdfs = []
    polygon = ox.geocode_to_gdf(location).geometry.iloc[0].simplify(tolerance=0.001)
    for category in tqdm(categories, desc=f"Loading OSM features by category for {location}", total=len(categories)):
        df_input = pd.read_csv(f"{input_dir}/{category}.csv")
        df_input = df_input.groupby("key").agg({"value": list})
        df_input["value"] = df_input["value"].apply(lambda x: True if x == [np.nan] else x)
        filters = {idx: row["value"] for idx, row in df_input.iterrows()}

        gdf_features = ox.features_from_polygon(polygon, filters).to_crs(output_crs)
        gdf_features["category"] = category
        all_gdfs.append(gdf_features)

    gdf_all = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=output_crs)
    return gdf_all

if __name__ == "__main__":
    gdf_osm_features = get_osm_features_by_category(config.location, config.categories) # getting OSM features
    os.makedirs(os.path.dirname(config.osm_features_parquet_path), exist_ok=True)
    gdf_osm_features.to_parquet(config.osm_features_parquet_path) # storing the features