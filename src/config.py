from enum import Enum
import os


class DayTimeNames(Enum):
    MORNING = "morning"
    NOON = "noon"
    AFTERNOON = "afternoon"
    EVENING = "evening"

    @classmethod
    def as_list(cls) -> list[str]:
        return [member.value for member in cls]


class DayTimes(Enum):
    MORNING = "2025-07-08 10:00:00"
    NOON = "2025-07-08 13:00:00"
    AFTERNOON = "2025-07-08 16:00:00"
    EVENING = "2025-07-08 19:00:00"

    @classmethod
    def as_list(cls) -> list[str]:
        return [member.value for member in cls]


vulnerability_groups = list(range(1, 6))

number_of_closest_pois = [1, 2, 20]

categories = [
    "healthcare",
    "services",
    "transport",
    "outdoor",
    "supplies",
    "restaurant",
    "culture",
    "education",
    "physical",
]

location = os.getenv("CITY", "Bamberg, Germany")

default_epsg = os.getenv("DEFAULT_EPSG", "EPSG:25832")

osm_epsg = os.getenv("OSM_EPSG", "EPSG:4326")

wgs_84 = "EPSG:4326"

evaluation_census_files = [
    {
        "name": "percentage_of_foreigners",
        "path": "data/evaluation/Zensus2022_Anteil_Auslaender_100m-Gitter.csv",
        "input_column_name": "AnteilAuslaender",
        "crs": "EPSG:3035",
    },
    {
        "name": "percentage_of_people_over_65",
        "path": "data/evaluation/Zensus2022_Anteil_ueber_65_100m-Gitter.csv",
        "input_column_name": "AnteilUeber65",
        "crs": "EPSG:3035",
    },
    {
        "name": "percentage_of_people_under_18",
        "path": "data/evaluation/Zensus2022_Anteil_unter_18_100m-Gitter.csv",
        "input_column_name": "AnteilUnter18",
        "crs": "EPSG:3035",
    },
    {
        "name": "population",
        "path": "data/evaluation/Zensus2022_Bevoelkerungszahl_100m-Gitter.csv",
        "input_column_name": "Einwohner",
        "crs": "EPSG:3035",
    },
]

hexagon_grid_output_parquet_path = "./results/hexagon_grid_bamberg.parquet"
hexagon_grid_output_geojson_path = "./results/hexagon_grid_bamberg.geojson"

graph_walk_pickle_path = "./results/walk.p"
graph_bike_pickle_path = "./results/bike.p"
graph_walk_pt_pickle_path = "./results/walk_PT.p"

osm_features_parquet_path = "./data/osm_features/bamberg_osm_features.parquet"

evaluation_result_column = "value_result"

gtfs_path = os.getenv("GTFS_PATH", "data/GTFS")
dgm_path = os.getenv("DGM_PATH", "data/dgm/downloads")
dsm_path = os.getenv("DSM_PATH", "data/dsm/downloads")
frontend_path = os.getenv(
    "FRONTEND_PATH", "/home/lukas/projects/masterthesis/frontend/dist/spa"
)
osm_tags_path = os.getenv("OSM_TAGS_PATH", "data/tags")

base_walking_speed = float(os.getenv("BASE_WALKING_SPEED", "1.34"))
lcz_path = os.getenv("LCZ_PATH", "data/lcz/lcz_v3.tif")

local_timezone = "Europe/Berlin"
global_timezone = "UTC"

def evaluation_raster_path(topic: str) -> str:
    """Returns the path to the evaluation raster for the given topic. The topic should be one of the names defined in the evaluation_census_files list.

    Args:
        topic (str): The name of the topic, e.g. "percentage_of_foreigners", "percentage_of_people_over_65", "percentage_of_people_under_18", "population".

    Returns:
        str: The path to the evaluation raster for the given topic, e.g. "data/evaluation/evaluation_raster_percentage_of_foreigners.parquet".
    """
    return f"data/evaluation/evaluation_raster_{topic}.parquet"


def time_edge_attribute_column_name(daytime: str, vulnerability_group: int) -> str:
    """Returns the name of the edge attribute column for the given daytime and vulnerability group. The daytime should be one of the values defined in the DayTimeNames enum, and the vulnerability group should be an integer between 1 and 5.

    Args:
        daytime (str): The name of the daytime, e.g. "morning", "noon", "afternoon", "evening".
        vulnerability_group (int): The vulnerability group, e.g. 1, 2, 3, 4, 5.

    Returns:
        str: The name of the edge attribute column for the given daytime and vulnerability group, e.g. "time_morning_vuln1", "time_noon_vuln2", "time_afternoon_vuln3", "time_evening_vuln4".
    """
    return f"time_{daytime}_vuln{vulnerability_group}"
