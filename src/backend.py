import json
import io

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import geopandas as gpd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
import numpy as np
from fastapi.staticfiles import StaticFiles
from . import config

census_files = [census_file["name"] for census_file in config.evaluation_census_files]

with open(config.hexagon_grid_output_geojson_path) as f:
    geojson_data = json.load(f)

gdf_hexagon = gpd.read_parquet(config.hexagon_grid_output_parquet_path)

result_column = config.evaluation_result_column

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/geojson")
async def root():
    return geojson_data


@app.get("/api/evaluation/{configuration}/{topic}/census.svg")
async def analysis(configuration: str, topic: str):
    """
    Returns a scatter plot as an SVG image showing the relationship between the travel time (x-axis) and the specified census topic (y-axis) for each hexagon. The travel time is calculated based on the specified configuration, and the census topic is obtained from the evaluation results. The function also checks if the provided topic and configuration are valid before performing the analysis.

    Args:
        configuration (str): The configuration for which to calculate travel time.
        topic (str): The census topic to analyze.

    Returns:
        Response: An SVG image showing the scatter plot.
    """
    if topic not in census_files:
        return {"error": f"Topic {topic} not found. Available topics: {census_files}"}

    if configuration not in gdf_hexagon.columns:
        return {"error": f"Configuration {configuration} not found."}

    gdf_result = gpd.read_parquet(config.evaluation_raster_path(topic))
    gdf_myhexagon = gdf_hexagon.merge(gdf_result, left_index=True, right_index=True)

    configuration_column_simplified = f"{configuration}_simplified"
    gdf_myhexagon[configuration_column_simplified] = gdf_myhexagon[configuration].apply(
        lambda x: sum(x.values()) / len(x)
    )

    y_label = " ".join(topic.split("_"))
    plt.rcParams["svg.fonttype"] = "none"

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.scatter(
        gdf_myhexagon[configuration_column_simplified], gdf_myhexagon[result_column]
    )

    ax.set_xlabel("travel time")
    y_label = y_label.replace("foreigners", "non-citizens")
    ax.set_ylabel(y_label)
    ax.set_title(f"scatter plot of travel time and {y_label}")
    ax.grid(True)

    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)

    return Response(content=buf.getvalue(), media_type="image/svg+xml")


@app.get("/api/evaluation/{configuration}/{topic}/correlation")
async def correlation(configuration: str, topic: str):
    """
    Calculates the Pearson and Spearman correlation coefficients between the travel time (calculated based on the specified configuration) and the specified census topic for each hexagon.

    Args:
        configuration (str): The configuration for which to calculate travel time.
        topic (str): The census topic to analyze.

    Returns:
        dict: A dictionary containing the Pearson and Spearman correlation coefficients and their corresponding p-values.
    """
    if topic not in census_files:
        return {"error": f"Topic {topic} not found. Available topics: {census_files}"}

    if configuration not in gdf_hexagon.columns:
        return {"error": f"Configuration {configuration} not found."}

    gdf_result = gpd.read_parquet(config.evaluation_raster_path(topic))
    gdf_myhexagon = gdf_hexagon.merge(gdf_result, left_index=True, right_index=True)

    configuration_column_simplified = f"{configuration}_simplified"
    gdf_myhexagon[configuration_column_simplified] = gdf_myhexagon[configuration].apply(
        lambda x: sum(x.values()) / len(x)
    )

    nas = np.logical_or(
        gdf_myhexagon[result_column].isna(),
        gdf_myhexagon[configuration_column_simplified].isna(),
    )  # truth array for all rows with a missing value in either column, to ignore them in the correlation calculation
    pearson_r, pearson_p = pearsonr(
        gdf_myhexagon[result_column][~nas],
        gdf_myhexagon[configuration_column_simplified][~nas],
    )
    spearman_r, spearman_p = spearmanr(
        gdf_myhexagon[result_column][~nas],
        gdf_myhexagon[configuration_column_simplified][~nas],
    )

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


@app.get("/api/evaluation/{configuration}/average_travel_time")
async def average_travel_time(
    configuration: str,
    culture: float = 1,
    education: float = 1,
    healthcare: float = 1,
    outdoor: float = 1,
    physical: float = 1,
    restaurant: float = 1,
    services: float = 1,
    supplies: float = 1,
    transport: float = 1,
):
    """
    Calculates the average travel time per hexagon and per inhabitant based on the specified configuration and weights for different categories. The function first checks if the provided configuration is valid, then applies the weights to the travel time values in the hexagon grid, and finally calculates the average travel time per hexagon and per inhabitant using the population data.

    Args:
        configuration (str): The configuration for which to calculate travel time.
        culture (float): The weight for the culture category.
        education (float): The weight for the education category.
        healthcare (float): The weight for the healthcare category.
        outdoor (float): The weight for the outdoor category.
        physical (float): The weight for the physical category.
        restaurant (float): The weight for the restaurant category.
        services (float): The weight for the services category.
        supplies (float): The weight for the supplies category.
        transport (float): The weight for the transport category.

    Returns:
        dict: A dictionary containing the average travel time per hexagon and per inhabitant.
    """
    if configuration not in gdf_hexagon.columns:
        return {"error": f"Configuration {configuration} not found."}

    gdf_myhexagon = gdf_hexagon.copy()

    weights = {
        "culture": culture,
        "education": education,
        "healthcare": healthcare,
        "outdoor": outdoor,
        "physical": physical,
        "restaurant": restaurant,
        "services": services,
        "supplies": supplies,
        "transport": transport,
    }

    gdf_myhexagon[configuration] = gdf_myhexagon[configuration].apply(lambda x: sum([value * weights.get(key, 1) for key, value in x.items()]) / (sum(weights.values()) + len(x) - len(weights)))

    average_travel_time_per_hexagon = gdf_myhexagon[configuration].mean()

    gdf_population = gpd.read_parquet(config.evaluation_raster_path("population"))
    gdf_myhexagon = gdf_myhexagon.merge(
        gdf_population, left_index=True, right_index=True
    )
    average_travel_time_per_inhabitant = (
        gdf_myhexagon[configuration] * gdf_myhexagon[result_column]
    ).sum() / gdf_myhexagon[result_column].sum()

    return {
        "average_travel_time_per_hexagon": average_travel_time_per_hexagon,
        "average_travel_time_per_inhabitant": average_travel_time_per_inhabitant,
    }


app.mount("/", StaticFiles(directory=config.frontend_path, html=True), name="static")
