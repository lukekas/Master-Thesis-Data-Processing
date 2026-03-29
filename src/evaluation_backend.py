import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import geopandas as gpd
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
from zoneinfo import ZoneInfo
import osmnx as ox
import shapely
import pandas as pd

from .graph_modeling import create_lcz_gpd

from . import config
from .digital_surface_model import DigitalSurfaceModel
from .shadow_analysis import ShadowCalculation

parquet_file = "data/evaluation/shadow_evaluation.parquet"
shadow_analysis_result_column = "shadow_analysis_result"

images_folder = "data/evaluation/images/"

lcz = create_lcz_gpd("data/lcz/lcz_v3.tif", location="Bamberg, Germany")


def get_image_information(path):
    def minutes_to_decimal(dms, ref):
        degrees = dms[0]
        minutes = dms[1]
        seconds = dms[2]
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        if ref in {"S", "W"}:
            decimal = -decimal
        return decimal

    img = Image.open(path)
    exif = img.getexif()
    exif_table = {TAGS.get(k, k): v for k, v in exif.items()}
    gps_table = {GPSTAGS.get(k, k): v for k, v in exif.get_ifd(0x8825).items()}
    latitude = minutes_to_decimal(gps_table["GPSLatitude"], gps_table["GPSLatitudeRef"])
    longitude = minutes_to_decimal(
        gps_table["GPSLongitude"], gps_table["GPSLongitudeRef"]
    )

    dt_raw = exif_table.get("DateTimeOriginal") or exif_table.get("DateTime")
    if not dt_raw:
        photo_datetime = None
    else:
        photo_datetime = (
            datetime.strptime(dt_raw, "%Y:%m:%d %H:%M:%S")
            .replace(tzinfo=ZoneInfo(config.local_timezone))
            .astimezone(ZoneInfo(config.global_timezone))
        )

    return latitude, longitude, photo_datetime


def create_gdf_result(images: list[str]) -> gpd.GeoDataFrame:
    data = {
        image: get_image_information(os.path.join(images_folder, image))
        + (
            image,
            None,
        )
        for image in images
    }
    gdf_result = gpd.GeoDataFrame(
        data.values(),
        geometry=gpd.points_from_xy(
            [d[1] for d in data.values()], [d[0] for d in data.values()]
        ),
        columns=[
            "latitude",
            "longitude",
            "photo_datetime",
            "image_name",
            shadow_analysis_result_column,
        ],
        crs="EPSG:4326",
    ).to_crs(epsg=25832)
    return gdf_result


gdf_result: gpd.GeoDataFrame = ...
dsm: DigitalSurfaceModel = ...
dgm: DigitalSurfaceModel = ...
gdf_buildings: gpd.GeoDataFrame = ...


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gdf_result, dsm, dgm, gdf_buildings
    # load gdf_result from parquet file if it exists, otherwise create it from images and store it in parquet file for next time
    images = [
        f for f in os.listdir(images_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if os.path.exists(parquet_file):
        gdf_result = gpd.read_parquet(parquet_file)
        different_images = set(list(gdf_result["image_name"])) ^ set(images)
        gdf_new_result = create_gdf_result(different_images)
        gdf_result = pd.concat([gdf_result, gdf_new_result], ignore_index=True)
    else:
        gdf_result = create_gdf_result(images)
        gdf_result.to_parquet(parquet_file)

    dsm = DigitalSurfaceModel(config.dsm_path)
    dgm = DigitalSurfaceModel(config.dgm_path)
    gdf_buildings = ox.features_from_place(
        config.location, tags={"building": True}
    ).to_crs(epsg=25832)

    yield
    # store gdf_result in parquet file for next time
    gdf_result.to_parquet(parquet_file)


app = FastAPI(lifespan=lifespan)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Welcome to the evaluation backend!",
        "number_of_images": len(gdf_result),
    }


@app.get("/images")
async def get_images():
    df_images = gdf_result[
        [
            "latitude",
            "longitude",
            "photo_datetime",
            "image_name",
            shadow_analysis_result_column,
        ]
    ].astype(object)
    df_images = df_images.where(df_images.notna(), None)

    return {"images": df_images.to_dict(orient="records")}


@app.get("/current_image")
async def get_current_image(image_name: str = None):
    if image_name is not None:
        current_image = gdf_result[gdf_result["image_name"] == image_name]
        if current_image.empty:
            return {"error": f"Image {image_name} not found."}
        current_image = current_image.iloc[0]
    else:
        current_image = gdf_result[gdf_result[shadow_analysis_result_column].isna()]
        if current_image.empty:
            current_image = gdf_result.iloc[0]
        else:
            current_image = current_image.iloc[0]

    if current_image.name == 0:
        next_image = gdf_result.iloc[1]["image_name"]
        previous_image = None
    elif current_image.name == len(gdf_result) - 1:
        next_image = None
        previous_image = gdf_result.iloc[-2]["image_name"]
    else:
        next_image = gdf_result.loc[current_image.name + 1]["image_name"]
        previous_image = gdf_result.loc[current_image.name - 1]["image_name"]

    return {
        "latitude": current_image["latitude"],
        "longitude": current_image["longitude"],
        "photo_datetime": current_image["photo_datetime"],
        "image_name": current_image["image_name"],
        "file_path": f"/image_file/{current_image['image_name']}",
        "shadow_path": f"/shadow_map/{current_image['image_name']}",
        "next_image": next_image,
        "previous_image": previous_image,
    }


@app.get("/image_file/{image_name}")
async def get_image_file(image_name: str):
    image_path = os.path.join(images_folder, image_name)
    if not os.path.exists(image_path):
        return {"error": f"Image {image_name} not found."}
    with open(image_path, "rb") as f:
        return Response(content=f.read(), media_type="image/jpeg")


@app.get("/shadow_map/{image_name}")
async def get_shadow_map(image_name: str):
    image_path = os.path.join(images_folder, image_name)
    lat, lon, photo_datetime = get_image_information(image_path)
    print(lat, lon, photo_datetime)
    gdf_point = gpd.GeoDataFrame(
        geometry=[shapely.geometry.Point(lon, lat)], crs="EPSG:4326"
    ).to_crs(epsg=25832)
    x, y = gdf_point.geometry.iloc[0].coords[0]
    point = shapely.geometry.Point(x, y)
    gdf_buildings_local = gdf_buildings[
        gdf_buildings.geometry.distance(point) < 100
    ].copy()  # filter for buildings around the point
    shadow_calculation_local = ShadowCalculation(dsm, dgm, gdf_buildings_local, lcz)
    m = shadow_calculation_local._get_gdf_shadows(photo_datetime).explore()
    gdf_point.explore(m=m, color="red", marker_kwds={"radius": 10})

    # Render map HTML directly
    html_content = m.get_root().render()

    # Return HTML response
    return Response(content=html_content, media_type="text/html")


@app.post("/submit_evaluation/{image_name}")
async def submit_evaluation(image_name: str, evaluation: float):
    if image_name not in gdf_result["image_name"].values:
        return {"error": f"Image {image_name} not found."}
    gdf_result.loc[
        gdf_result["image_name"] == image_name, shadow_analysis_result_column
    ] = evaluation
    gdf_result.to_parquet(parquet_file)
    return {"message": f"Evaluation for {image_name} submitted successfully."}


@app.get("/evaluation_results_map")
async def evaluation_results_map():
    colors = [
        (
            "green"
            if percentage == 1
            else "red" if percentage == 0 else "grey" if percentage == 0.5 else "blue"
        )
        for percentage in gdf_result[shadow_analysis_result_column]
    ]
    m = gdf_result.explore(
        color=colors,
        tiles="https://sgx.geodatenzentrum.de/wmts_topplus_open/tile/1.0.0/web_grau/default/WEBMERCATOR/{z}/{y}/{x}.png",
        attr='Map data: &copy; <a href="http://www.govdata.de/dl-de/by-2-0">dl-de/by-2-0</a>',
    )
    html_content = m.get_root().render()
    return Response(content=html_content, media_type="text/html")


@app.get("/evaluation_results")
async def evaluation_results():
    gdf_result.value_counts(shadow_analysis_result_column)
    return {
        "value_counts": gdf_result[shadow_analysis_result_column]
        .value_counts()
        .to_dict(),
        "performance": gdf_result[shadow_analysis_result_column].mean(),
    }
