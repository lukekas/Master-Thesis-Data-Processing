import os
import subprocess
from . import config
import geopandas as gpd
import numpy as np

if __name__ == "__main__":
    result = {}
    for walking_speed in np.arange(1.14, 1.54, 0.1): # iterate over walking speeds and for each walking speed, run the graph modeling and hexagon grid scripts to get the average travel time per hexagon for each configuration, and store the results in a dictionary
        environment = os.environ.copy()
        environment["BASE_WALKING_SPEED"] = str(walking_speed)

        subprocess.run(["uv", "run", "python", "-m" "src.graph_modeling"], check=True, env=environment)
        subprocess.run(["uv", "run", "python", "-m" "src.hexagon_grid"], check=True)
        df_result = gpd.read_parquet(config.hexagon_grid_output_parquet_path)

        result[str(walking_speed)] = {}

        for configuration in ["average_time_walk_20", "average_time_walk_PT_20","average_time_walk_morning_vuln3_20" ,"average_time_walk_PT_evening_vuln5_20"]:
            average_travel_time_per_hexagon = df_result[configuration].apply(lambda x: sum(x.values()) / len(x)).mean()
            result[str(walking_speed)][configuration] = average_travel_time_per_hexagon

    print(result)