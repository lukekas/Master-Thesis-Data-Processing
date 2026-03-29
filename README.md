# Master's thesis "A geoinformatics model for the 15-minute city area under heat stress"
## Introduction
This repository contains scripts and data for my master's thesis' data processing. The thesis aims at integrating the two approaches https://github.com/johanneshbr7/accessibility-analysis-isocalors/ and https://doi.org/10.5194/agile-giss-6-2-2025. 

## Important files and folders
- `data`: contains data:
    - `tags`: list of tags from public transportation approach
- `dsm`/`dgm`: digital surface model and digital ground model for Bamberg, downloaded from https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=dgm1
- `evaluation`: Jupyter notebooks containing evaluation scripts
- `src`: contains code for the actual execution
    - `backend.py`: webserver hosting the backend
    - `config.py`: storage for variables that are used in multiple other files
    - `digital_surface_model.py`: contains class DigitalSurfaceModel (crucial for height calculation)
    - `dsm_download.py`: download of files based on a .meta4 file
    - `evaluation_backend.py`: webserver hosting the backend for the evaluation
    - `graph_modeling.py`: holds the Graph class and code for producing graphs for biking, walking, and walking with public transportation
    - `hexagon_grid.py`: calculation of the 15-minute city characteristics per hexagon grid cell based on previously created graphs
    - `poi_data.py`: script for getting the necessary place of interest data from OpenStreetMap
    - `shadow_analysis.py`: holds class ShadowCalculation for shadow geometry calculation with pydbshadow
- `heidelberg-reproduction/ShadowCalculationHeidelberg.ipynb`: shadow calculation for Heidelberg based on ShadowCalculation class from `shadow_analysis.py`

## Usage

### Installation
Installation of packages with [uv](https://docs.astral.sh/uv/):
```bash
uv sync
```

### Execution of the scripts
```bash
uv run uv run python -m src.poi_data
uv run python -m src.graph_modeling
uv run python -m src.hexagon_grid
uv run fastapi run src/backend.py --host=0.0.0.0 --port 8000
```

Thereby, the POIs are downloaded, the graphs created, and the hexagonal analysis is performed. Lastly, the webserver is started.


### Execution with Docker

Data needs to be mounted into the Docker Image. Here is an example for Bamberg:

```bash
docker run -it -p 9000:80 -v ./data/vienna/dsm:/app/data/dsm/downloads -v ./data/vienna/dgm:/app/data/dgm/downloads -v ./data/vienna/gtfs:/app/data/GTFS -v ./data/vienna/cache:/app/cache -e CITY="Bamberg, Germany" backend
```

### Tests
The implemented unit tests can be run with

```bash
uv run pytest tests
```
