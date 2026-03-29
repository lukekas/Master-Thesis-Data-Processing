import os
import sys

import geopandas as gpd
import networkx as nx
import pytest
import shapely

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hexagon_grid import (
    build_average_time_column_names,
    create_hexagonal_grid,
    materialize_time_columns,
)
import config


def _square_polygon(size: float = 1000.0) -> shapely.Polygon:
    """Return a simple square polygon in the default projected CRS.

    Args:
        size: The length of the sides of the square in the same units as the CRS (default is meters).

    Returns:
        shapely.Polygon: A square polygon with corners at (0, 0), (size, 0), (size, size), and (0, size).
    """
    return shapely.box(0, 0, size, size)


class TestBuildAverageTimeColumnNames:
    """Tests for method build_average_time_column_names."""

    def test_returns_list(self):
        result = build_average_time_column_names("walk")
        assert isinstance(result, list)

    def test_contains_base_columns_for_each_n(self):
        result = build_average_time_column_names("bike")
        for n in config.number_of_closest_pois:
            assert f"average_time_bike_{n}" in result

    def test_contains_vulnerability_columns(self):
        result = build_average_time_column_names("walk")
        daytimes = config.DayTimeNames.as_list()
        for daytime in daytimes:
            for vuln in config.vulnerability_groups:
                for n in config.number_of_closest_pois:
                    assert f"average_time_walk_{daytime}_vuln{vuln}_{n}" in result

    def test_no_duplicate_columns(self):
        result = build_average_time_column_names("walk_PT")
        assert len(result) == len(set(result))

    def test_transportation_mode_in_all_column_names(self):
        mode = "bike"
        result = build_average_time_column_names(mode)
        assert all(mode in col for col in result)


class TestCreateHexagonalGrid:
    """Tests for method create_hexagonal_grid."""

    def test_returns_geodataframe(self):
        polygon = _square_polygon()
        result = create_hexagonal_grid(polygon, side_length=200)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_has_geometry_column(self):
        polygon = _square_polygon()
        result = create_hexagonal_grid(polygon, side_length=200)
        assert "geometry" in result.columns

    def test_has_centroids_column(self):
        polygon = _square_polygon()
        result = create_hexagonal_grid(polygon, side_length=200)
        assert "centroids" in result.columns

    def test_all_hexagons_intersect_polygon(self):
        polygon = _square_polygon()
        result = create_hexagonal_grid(polygon, side_length=200)
        assert all(result.geometry.intersects(polygon))

    def test_index_is_reset(self):
        polygon = _square_polygon()
        result = create_hexagonal_grid(polygon, side_length=200)
        assert list(result.index) == list(range(len(result)))

    def test_larger_side_length_produces_fewer_hexagons(self):
        polygon = _square_polygon(2000.0)
        small = create_hexagonal_grid(polygon, side_length=100)
        large = create_hexagonal_grid(polygon, side_length=400)
        assert len(small) > len(large)

    def test_crs_is_set(self):
        polygon = _square_polygon()
        result = create_hexagonal_grid(
            polygon, side_length=200, crs=config.default_epsg
        )
        assert result.crs is not None
        assert result.crs.to_epsg() is not None

    def test_hexagons_are_polygons(self):
        polygon = _square_polygon()
        result = create_hexagonal_grid(polygon, side_length=200)
        assert all(isinstance(geom, shapely.Polygon) for geom in result.geometry)

    def test_hexagon_has_six_vertices(self):
        polygon = _square_polygon()
        result = create_hexagonal_grid(polygon, side_length=200)
        # exterior ring has 7 coords (first == last); interior is empty for regular hexagon
        first_hex = result.geometry.iloc[0]
        assert len(first_hex.exterior.coords) - 1 == 6

    def test_nonempty_grid_for_valid_polygon(self):
        polygon = _square_polygon(500.0)
        result = create_hexagonal_grid(polygon, side_length=100)
        assert len(result) > 0

    def test_centroids_are_point_geometries(self):
        polygon = _square_polygon()
        result = create_hexagonal_grid(polygon, side_length=200)
        assert all(isinstance(p, shapely.Point) for p in result["centroids"])


def _graph_with_time_edge() -> nx.MultiDiGraph:
    """Return a tiny graph with one edge that has a 'time' attribute.

    Returns:
        nx.MultiDiGraph: A MultiDiGraph with one edge that has a 'time' attribute.
    """
    G = nx.MultiDiGraph()
    G.add_edge(1, 2, key=0, time=5.0)
    return G


def _graph_with_existing_time_column(daytime: str, vuln: int) -> nx.MultiDiGraph:
    """Return a graph where the specific time column already exists on the edge.

    Args:
        daytime: The daytime to use in the column name (e.g., "morning").
        vuln: The vulnerability group to use in the column name (e.g., 1).

    Returns:
        nx.MultiDiGraph: A MultiDiGraph with one edge that has both 'time' and the specific time column.
    """
    G = nx.MultiDiGraph()
    col = f"time_{daytime}_vuln{vuln}"
    G.add_edge(1, 2, key=0, time=5.0, **{col: 99.0})
    return G


class TestMaterializeTimeColumns:
    """Tests for method materialize_time_columns."""

    def test_adds_all_expected_columns(self):
        G = _graph_with_time_edge()
        materialize_time_columns(G)
        daytimes = config.DayTimeNames.as_list()
        for daytime in daytimes:
            for vuln in config.vulnerability_groups:
                col = f"time_{daytime}_vuln{vuln}"
                for _, _, data in G.edges(data=True):
                    assert col in data

    def test_copies_base_time_when_column_missing(self):
        G = _graph_with_time_edge()
        materialize_time_columns(G)
        daytime = config.DayTimeNames.as_list()[0]
        col = f"time_{daytime}_vuln1"
        for _, _, data in G.edges(data=True):
            assert data[col] == pytest.approx(5.0)

    def test_preserves_existing_column_value(self):
        daytime = config.DayTimeNames.as_list()[0]
        vuln = 1
        G = _graph_with_existing_time_column(daytime, vuln)
        materialize_time_columns(G)
        col = f"time_{daytime}_vuln{vuln}"
        for _, _, data in G.edges(data=True):
            assert data[col] == pytest.approx(99.0)

    def test_does_not_modify_base_time(self):
        G = _graph_with_time_edge()
        materialize_time_columns(G)
        for _, _, data in G.edges(data=True):
            assert data["time"] == pytest.approx(5.0)

    def test_works_with_multiple_edges(self):
        G = nx.MultiDiGraph()
        G.add_edge(1, 2, key=0, time=3.0)
        G.add_edge(2, 3, key=0, time=7.0)
        materialize_time_columns(G)
        daytime = config.DayTimeNames.as_list()[0]
        col = f"time_{daytime}_vuln1"
        times = sorted(data[col] for _, _, data in G.edges(data=True))
        assert times == pytest.approx([3.0, 7.0])
