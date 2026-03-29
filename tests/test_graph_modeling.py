import os
import sys
import pickle
import pandas as pd
import geopandas as gpd
import shapely
import networkx as nx
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from graph_modeling import (
    Graph,
    EdgeType,
    TransportationMode,
    Attributes,
)
import config


def _make_graph_instance(G: nx.MultiDiGraph = None) -> Graph:
    """Create a Graph without triggering __init__ (which calls osmnx).

    Args:
        G (nx.MultiDiGraph, optional): The underlying NetworkX graph. Defaults to None.

    Returns:
        Graph: A Graph instance.
    """
    g = object.__new__(Graph)
    g.G = G if G is not None else nx.MultiDiGraph()
    g.location = "Somewhere in Germany"
    g.dsm = MagicMock()
    g.dgm = MagicMock()
    g.transportation_mode = TransportationMode.WALK
    g.include_solar_exposure_index = False
    g.lcz = gpd.GeoDataFrame({"lcz": []}, geometry=[], crs="EPSG:4326")
    return g


def _walk_edge(length: float) -> tuple:
    """Return a (u, v, key, data) edge tuple for a walk edge.

    Args:
        length (float): the length of the edge in meters, used to compute duration time.

    Returns: tuple: (u, v, key, data) edge tuple for a walk edge with the given length
    """
    return (
        1,
        2,
        0,
        {
            "length": length,
            Attributes.TRANSPORTATION_TYPE.value: EdgeType.OSM_WALK.value,
        },
    )


def _bike_edge(length: float) -> tuple:
    """Return a (u, v, key, data) edge tuple for a bike edge.

    Args:
        length (float): the length of the edge in meters, used to compute duration time.

    Returns: tuple: (u, v, key, data) edge tuple for a bike edge with the given length
    """
    return (
        1,
        2,
        0,
        {
            "length": length,
            Attributes.TRANSPORTATION_TYPE.value: EdgeType.OSM_BIKE.value,
        },
    )


class TestAttributesHelpers:
    """Tests for helper methods in the Attributes class that compute attribute values based on input parameters."""

    def test_duration_time_string(self):
        result = Attributes.duration_time(config.DayTimeNames.MORNING, 2)
        assert result == "time_morning_vuln2"

    def test_solar_exposure_with_daytimenames(self):
        result = Attributes.solar_exposure(config.DayTimeNames.NOON)
        assert result == "solar_exposure_noon"

    def test_solar_exposure_with_string(self):
        result = Attributes.solar_exposure("evening")
        assert result == "solar_exposure_evening"


class TestGetDurationTimeByEdge:
    """Tests for the get_duration_time_by_edge method."""

    def test_pt_edge_returns_empty_attrs(self):
        edge = (
            1,
            2,
            0,
            {
                "length": 100.0,
                Attributes.TRANSPORTATION_TYPE.value: EdgeType.PUBLIC_TRANSPORTATION.value,
                Attributes.DURATION_TIME.value: 3.0,
            },
        )
        key, attrs = Graph.get_duration_time_by_edge(edge, 10, 12, False)
        assert attrs == {}
        assert key == (1, 2, 0)

    def test_walk_edge_sets_inclination(self):
        edge = _walk_edge(length=100.0)
        altitude_u, altitude_v = 0, 5  # +5 % inclination
        _, attrs = Graph.get_duration_time_by_edge(edge, altitude_u, altitude_v, False)
        assert (
            pytest.approx(attrs[Attributes.INCLINATION_PERCENTAGE.value], rel=1e-6)
            == 0.05
        )

    def test_walk_edge_duration_time_in_minutes(self):
        edge = _walk_edge(length=134.0)
        _, attrs = Graph.get_duration_time_by_edge(edge, 0, 0, False)  # flat
        expected_minutes = 134.0 / config.base_walking_speed / 60
        assert (
            pytest.approx(attrs[Attributes.DURATION_TIME.value], rel=1e-4)
            == expected_minutes
        )

    def test_walk_steep_downhill_faster(self):
        edge_flat = _walk_edge(length=100.0)
        edge_down = _walk_edge(length=100.0)
        _, attrs_flat = Graph.get_duration_time_by_edge(edge_flat, 0, 0, False)
        _, attrs_down = Graph.get_duration_time_by_edge(edge_down, 20, 0, False)
        assert (
            attrs_down[Attributes.DURATION_TIME.value]
            < attrs_flat[Attributes.DURATION_TIME.value]
        )

    def test_walk_steep_uphill_slower(self):
        edge_flat = _walk_edge(length=100.0)
        edge_up = _walk_edge(length=100.0)
        _, attrs_flat = Graph.get_duration_time_by_edge(edge_flat, 0, 0, False)
        _, attrs_up = Graph.get_duration_time_by_edge(edge_up, 0, 20, False)
        assert (
            attrs_up[Attributes.DURATION_TIME.value]
            > attrs_flat[Attributes.DURATION_TIME.value]
        )

    def test_walk_none_altitude_uses_zero_difference(self):
        edge = _walk_edge(length=100.0)
        _, attrs_none = Graph.get_duration_time_by_edge(edge, None, None, False)
        _, attrs_flat = Graph.get_duration_time_by_edge(edge, 0, 0, False)
        assert (
            pytest.approx(attrs_none[Attributes.DURATION_TIME.value])
            == attrs_flat[Attributes.DURATION_TIME.value]
        )

    def test_walk_nan_altitude_uses_zero_difference(self):
        edge = _walk_edge(length=100.0)
        _, attrs_nan = Graph.get_duration_time_by_edge(
            edge, float("nan"), float("nan"), False
        )
        _, attrs_flat = Graph.get_duration_time_by_edge(edge, 0, 0, False)
        assert (
            pytest.approx(attrs_nan[Attributes.DURATION_TIME.value])
            == attrs_flat[Attributes.DURATION_TIME.value]
        )

    def test_bike_edge_sets_duration(self):
        edge = _bike_edge(length=100.0)
        _, attrs = Graph.get_duration_time_by_edge(edge, 0, 0, False)
        assert Attributes.DURATION_TIME.value in attrs

    def test_bike_downhill_faster_than_flat(self):
        flat = _bike_edge(length=1000.0)
        downhill = _bike_edge(length=1000.0)
        _, attrs_flat = Graph.get_duration_time_by_edge(flat, 0, 0, False)
        _, attrs_down = Graph.get_duration_time_by_edge(downhill, 50, 0, False)
        assert (
            attrs_down[Attributes.DURATION_TIME.value]
            < attrs_flat[Attributes.DURATION_TIME.value]
        )

    def test_bike_minimum_speed_applied(self):
        edge = _bike_edge(length=100.0)
        _, attrs = Graph.get_duration_time_by_edge(edge, 0, 200, False)
        min_duration = 100.0 / 1.10 / 60
        assert (
            pytest.approx(attrs[Attributes.DURATION_TIME.value], rel=1e-4)
            == min_duration
        )

    def test_change_pt_time_added_to_walk_edge(self):
        edge = (
            1,
            2,
            0,
            {
                "length": 100.0,
                Attributes.TRANSPORTATION_TYPE.value: EdgeType.OSM_TO_STATION.value,
                Attributes.CHANGE_PT_TIME.value: 120.0,  # 120 seconds
            },
        )
        _, attrs_with = Graph.get_duration_time_by_edge(edge, 0, 0, False)
        edge_no_change = (
            1,
            2,
            0,
            {
                "length": 100.0,
                Attributes.TRANSPORTATION_TYPE.value: EdgeType.OSM_TO_STATION.value,
            },
        )
        _, attrs_without = Graph.get_duration_time_by_edge(edge_no_change, 0, 0, False)
        assert (
            attrs_with[Attributes.DURATION_TIME.value]
            > attrs_without[Attributes.DURATION_TIME.value]
        )


class TestGetPtEdges:
    """Tests for the _get_pt_edges method."""

    def _make_stop_times(self, rows):
        """Build a minimal stop_times DataFrame."""
        df = pd.DataFrame(
            rows, columns=["trip_id", "stop_id", "arrival_time", "departure_time"]
        )
        df["arrival_time"] = pd.to_datetime(df["arrival_time"], format="%H:%M:%S")
        df["departure_time"] = pd.to_datetime(df["departure_time"], format="%H:%M:%S")
        return df

    def test_consecutive_stops_same_trip_create_edge(self):
        df = self._make_stop_times(
            [
                ("T1", "906", "08:00:00", "08:00:00"),
                ("T1", "931", "08:05:00", "08:05:00"),
            ]
        )
        g = _make_graph_instance()
        edges = g._get_pt_edges(df)
        assert len(edges) == 1
        assert edges[0][0] == "906"
        assert edges[0][1] == "931"

    def test_trip_change_does_not_create_edge(self):
        df = self._make_stop_times(
            [
                ("T1", "906", "08:00:00", "08:00:00"),
                ("T2", "931", "08:05:00", "08:05:00"),
            ]
        )
        g = _make_graph_instance()
        edges = g._get_pt_edges(df)
        assert len(edges) == 0

    def test_zero_duration_replaced_with_half_minute(self):
        df = self._make_stop_times(
            [
                ("T1", "906", "08:00:00", "08:00:00"),
                ("T1", "931", "08:00:00", "08:00:00"),
            ]
        )
        g = _make_graph_instance()
        edges = g._get_pt_edges(df)
        assert len(edges) == 1
        assert edges[0][2][Attributes.DURATION_TIME.value] == 0.5

    def test_negative_duration_excluded(self):
        # departure after arrival at next stop (e.g. data error)
        df = self._make_stop_times(
            [
                ("T1", "906", "08:00:00", "08:10:00"),
                ("T1", "931", "08:05:00", "08:05:00"),
            ]
        )
        g = _make_graph_instance()
        edges = g._get_pt_edges(df)
        assert all(e[2][Attributes.DURATION_TIME.value] > 0 for e in edges)

    def test_transportation_type_set(self):
        df = self._make_stop_times(
            [
                ("T1", "906", "08:00:00", "08:00:00"),
                ("T1", "931", "08:05:00", "08:05:00"),
            ]
        )
        g = _make_graph_instance()
        edges = g._get_pt_edges(df)
        assert (
            edges[0][2][Attributes.TRANSPORTATION_TYPE.value]
            == EdgeType.PUBLIC_TRANSPORTATION.value
        )


class TestLoadStopTimes:
    """Tests for the _load_stop_times method."""

    def _base_stop_times(self):
        return pd.DataFrame(
            {
                "stop_id": ["ZOB", "Spinnerei", "Markusplatz"],
                "trip_id": ["T1", "T1", "T2"],
                "arrival_time": ["08:00:00", "08:05:00", "25:00:00"],
                "departure_time": ["08:00:00", "08:05:00", "25:01:00"],
            }
        )

    def test_hour_over_24_is_normalised(self):
        df_stops = pd.DataFrame({"stop_id": ["ZOB", "Spinnerei", "Markusplatz"]})
        g = _make_graph_instance()
        with patch("pandas.read_csv", return_value=self._base_stop_times()):
            result = g._load_stop_times("wird-eh-gemockt", df_stops)
        assert all(result["arrival_time"].dt.hour < 24)

    def test_only_matching_stops_kept(self):
        df_stops = pd.DataFrame(
            {"stop_id": ["ZOB"]}
        )  # only stop ZOB matches stop_times
        g = _make_graph_instance()
        with patch("pandas.read_csv", return_value=self._base_stop_times()):
            result = g._load_stop_times("wird-eh-gemockt", df_stops)
        assert set(result["stop_id"]) == {"ZOB"}


class TestGetFrequencyForStops:
    """Tests for the _get_frequency_for_stops method"""

    def _make_data(self):
        df_stops = pd.DataFrame({"stop_id": ["S1", "S2"]})
        df_stop_times = pd.DataFrame(
            {
                "stop_id": ["S1", "S1", "S1", "S2"],
                "trip_id": ["123", "123", "123", "456"],
                "arrival_time": pd.to_datetime(
                    ["08:00:00", "08:20:00", "08:40:00", "09:00:00"], format="%H:%M:%S"
                ),
                "departure_time": pd.to_datetime(
                    ["08:00:00", "08:20:00", "08:40:00", "09:00:00"], format="%H:%M:%S"
                ),
            }
        )
        df_trips = pd.DataFrame({"trip_id": ["123", "456"], "route_id": ["R1", "R2"]})
        df_routes = pd.DataFrame(
            {"route_id": ["R1", "R2"], "route_short_name": ["1", "2"]}
        )
        return df_stops, df_stop_times, df_trips, df_routes

    def test_returns_dataframe(self):
        g = _make_graph_instance()
        df_stops, df_stop_times, df_trips, df_routes = self._make_data()
        result = g._get_frequency_for_stops(
            df_stops, df_stop_times, df_trips, df_routes
        )
        assert isinstance(result, pd.DataFrame)

    def test_frequency_column_present(self):
        g = _make_graph_instance()
        df_stops, df_stop_times, df_trips, df_routes = self._make_data()
        result = g._get_frequency_for_stops(
            df_stops, df_stop_times, df_trips, df_routes
        )
        assert "frequency" in result.columns

    def test_frequency_value_for_20min_interval(self):
        g = _make_graph_instance()
        df_stops, df_stop_times, df_trips, df_routes = self._make_data()
        result = g._get_frequency_for_stops(
            df_stops, df_stop_times, df_trips, df_routes
        )
        assert result.loc["S1", "frequency"] == 1200.0


class TestRemoveMotorwayEdges:
    """Tests for the __remove_motorway_edges method"""

    def _graph_with_motorway(self):
        G = nx.MultiDiGraph()
        G.add_node("bli")
        G.add_node("bla")
        G.add_node("blub")
        G.add_edge("bli", "bla", key=0, highway="motorway")
        G.add_edge("bla", "blub", key=0, highway="residential")
        G.add_edge("blub", "bla", key=0, highway="residential")
        return G

    def test_motorway_edges_removed(self):
        g = _make_graph_instance(self._graph_with_motorway())
        g._Graph__remove_motorway_edges()
        for _, _, data in g.G.edges(data=True):
            assert "motorway" not in str(data.get("highway"))

    def test_isolated_nodes_removed(self):
        g = _make_graph_instance(self._graph_with_motorway())
        g._Graph__remove_motorway_edges()
        assert "bli" not in g.G.nodes

    def test_largest_component_kept(self):
        G = nx.MultiDiGraph()
        G.add_edge("bli", "bla", key=0, highway="residential")
        G.add_edge("bla", "bli", key=0, highway="residential")
        G.add_edge("bla", "blub", key=0, highway="residential")
        G.add_edge("blub", "bla", key=0, highway="residential")
        G.add_edge("bla", "blub", key=0, highway="residential")
        G.add_edge("blub", "bla", key=0, highway="residential")
        g = _make_graph_instance(G)
        g._Graph__remove_motorway_edges()
        assert set(g.G.nodes) == {"bli", "bla", "blub"}


class TestAddSolarExposureIndex:
    """Tests for the add_solar_exposure_index method."""

    def _make_gdf_edges(self):
        """Return a GeoDataFrame resembling osmnx edge output with one walk and one PT edge."""
        index = pd.MultiIndex.from_tuples([(1, 2, 0), (2, 3, 0)], names=["u", "v", "key"])
        return gpd.GeoDataFrame(
            {
                Attributes.TRANSPORTATION_TYPE.value: [
                    EdgeType.OSM_WALK.value,
                    EdgeType.PUBLIC_TRANSPORTATION.value,
                ],
                "length": [100.0, 200.0],
                "geometry": [
                    shapely.LineString([(400000, 5500000), (400100, 5500000)]),
                    shapely.LineString([(400100, 5500000), (400200, 5500000)]),
                ],
            },
            index=index,
            geometry="geometry",
            crs=config.default_epsg,
        )

    def _make_graph(self):
        G = nx.MultiDiGraph()
        G.add_node(1)
        G.add_node(2)
        G.add_node(3)
        G.add_edge(1, 2, key=0, **{Attributes.TRANSPORTATION_TYPE.value: EdgeType.OSM_WALK.value, "length": 100.0})
        G.add_edge(2, 3, key=0, **{Attributes.TRANSPORTATION_TYPE.value: EdgeType.PUBLIC_TRANSPORTATION.value, "length": 200.0})
        return G

    def _setup_mocks(self, mock_graph_to_gdfs, mock_features_from_place, mock_shadow_cls,
                     mock_executor_cls, mock_as_completed, mock_tqdm, gdf):
        mock_graph_to_gdfs.return_value = gdf

        mock_buildings = MagicMock()
        mock_buildings.to_crs.return_value = mock_buildings
        mock_features_from_place.return_value = mock_buildings

        mock_shadow_cls.return_value = MagicMock()

        filtered_index = gdf[gdf[Attributes.TRANSPORTATION_TYPE.value].isin(
            [EdgeType.OSM_WALK.value, EdgeType.OSM_BIKE.value, EdgeType.OSM_TO_STATION.value]
        )].index

        futures = []
        for i in range(4):
            f = MagicMock()
            f.result.return_value = pd.Series([0.1 * (i + 1)], index=filtered_index)
            futures.append(f)

        mock_executor = MagicMock()
        mock_executor.submit.side_effect = futures
        ctx = MagicMock()
        ctx.__enter__.return_value = mock_executor
        ctx.__exit__.return_value = False
        mock_executor_cls.return_value = ctx

        mock_as_completed.return_value = iter(futures)
        mock_tqdm.side_effect = lambda x, **kw: x

        return futures

    @patch("graph_modeling.tqdm.tqdm")
    @patch("graph_modeling.as_completed")
    @patch("graph_modeling.ProcessPoolExecutor")
    @patch("graph_modeling.ShadowCalculation")
    @patch("graph_modeling.ox.features_from_place")
    @patch("graph_modeling.ox.graph_to_gdfs")
    def test_walk_edges_get_all_solar_exposure_attributes(
        self, mock_graph_to_gdfs, mock_features_from_place, mock_shadow_cls,
        mock_executor_cls, mock_as_completed, mock_tqdm
    ):
        gdf = self._make_gdf_edges()
        self._setup_mocks(mock_graph_to_gdfs, mock_features_from_place, mock_shadow_cls,
                          mock_executor_cls, mock_as_completed, mock_tqdm, gdf)
        G = self._make_graph()
        g = _make_graph_instance(G)

        g.add_solar_exposure_index()

        edge_data = G.edges[1, 2, 0]
        for time_name in config.DayTimeNames.as_list():
            assert Attributes.solar_exposure(time_name) in edge_data

    @patch("graph_modeling.tqdm.tqdm")
    @patch("graph_modeling.as_completed")
    @patch("graph_modeling.ProcessPoolExecutor")
    @patch("graph_modeling.ShadowCalculation")
    @patch("graph_modeling.ox.features_from_place")
    @patch("graph_modeling.ox.graph_to_gdfs")
    def test_pt_edges_do_not_get_solar_exposure_attributes(
        self, mock_graph_to_gdfs, mock_features_from_place, mock_shadow_cls,
        mock_executor_cls, mock_as_completed, mock_tqdm
    ):
        gdf = self._make_gdf_edges()
        self._setup_mocks(mock_graph_to_gdfs, mock_features_from_place, mock_shadow_cls,
                          mock_executor_cls, mock_as_completed, mock_tqdm, gdf)
        G = self._make_graph()
        g = _make_graph_instance(G)

        g.add_solar_exposure_index()

        pt_edge_data = G.edges[2, 3, 0]
        for time_name in config.DayTimeNames.as_list():
            assert Attributes.solar_exposure(time_name) not in pt_edge_data

    @patch("graph_modeling.tqdm.tqdm")
    @patch("graph_modeling.as_completed")
    @patch("graph_modeling.ProcessPoolExecutor")
    @patch("graph_modeling.ShadowCalculation")
    @patch("graph_modeling.ox.features_from_place")
    @patch("graph_modeling.ox.graph_to_gdfs")
    def test_shadow_calculation_called_once_per_time_of_day(
        self, mock_graph_to_gdfs, mock_features_from_place, mock_shadow_cls,
        mock_executor_cls, mock_as_completed, mock_tqdm
    ):
        gdf = self._make_gdf_edges()
        self._setup_mocks(mock_graph_to_gdfs, mock_features_from_place, mock_shadow_cls,
                          mock_executor_cls, mock_as_completed, mock_tqdm, gdf)
        G = self._make_graph()
        g = _make_graph_instance(G)

        g.add_solar_exposure_index()

        executor = mock_executor_cls.return_value.__enter__.return_value
        assert executor.submit.call_count == len(config.DayTimeNames.as_list())

    @patch("graph_modeling.tqdm.tqdm")
    @patch("graph_modeling.as_completed")
    @patch("graph_modeling.ProcessPoolExecutor")
    @patch("graph_modeling.ShadowCalculation")
    @patch("graph_modeling.ox.features_from_place")
    @patch("graph_modeling.ox.graph_to_gdfs")
    def test_solar_exposure_attribute_values_match_shadow_calculation_result(
        self, mock_graph_to_gdfs, mock_features_from_place, mock_shadow_cls,
        mock_executor_cls, mock_as_completed, mock_tqdm
    ):
        gdf = self._make_gdf_edges()
        futures = self._setup_mocks(mock_graph_to_gdfs, mock_features_from_place, mock_shadow_cls,
                                    mock_executor_cls, mock_as_completed, mock_tqdm, gdf)
        G = self._make_graph()
        g = _make_graph_instance(G)

        g.add_solar_exposure_index()

        edge_data = G.edges[1, 2, 0]
        for i, time_name in enumerate(config.DayTimeNames.as_list()):
            expected = futures[i].result().iloc[0]
            assert pytest.approx(edge_data[Attributes.solar_exposure(time_name)]) == expected


class TestStore:
    def test_creates_pickle_file(self, tmp_path):
        G = nx.MultiDiGraph()
        G.add_node("lukas123", x=1.0, y=2.0)
        g = _make_graph_instance(G)
        out = str(tmp_path / "graph.p")
        g.store(out)
        assert os.path.exists(out)

    def test_pickle_round_trips_graph(self, tmp_path):
        G = nx.MultiDiGraph()
        G.add_node("lukas123", x=1.0, y=2.0)
        g = _make_graph_instance(G)
        out = str(tmp_path / "graph.p")
        g.store(out)
        with open(out, "rb") as f:
            loaded = pickle.load(f)
        assert "lukas123" in loaded.nodes
