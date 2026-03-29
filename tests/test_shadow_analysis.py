import os
import sys
import pandas as pd
import geopandas as gpd
import shapely
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from shadow_analysis import ShadowCalculation


def _simple_building_gdf(heights=(15.0, 20.0)) -> gpd.GeoDataFrame:
    """Minimal buildings GeoDataFrame in EPSG:4326 with a 'height' column.

    Args:
        heights: iterable of building heights in meters, one per building. Default is (15.0, 20.0).

    Returns: GeoDataFrame with one row per building, a 'height' column, and simple box geometries.
    """
    polys = [
        shapely.geometry.box(10.920, 49.910, 10.921, 49.911),
        shapely.geometry.box(10.922, 49.912, 10.923, 49.913),
    ]
    return gpd.GeoDataFrame(
        {"height": list(heights)},
        geometry=polys,
        crs="EPSG:4326",
    )


def _shadow_gdf(polygon: shapely.geometry.Polygon) -> gpd.GeoDataFrame:
    """Return a shadow GeoDataFrame with a single polygon in EPSG:4326.

    Args:
        polygon: the geometry of the shadow to return

    Returns:
        GeoDataFrame: A GeoDataFrame containing the shadow geometry.
    """
    return gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")


def _make_shadow_calc(gdf_buildings: gpd.GeoDataFrame = None) -> ShadowCalculation:
    """Construct ShadowCalculation bypassing __init__ (which calls pybdshadow).

    Args:
        gdf_buildings: The GeoDataFrame of buildings for which to calculate shadows. Defaults to a simple set of buildings.

    Returns:
        ShadowCalculation: A ShadowCalculation instance.
    """
    sc = object.__new__(ShadowCalculation)
    sc.crs = "EPSG:4326"
    sc.dsm = MagicMock()
    sc.dgm = MagicMock()
    sc.store = {}
    sc.gdf_buildings = (
        gdf_buildings if gdf_buildings is not None else _simple_building_gdf()
    )
    sc.lcz = gpd.GeoDataFrame({"lcz": []}, geometry=[], crs="EPSG:4326")
    return sc


class TestPreprocessBuildings:
    """Tests for the _preprocess_buildings method."""

    def _make_dsm(self, altitude: float) -> MagicMock:
        """DSM/DGM mock that writes a fixed altitude into the requested column.

        Args:
            altitude: The fixed altitude to write into the requested column.

        Returns:
            MagicMock: A mock object that simulates the DSM/DGM behavior.
        """

        def _fill_altitude(df, **kw):
            df = df.copy()
            df[kw["altitude_col"]] = altitude
            return df

        mock = MagicMock()
        mock.get_altitude_for_dataframe.side_effect = _fill_altitude
        return mock

    def test_calls_bd_preprocess(self):
        gdf = _simple_building_gdf()
        sc = _make_shadow_calc(gdf)
        with patch(
            "shadow_analysis.pybdshadow.bd_preprocess", return_value=gdf
        ) as mock_bp:
            sc._preprocess_buildings(set_height=False)
        mock_bp.assert_called_once()

    def test_set_height_false_skips_altitude_lookup(self):
        gdf = _simple_building_gdf()
        sc = _make_shadow_calc(gdf)
        with patch("shadow_analysis.pybdshadow.bd_preprocess", return_value=gdf):
            sc._preprocess_buildings(set_height=False)
        sc.dsm.get_altitude_for_dataframe.assert_not_called()
        sc.dgm.get_altitude_for_dataframe.assert_not_called()

    def test_set_height_true_calls_dsm_and_dgm(self):
        gdf = _simple_building_gdf()
        sc = _make_shadow_calc(gdf)
        sc.dsm = self._make_dsm(altitude=25.0)
        sc.dgm = self._make_dsm(altitude=5.0)
        with patch("shadow_analysis.pybdshadow.bd_preprocess", return_value=gdf):
            sc._preprocess_buildings(set_height=True)
        sc.dsm.get_altitude_for_dataframe.assert_called_once()
        sc.dgm.get_altitude_for_dataframe.assert_called_once()

    def test_height_clipped_to_10m_minimum(self):
        gdf = _simple_building_gdf()
        shadow_calculation = _make_shadow_calc(gdf)

        shadow_calculation.dsm.get_altitude_for_dataframe.side_effect = (
            lambda df, x_col, y_col, altitude_col: df.assign(**{altitude_col: 3.0})
        )
        shadow_calculation.dgm.get_altitude_for_dataframe.side_effect = (
            lambda df, x_col, y_col, altitude_col: df.assign(**{altitude_col: 0.0})
        )

        captured = {}

        def capture_preprocess(buildings_gdf):
            captured["height"] = buildings_gdf["height"].values.copy()
            return buildings_gdf

        with patch(
            "shadow_analysis.pybdshadow.bd_preprocess", side_effect=capture_preprocess
        ):
            shadow_calculation._preprocess_buildings(set_height=True)

        assert all(h >= 10.0 for h in captured["height"])

    def test_empty_geometries_dropped_when_set_height(self):
        polys = [
            shapely.geometry.box(10.920, 49.910, 10.921, 49.911),
            shapely.geometry.Polygon(),
        ]
        gdf = gpd.GeoDataFrame(
            {"height": [15.0, 20.0]}, geometry=polys, crs="EPSG:4326"
        )
        shadow_calculation = _make_shadow_calc(gdf)
        shadow_calculation.dsm.get_altitude_for_dataframe.side_effect = (
            lambda df, x_col, y_col, altitude_col: df.assign(**{altitude_col: 20.0})
        )
        shadow_calculation.dgm.get_altitude_for_dataframe.side_effect = (
            lambda df, x_col, y_col, altitude_col: df.assign(**{altitude_col: 5.0})
        )

        captured = {}

        def capture_preprocess(buildings_gdf):
            captured["gdf"] = buildings_gdf.copy()
            return buildings_gdf

        with patch(
            "shadow_analysis.pybdshadow.bd_preprocess", side_effect=capture_preprocess
        ):
            shadow_calculation._preprocess_buildings(set_height=True)

        assert all(not geom.is_empty for geom in captured["gdf"].geometry)

    def test_result_stored_on_instance(self):
        gdf = _simple_building_gdf()
        shadow_calculation = _make_shadow_calc(gdf)
        sentinel = _simple_building_gdf()
        with patch("shadow_analysis.pybdshadow.bd_preprocess", return_value=sentinel):
            shadow_calculation._preprocess_buildings(set_height=False)
        assert shadow_calculation.gdf_buildings is sentinel


class TestGetGdfShadows:
    """Tests for the _get_gdf_shadows method."""

    def test_calls_pybdshadow_when_not_cached(self):
        sc = _make_shadow_calc()
        dt = pd.Timestamp("2025-07-08 10:00:00", tz="UTC")
        fake_shadows = _shadow_gdf(shapely.geometry.box(10.920, 49.910, 10.925, 49.915))

        with patch(
            "shadow_analysis.pybdshadow.bdshadow_sunlight", return_value=fake_shadows
        ) as mock_fn:
            result = sc._get_gdf_shadows(dt)

        mock_fn.assert_called_once()
        assert result is fake_shadows

    def test_caches_result_in_store(self):
        sc = _make_shadow_calc()
        dt = pd.Timestamp("2025-07-08 10:00:00", tz="UTC")
        fake_shadows = _shadow_gdf(shapely.geometry.box(10.920, 49.910, 10.925, 49.915))

        with patch(
            "shadow_analysis.pybdshadow.bdshadow_sunlight", return_value=fake_shadows
        ):
            sc._get_gdf_shadows(dt)

        assert dt in sc.store
        assert sc.store[dt] is fake_shadows

    def test_returns_cached_result_without_recalculating(self):
        sc = _make_shadow_calc()
        dt = pd.Timestamp("2025-07-08 10:00:00", tz="UTC")
        cached = _shadow_gdf(shapely.geometry.box(10.920, 49.910, 10.925, 49.915))
        sc.store[dt] = cached

        with patch("shadow_analysis.pybdshadow.bdshadow_sunlight") as mock_fn:
            result = sc._get_gdf_shadows(dt, use_store=True)

        mock_fn.assert_not_called()
        assert result is cached

    def test_bypasses_cache_when_use_store_false(self):
        sc = _make_shadow_calc()
        dt = pd.Timestamp("2025-07-08 10:00:00", tz="UTC")
        cached = _shadow_gdf(shapely.geometry.box(10.920, 49.910, 10.925, 49.915))
        sc.store[dt] = cached
        fresh = _shadow_gdf(shapely.geometry.box(10.930, 49.920, 10.935, 49.925))

        with patch("shadow_analysis.pybdshadow.bdshadow_sunlight", return_value=fresh):
            result = sc._get_gdf_shadows(dt, use_store=False)

        assert result is fresh


class TestCalculateSolarExposureIndex:
    """Tests for the calculate_solar_exposure_index method."""

    def test_solar_exposure_is_one_minus_shadow_coverage(self):
        shadow_calculation = _make_shadow_calc()
        shadow_pct = pd.Series([0.3, 0.6, 0.0])
        shadow_calculation.calculate_street_shadow_coverage_percentage_batch = (
            MagicMock(return_value=shadow_pct)
        )
        streets = MagicMock()
        dt = pd.Timestamp("2025-07-08 10:00:00", tz="UTC")
        result = shadow_calculation.calculate_solar_exposure_index(streets, dt)
        expected = 1 - shadow_pct
        pd.testing.assert_series_equal(result, expected)

    def test_fully_shadowed_street_gives_zero_exposure(self):
        shadow_calculation = _make_shadow_calc()
        shadow_pct = pd.Series([1.0])
        shadow_calculation.calculate_street_shadow_coverage_percentage_batch = (
            MagicMock(return_value=shadow_pct)
        )
        streets = MagicMock()
        dt = pd.Timestamp("2025-07-08 10:00:00", tz="UTC")
        result = shadow_calculation.calculate_solar_exposure_index(streets, dt)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_unshadowed_street_gives_full_exposure(self):
        shadow_calculation = _make_shadow_calc()
        shadow_pct = pd.Series([0.0])
        shadow_calculation.calculate_street_shadow_coverage_percentage_batch = (
            MagicMock(return_value=shadow_pct)
        )
        streets = MagicMock()
        dt = pd.Timestamp("2025-07-08 10:00:00", tz="UTC")
        result = shadow_calculation.calculate_solar_exposure_index(streets, dt)
        assert result.iloc[0] == pytest.approx(1.0)
