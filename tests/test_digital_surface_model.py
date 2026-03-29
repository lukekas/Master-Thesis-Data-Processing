import os
import sys
import numpy as np
import pandas as pd
import shapely
import pytest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.digital_surface_model import DigitalSurfaceModel


def _make_mock_src(bounds):
    """Return a minimal mock rasterio dataset for a given (left, bottom, right, top) bounds."""
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    mock_src = MagicMock()
    mock_src.crs = CRS.from_epsg(25832)
    mock_src.bounds = bounds
    band = np.arange(100, dtype=np.float32).reshape(10, 10) # simple example band
    mock_src.read.return_value = band
    mock_src.transform = from_bounds(*bounds, width=10, height=10)
    mock_src.__enter__ = lambda s: s
    mock_src.__exit__ = MagicMock(return_value=False)
    return mock_src


def _make_dsm(tmp_path, tif_files):
    """
    Construct a DigitalSurfaceModel whose _set_up_index reads mock rasterio sources.

    `tif_files` is a list of (filename, bounds) tuples.
    """
    mock_srcs = {fname: _make_mock_src(bounds) for fname, bounds in tif_files}

    def fake_listdir(_path):
        return list(mock_srcs.keys())

    def fake_rasterio_open(path):
        fname = os.path.basename(path)
        return mock_srcs[fname]

    with patch("src.digital_surface_model.os.listdir", side_effect=fake_listdir), \
         patch("src.digital_surface_model.rasterio.open", side_effect=fake_rasterio_open):
        dsm = DigitalSurfaceModel(str(tmp_path))

    # store mocks for later use in tests that call rasterio.open again
    dsm._mock_srcs = mock_srcs
    return dsm


class TestSetUpIndex:
    """Tests for the _set_up_index method"""
    def test_index_contains_one_entry_per_tif(self, tmp_path):
        tif_files = [
            ("tile_a.tif", (0.0, 0.0, 1000.0, 1000.0)),
            ("tile_b.tif", (1000.0, 0.0, 2000.0, 1000.0)),
        ]
        dsm = _make_dsm(tmp_path, tif_files)
        assert len(dsm._dsm_index) == 2

    def test_non_tif_files_are_ignored(self, tmp_path):
        mock_src = _make_mock_src((0.0, 0.0, 1000.0, 1000.0))
        mock_src.__enter__ = lambda s: s
        mock_src.__exit__ = MagicMock(return_value=False)

        with patch("src.digital_surface_model.os.listdir", return_value=["tile.tif", "readme.txt", "data.csv"]), \
             patch("src.digital_surface_model.rasterio.open", return_value=mock_src):
            dsm = DigitalSurfaceModel(str(tmp_path))

        assert len(dsm._dsm_index) == 1

    def test_geometry_is_box_matching_bounds(self, tmp_path):
        bounds = (0.0, 0.0, 1000.0, 1000.0)
        tif_files = [("tile.tif", bounds)]
        dsm = _make_dsm(tmp_path, tif_files)
        expected = shapely.geometry.box(*bounds)
        assert dsm._dsm_index.iloc[0].geometry.equals(expected)


class TestGetDsmFile:
    """Tests for the _get_dsm_file method"""
    @pytest.fixture
    def dsm(self, tmp_path):
        return _make_dsm(tmp_path, [
            ("tile_left.tif",  (0.0,    0.0, 1000.0, 1000.0)),
            ("tile_right.tif", (1000.0, 0.0, 2000.0, 1000.0)),
        ])

    def test_returns_path_for_point_in_left_tile(self, dsm, tmp_path):
        result = dsm._get_dsm_file(500.0, 500.0)
        assert result == os.path.join(str(tmp_path), "tile_left.tif")

    def test_returns_path_for_point_in_right_tile(self, dsm, tmp_path):
        result = dsm._get_dsm_file(1500.0, 500.0)
        assert result == os.path.join(str(tmp_path), "tile_right.tif")

    def test_raises_for_point_outside_all_tiles(self, dsm):
        with pytest.raises(ValueError, match="No DSM file contains the given point"):
            dsm._get_dsm_file(9999.0, 9999.0)

    def test_on_error_empty_returns_empty_string(self, dsm):
        result = dsm._get_dsm_file(9999.0, 9999.0, on_error="empty")
        assert result == ""


class TestRoundCoordinates:
    """Tests for the _round_coordinates method"""
    def test_rounds_to_one_decimal(self):
        vals = np.array([1.234, 5.678, 9.125])
        result = DigitalSurfaceModel._round_coordinates(vals)
        expected = np.array([1.3, 5.7, 9.1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_array_of_zeros(self):
        vals = np.array([0.0, 0.0, 0.0])
        result = DigitalSurfaceModel._round_coordinates(vals)
        np.testing.assert_array_equal(result, np.array([0.1, 0.1, 0.1]))

    def test_already_rounded_values_unchanged(self):
        vals = np.array([1.1, 2.3, 4.7])
        result = DigitalSurfaceModel._round_coordinates(vals)
        np.testing.assert_array_almost_equal(result, vals)

    def test_returns_numpy_array(self):
        vals = np.array([1.23, 4.56])
        result = DigitalSurfaceModel._round_coordinates(vals)
        assert isinstance(result, np.ndarray)

class TestGetAltitudeDataByTifFilename:
    """Tests for the get_altitude_data_by_tif_filename method"""
    @pytest.fixture
    def dsm(self, tmp_path):
        return _make_dsm(tmp_path, [("tile.tif", (0.0, 0.0, 1000.0, 1000.0))])

    def test_returns_none_for_empty_filename(self, dsm):
        group = pd.DataFrame({"x": [100.0], "y": [100.0]})
        result = dsm._get_altitude_data_by_tif_filename("", group)
        assert result is None

    def test_adds_altitude_column(self, dsm, tmp_path):
        mock_src = dsm._mock_srcs["tile.tif"]
        group = pd.DataFrame({"x": [100.0], "y": [100.0]})

        with patch("src.digital_surface_model.rasterio.open", return_value=mock_src):
            result = dsm._get_altitude_data_by_tif_filename(
                os.path.join(str(tmp_path), "tile.tif"), group
            )

        assert "altitude" in result.columns
        assert pd.api.types.is_numeric_dtype(result["altitude"])

    def test_returns_dataframe(self, dsm, tmp_path):
        mock_src = dsm._mock_srcs["tile.tif"]
        group = pd.DataFrame({"x": [100.0, 200.0], "y": [100.0, 200.0]})

        with patch("src.digital_surface_model.rasterio.open", return_value=mock_src):
            result = dsm._get_altitude_data_by_tif_filename(
                os.path.join(str(tmp_path), "tile.tif"), group
            )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


class TestGetAltitudeForDataframe:
    """Tests for the get_altitude_for_dataframe method."""
    @pytest.fixture
    def dsm(self, tmp_path):
        return _make_dsm(tmp_path, [("tile.tif", (0.0, 0.0, 1000.0, 1000.0))])

    def _patched_call(self, dsm, tmp_path, df):
        mock_src = dsm._mock_srcs["tile.tif"]
        with patch("src.digital_surface_model.rasterio.open", return_value=mock_src), \
             patch("src.digital_surface_model.ProcessPoolExecutor", ThreadPoolExecutor): # avoid multiprocessing in tests
            return dsm.get_altitude_for_dataframe(df.copy(), "x", "y", "altitude")

    def test_altitude_column_added(self, dsm, tmp_path):
        df = pd.DataFrame({"x": [100.0, 200.0], "y": [100.0, 200.0]})
        result = self._patched_call(dsm, tmp_path, df)
        assert "altitude" in result.columns
        assert pd.api.types.is_float_dtype(result["altitude"])
        assert len(result) == len(df) # no rows lost

    def test_out_of_bounds_points_get_nan(self, dsm):
        df = pd.DataFrame({"x": [100.0, 9999.0], "y": [100.0, 9999.0]})
        mock_src = dsm._mock_srcs["tile.tif"]
        with patch("src.digital_surface_model.rasterio.open", return_value=mock_src), \
             patch("src.digital_surface_model.ProcessPoolExecutor", ThreadPoolExecutor):
            result = dsm.get_altitude_for_dataframe(df.copy(), "x", "y", "altitude")
        assert pd.isna(result.loc[result["x"] > 9000, "altitude"]).all()
