import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pyproj import Transformer
from rasterio.transform import from_origin

from ga_lem_inverter.integrations.pecube_fitness import (
    load_observations,
    pecube_grid_from_dem_profile,
    pecube_spatial_adapter_from_dem_profile,
)


class PecubeCoordinateTests(unittest.TestCase):
    def test_geographic_dem_uses_original_regular_grid(self):
        """EPSG:4326 DEM 已经是规则经纬度网格时，不应额外重采样。"""
        transform = from_origin(105.0, 1.0, 0.01, 0.02)
        adapter = pecube_spatial_adapter_from_dem_profile(
            {"crs": "EPSG:4326", "transform": transform},
            (4, 5),
        )

        self.assertIsNotNone(adapter)
        self.assertFalse(adapter.resample)
        self.assertEqual(adapter.target_shape, (4, 5))
        self.assertEqual(adapter.grid.crs, "EPSG:4326")
        self.assertAlmostEqual(adapter.grid.lon0, 105.005)
        self.assertAlmostEqual(adapter.grid.lat0, 0.93)
        self.assertAlmostEqual(adapter.grid.dlon, 0.01)
        self.assertAlmostEqual(adapter.grid.dlat, 0.02)

        data = np.arange(20, dtype=float).reshape(4, 5)
        transformed = adapter.transform_array(data)
        self.assertTrue(np.array_equal(transformed, data))
        self.assertIsNot(transformed, data)

        grid = pecube_grid_from_dem_profile({"crs": "EPSG:4326", "transform": transform}, (4, 5))
        self.assertEqual(grid, adapter.grid)

    def test_projected_dem_reprojects_arrays_to_regular_geographic_grid(self):
        """投影 DEM 必须先重采样到规则经纬度网格，再交给 Pecube。"""
        transform = from_origin(500000.0, 4000.0, 900.0, 900.0)
        adapter = pecube_spatial_adapter_from_dem_profile(
            {"crs": "EPSG:32648", "transform": transform},
            (4, 5),
        )

        self.assertIsNotNone(adapter)
        self.assertTrue(adapter.resample)
        self.assertEqual(adapter.grid.crs, "EPSG:4326")
        self.assertEqual(adapter.grid.source, "dem_profile_reprojected")
        self.assertGreater(adapter.target_shape[0], 1)
        self.assertGreater(adapter.target_shape[1], 1)
        self.assertGreater(adapter.grid.dlon, 0.0)
        self.assertGreater(adapter.grid.dlat, 0.0)

        data = np.arange(20, dtype=float).reshape(4, 5)
        transformed = adapter.transform_array(data)
        self.assertEqual(transformed.shape, adapter.target_shape)
        self.assertTrue(np.isfinite(transformed).any())

    def test_projected_sample_coordinates_convert_to_pecube_lonlat(self):
        """真实投影坐标样品应自动转换到 Pecube 的经纬度输出坐标系。"""
        transform = from_origin(500000.0, 4000.0, 900.0, 900.0)
        adapter = pecube_spatial_adapter_from_dem_profile(
            {"crs": "EPSG:32648", "transform": transform},
            (4, 5),
        )
        self.assertIsNotNone(adapter)

        sample_x, sample_y = transform * (2.5, 1.5)
        expected_lon, expected_lat = Transformer.from_crs("EPSG:32648", "EPSG:4326", always_xy=True).transform(
            sample_x,
            sample_y,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.csv"
            self._write_samples(path, sample_x, sample_y)
            observations = load_observations(
                path,
                adapter.target_shape,
                coordinate_system="dem_crs",
                dlon=adapter.grid.dlon,
                dlat=adapter.grid.dlat,
                lon0=adapter.grid.lon0,
                lat0=adapter.grid.lat0,
                dem_crs=adapter.source_crs,
                dem_transform=adapter.source_transform,
                source_dem_shape=adapter.source_shape,
            )

        self.assertEqual(len(observations), 1)
        self.assertAlmostEqual(observations[0].x, expected_lon, places=6)
        self.assertAlmostEqual(observations[0].y, expected_lat, places=6)

    def test_grid_index_samples_use_source_dem_grid_before_lonlat_conversion(self):
        """grid_index 样品在自动模式下表示原 DEM 像素索引，而不是 Pecube 输出索引。"""
        transform = from_origin(500000.0, 4000.0, 900.0, 900.0)
        adapter = pecube_spatial_adapter_from_dem_profile(
            {"crs": "EPSG:32648", "transform": transform},
            (4, 5),
        )
        self.assertIsNotNone(adapter)

        expected_x, expected_y = transform * (2.5, 1.5)
        expected_lon, expected_lat = Transformer.from_crs("EPSG:32648", "EPSG:4326", always_xy=True).transform(
            expected_x,
            expected_y,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.csv"
            self._write_samples(path, 2.0, 1.0)
            observations = load_observations(
                path,
                adapter.target_shape,
                coordinate_system="grid_index",
                dlon=adapter.grid.dlon,
                dlat=adapter.grid.dlat,
                lon0=adapter.grid.lon0,
                lat0=adapter.grid.lat0,
                dem_crs=adapter.source_crs,
                dem_transform=adapter.source_transform,
                source_dem_shape=adapter.source_shape,
            )

        self.assertEqual(len(observations), 1)
        self.assertAlmostEqual(observations[0].x, expected_lon, places=6)
        self.assertAlmostEqual(observations[0].y, expected_lat, places=6)

    @staticmethod
    def _write_samples(path: Path, x: float, y: float) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "x", "y", "elevation", "system", "observed_age", "sigma"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "sample_id": "S1",
                    "x": x,
                    "y": y,
                    "elevation": 100.0,
                    "system": "AHe",
                    "observed_age": 5.0,
                    "sigma": 0.5,
                }
            )


if __name__ == "__main__":
    unittest.main()
