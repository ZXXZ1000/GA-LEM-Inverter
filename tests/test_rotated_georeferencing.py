import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.affinity import rotate
from shapely.geometry import box

from ga_lem_inverter.integrations.pecube_fitness import pecube_spatial_adapter_from_dem_profile
from ga_lem_inverter.pipeline.data import build_rotated_profile_from_study_area, reproject_array_to_profile


class RotatedGeoreferencingAcceptanceTests(unittest.TestCase):
    def test_rotated_study_area_builds_shared_profile_for_dem_and_ksp(self):
        """产品验收：非正裁剪区必须生成带旋转 transform 的 DEM/Ksp 共用网格。"""
        profile = {
            "crs": "EPSG:32648",
            "transform": from_origin(0.0, 1000.0, 10.0, 10.0),
            "height": 100,
            "width": 100,
            "dtype": "float32",
        }
        polygon = rotate(box(200.0, 300.0, 720.0, 560.0), 28.0, origin="centroid")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "study_area.shp"
            gpd.GeoDataFrame({"id": [1]}, geometry=[polygon], crs="EPSG:32648").to_file(path)

            rotated_profile = build_rotated_profile_from_study_area(profile, str(path), spacing=10.0)
            dem = np.arange(10000, dtype=np.float32).reshape(100, 100)
            ksp = np.ones((100, 100), dtype=np.float32)
            rotated_dem = reproject_array_to_profile(dem, profile, rotated_profile)
            rotated_ksp = reproject_array_to_profile(ksp, profile, rotated_profile)

        transform = rotated_profile["transform"]
        self.assertEqual(rotated_dem.shape, rotated_ksp.shape)
        self.assertEqual(rotated_dem.shape, (rotated_profile["height"], rotated_profile["width"]))
        self.assertNotAlmostEqual(transform.b, 0.0)
        self.assertNotAlmostEqual(transform.d, 0.0)
        self.assertTrue(np.isfinite(rotated_dem).any())
        self.assertTrue(np.isfinite(rotated_ksp).any())

    def test_rotated_projected_profile_can_drive_pecube_auto_grid(self):
        """产品验收：Pecube 自动坐标必须能读取旋转后的投影 affine 并生成经纬度输出网格。"""
        profile = {
            "crs": "EPSG:32648",
            "transform": from_origin(500000.0, 4000.0, 900.0, 900.0),
        }
        polygon = rotate(box(500000.0, 0.0, 504500.0, 2700.0), 25.0, origin="centroid")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "study_area.shp"
            gpd.GeoDataFrame({"id": [1]}, geometry=[polygon], crs="EPSG:32648").to_file(path)
            rotated_profile = build_rotated_profile_from_study_area(profile, str(path), spacing=900.0)

        adapter = pecube_spatial_adapter_from_dem_profile(
            rotated_profile,
            (rotated_profile["height"], rotated_profile["width"]),
        )

        self.assertIsNotNone(adapter)
        self.assertTrue(adapter.resample)
        self.assertEqual(adapter.source_shape, (rotated_profile["height"], rotated_profile["width"]))
        self.assertEqual(adapter.grid.crs, "EPSG:4326")
        self.assertGreater(adapter.grid.dlon, 0.0)
        self.assertGreater(adapter.grid.dlat, 0.0)

        transformed = adapter.transform_array(np.ones(adapter.source_shape, dtype=float))
        self.assertEqual(transformed.shape, adapter.target_shape)
        self.assertTrue(np.isfinite(transformed).any())


if __name__ == "__main__":
    unittest.main()
