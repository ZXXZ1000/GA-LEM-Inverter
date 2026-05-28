import tempfile
import unittest
from pathlib import Path

import configparser
import csv
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.affinity import rotate
from shapely.geometry import LineString, box

from ga_lem_inverter.integrations.pecube_fitness import load_observations, pecube_spatial_adapter_from_dem_profile
from ga_lem_inverter.pipeline.data import (
    build_rotated_profile_from_dem_footprint,
    build_rotated_profile_from_study_area,
    load_dem_data,
    reproject_array_to_profile,
    reproject_files_to_geographic,
)
from ga_lem_inverter.pipeline.erosion import create_erosion_field
from ga_lem_inverter.workflows.main_inversion import create_model_grid_erosion_field, fill_Nan


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
            rotated_dem = reproject_array_to_profile(dem, profile, rotated_profile)
            rotated_ksp = create_model_grid_erosion_field(
                shape=rotated_dem.shape,
                base_k_sp=1.0,
                fault_k_sp=4.0,
                fault_shp_path=None,
                study_area_shp_path=None,
                dem_profile=rotated_profile,
                border_width=1,
            )

        transform = rotated_profile["transform"]
        self.assertEqual(rotated_dem.shape, rotated_ksp.shape)
        self.assertEqual(rotated_dem.shape, (rotated_profile["height"], rotated_profile["width"]))
        self.assertNotAlmostEqual(transform.b, 0.0)
        self.assertNotAlmostEqual(transform.d, 0.0)
        self.assertTrue(np.isfinite(rotated_dem).any())
        self.assertTrue(np.isfinite(rotated_ksp).any())
        self.assertFalse(np.isnan(fill_Nan(rotated_dem)).any())
        self.assertFalse(np.isnan(rotated_ksp).any())
        self.assertEqual(set(np.unique(rotated_ksp)), {0.0, 1.0})

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

    def test_projected_samples_align_to_rotated_pecube_grid(self):
        """产品验收：真实投影样品坐标必须能通过旋转后的 DEM profile 落入 Pecube 网格。"""
        profile = {
            "crs": "EPSG:32648",
            "transform": from_origin(500000.0, 10000.0, 100.0, 100.0),
        }
        polygon = rotate(box(500500.0, 2000.0, 507500.0, 7800.0), 31.0, origin="centroid")

        with tempfile.TemporaryDirectory() as tmpdir:
            study_path = Path(tmpdir) / "study_area.shp"
            sample_path = Path(tmpdir) / "samples.csv"
            gpd.GeoDataFrame({"id": [1]}, geometry=[polygon], crs="EPSG:32648").to_file(study_path)
            rotated_profile = build_rotated_profile_from_study_area(profile, str(study_path), spacing=100.0)
            adapter = pecube_spatial_adapter_from_dem_profile(
                rotated_profile,
                (rotated_profile["height"], rotated_profile["width"]),
            )
            self.assertIsNotNone(adapter)

            sample_x, sample_y = polygon.centroid.x, polygon.centroid.y
            self._write_samples(sample_path, sample_x, sample_y)
            observations = load_observations(
                sample_path,
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
        self.assertGreaterEqual(observations[0].x, adapter.grid.lon0)
        self.assertGreaterEqual(observations[0].y, adapter.grid.lat0)
        self.assertLessEqual(observations[0].x, adapter.grid.lon0 + (adapter.target_shape[1] - 1) * adapter.grid.dlon)
        self.assertLessEqual(observations[0].y, adapter.grid.lat0 + (adapter.target_shape[0] - 1) * adapter.grid.dlat)

    def test_fault_ksp_rasterizes_on_dem_transform_before_rotation(self):
        """产品验收：断层 Ksp 必须先按 DEM transform 落到正确像素，再进入统一旋转。"""
        transform = from_origin(0.0, 1000.0, 10.0, 10.0)
        study_area = box(100.0, 100.0, 900.0, 900.0)
        fault_line = LineString([(500.0, 120.0), (500.0, 880.0)])

        with tempfile.TemporaryDirectory() as tmpdir:
            study_path = Path(tmpdir) / "study_area.shp"
            fault_path = Path(tmpdir) / "fault.shp"
            gpd.GeoDataFrame({"id": [1]}, geometry=[study_area], crs="EPSG:32648").to_file(study_path)
            gpd.GeoDataFrame({"id": [1]}, geometry=[fault_line], crs="EPSG:32648").to_file(fault_path)

            ksp = create_erosion_field(
                shape=(100, 100),
                base_k_sp=1.0,
                fault_k_sp=4.0,
                fault_shp_path=str(fault_path),
                study_area_shp_path=str(study_path),
                rotation_angle=0,
                border_width=1,
                raster_transform=transform,
                raster_crs="EPSG:32648",
            )

        boosted_pixels = np.argwhere(ksp > 1.0)
        self.assertGreater(boosted_pixels.shape[0], 0)
        self.assertLess(abs(float(np.median(boosted_pixels[:, 1])) - 50.0), 2.0)

    def test_fault_ksp_rasterizes_directly_on_rotated_model_grid(self):
        """产品验收：rotated Ksp 必须在最终网格原生生成，不应出现重采样中间值。"""
        profile = {
            "crs": "EPSG:32648",
            "transform": from_origin(0.0, 1000.0, 10.0, 10.0),
            "height": 100,
            "width": 100,
            "dtype": "float32",
        }
        study_area = rotate(box(200.0, 300.0, 720.0, 560.0), 28.0, origin="centroid")
        fault_line = LineString([study_area.centroid.coords[0], (study_area.centroid.x + 250.0, study_area.centroid.y)])

        with tempfile.TemporaryDirectory() as tmpdir:
            study_path = Path(tmpdir) / "study_area.shp"
            fault_path = Path(tmpdir) / "fault.shp"
            gpd.GeoDataFrame({"id": [1]}, geometry=[study_area], crs="EPSG:32648").to_file(study_path)
            gpd.GeoDataFrame({"id": [1]}, geometry=[fault_line], crs="EPSG:32648").to_file(fault_path)
            rotated_profile = build_rotated_profile_from_study_area(profile, str(study_path), spacing=10.0)

            ksp = create_model_grid_erosion_field(
                shape=(rotated_profile["height"], rotated_profile["width"]),
                base_k_sp=1.0,
                fault_k_sp=4.0,
                fault_shp_path=str(fault_path),
                study_area_shp_path=str(study_path),
                dem_profile=rotated_profile,
                border_width=1,
            )

        self.assertEqual(ksp.shape, (rotated_profile["height"], rotated_profile["width"]))
        self.assertFalse(np.isnan(ksp).any())
        self.assertGreater(np.count_nonzero(ksp > 1.0), 0)
        self.assertTrue(set(np.unique(ksp)).issubset({0.0, 1.0, 5.0}))

    def test_reprojected_raster_writes_outside_footprint_as_nan(self):
        """产品验收：经纬度 DEM 投影到 UTM 后，外接网格空白区必须是 NaN 而不是 0 海拔。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            terrain_path = Path(tmpdir) / "hangay_like.tif"
            with rasterio.open(
                terrain_path,
                "w",
                driver="GTiff",
                height=24,
                width=36,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_origin(96.5, 48.2, 0.05, 0.05),
            ) as dst:
                dst.write(np.full((24, 36), 1800.0, dtype=np.float32), 1)

            config = configparser.ConfigParser()
            config["Paths"] = {
                "terrain_path": str(terrain_path),
                "fault_shp_path": "none",
                "study_area_shp_path": "none",
                "output_path": str(Path(tmpdir) / "outputs"),
            }

            updated = reproject_files_to_geographic(config, "EPSG:32647")
            self.assertEqual(Path(updated["Paths"]["terrain_path"]).parent.name, "reprojected_inputs")
            with rasterio.open(updated["Paths"]["terrain_path"]) as src:
                data = src.read(1)
                nodata = src.nodata

        self.assertEqual(str(nodata), "nan")
        self.assertTrue(np.isnan(data).any())
        self.assertEqual(np.count_nonzero(data == 0.0), 0)

    def test_load_dem_data_preserves_geotiff_nodata_as_nan(self):
        """产品验收：用户直接输入已裁切 DEM 时，GeoTIFF nodata 不能被当作有效地形。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            terrain_path = Path(tmpdir) / "clipped_dem.tif"
            dem = np.full((12, 16), 2000.0, dtype=np.float32)
            dem[:2, :] = -9999.0
            with rasterio.open(
                terrain_path,
                "w",
                driver="GTiff",
                height=12,
                width=16,
                count=1,
                dtype="float32",
                crs="EPSG:32647",
                transform=from_origin(300000.0, 5400000.0, 30.0, 30.0),
                nodata=-9999.0,
            ) as dst:
                dst.write(dem, 1)

            loaded, profile = load_dem_data(str(terrain_path), ratio=1.0)

        self.assertTrue(np.isnan(loaded[:2, :]).all())
        self.assertTrue(np.isnan(profile["nodata"]))

    def test_dem_footprint_rotation_is_inferred_without_study_area(self):
        """产品验收：用户只给已裁切 DEM 时，主流程仍能从有效 footprint 推断旋转网格。"""
        profile = {
            "crs": "EPSG:32647",
            "transform": from_origin(300000.0, 5400000.0, 30.0, 30.0),
            "height": 120,
            "width": 160,
            "dtype": "float32",
            "nodata": np.nan,
        }
        dem = np.full((120, 160), np.nan, dtype=np.float32)
        y, x = np.indices(dem.shape)
        footprint = (x > 15) & (x < 145) & (y > 12 + 0.2 * x) & (y < 88 + 0.2 * x)
        dem[footprint] = 2500.0

        rotated_profile, rotation_angle = build_rotated_profile_from_dem_footprint(profile, dem, spacing=30.0)

        self.assertGreater(abs(rotation_angle), 0.25)
        self.assertNotAlmostEqual(rotated_profile["transform"].b, 0.0)
        self.assertNotAlmostEqual(rotated_profile["transform"].d, 0.0)
        self.assertLess(rotated_profile["height"], profile["height"])

    def test_tall_dem_footprint_does_not_force_90_degree_rotation(self):
        """产品验收：DEM 有效区只是高大于宽时，不应为了长边横放而顺时针转 90 度。"""
        profile = {
            "crs": "EPSG:32647",
            "transform": from_origin(300000.0, 5400000.0, 30.0, 30.0),
            "height": 180,
            "width": 90,
            "dtype": "float32",
            "nodata": np.nan,
        }
        dem = np.full((180, 90), np.nan, dtype=np.float32)
        dem[20:160, 25:65] = 2500.0

        rotated_profile, rotation_angle = build_rotated_profile_from_dem_footprint(profile, dem, spacing=30.0)

        self.assertEqual(rotation_angle, 0.0)
        self.assertAlmostEqual(rotated_profile["transform"].b, 0.0)
        self.assertAlmostEqual(rotated_profile["transform"].d, 0.0)

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
