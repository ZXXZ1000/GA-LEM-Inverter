"""Demo2: real DEM + real fault + rotated study-area alignment check.

这个脚本不是主优化入口，只用于肉眼检查空间对齐：
- 下载/生成一个小范围真实 DEM；
- 从 USGS Quaternary Faults 数据中裁出真实断层；
- 创建一个旋转研究区 polygon；
- 按当前主流程把 DEM、Ksp 和研究区同步旋转到同一个 georeferenced grid；
- 输出可视化图和 summary，检查是否错位、变形、残留 NoData。
"""

from __future__ import annotations

import math
import os
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
from PIL import Image
from pyproj import CRS
from rasterio.mask import mask
from rasterio.transform import from_bounds
from rasterio.warp import Resampling
from shapely.affinity import rotate
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ga_lem_inverter.pipeline.data import (
    build_rotated_profile_from_study_area,
    calculate_shp_rotation_angle,
    reproject_array_to_profile,
)
from ga_lem_inverter.pipeline.erosion import create_erosion_field
from ga_lem_inverter.workflows.main_inversion import fill_Nan


DATA_DIR = ROOT / "demo" / "data" / "demo2"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = ROOT / "demo" / "outputs" / "demo2"

QFAULT_URL = "https://earthquake.usgs.gov/static/lfs/nshm/qfaults/Qfaults_GIS.zip"
QFAULT_ZIP = RAW_DIR / "Qfaults_GIS.zip"
QFAULT_MEMBER = "SHP/Qfaults_US_Database.shp"

# Southern California / Garlock fault neighborhood.
BBOX_WGS84 = (-118.55, 34.75, -117.75, 35.35)
TARGET_CRS = "EPSG:32611"
TERRAIN_ZOOM = 10


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _download_qfaults_if_needed()
    dem_path = _build_real_dem()
    study_area_path, fault_path = _build_vectors(dem_path)
    artifacts = _run_alignment_check(dem_path, study_area_path, fault_path)
    _write_summary(artifacts)
    print(f"Demo2 alignment outputs: {OUTPUT_DIR}")


def _download_qfaults_if_needed() -> None:
    if QFAULT_ZIP.exists() and zipfile.is_zipfile(QFAULT_ZIP):
        return
    with requests.get(QFAULT_URL, stream=True, timeout=30) as response:
        response.raise_for_status()
        with QFAULT_ZIP.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _build_real_dem() -> Path:
    """Build a small GeoTIFF from Mapzen Terrarium real elevation tiles."""
    out_path = DATA_DIR / "demo2_real_dem.tif"
    if out_path.exists():
        return out_path

    lon_min, lat_min, lon_max, lat_max = BBOX_WGS84
    tile_min_x, tile_max_y = _lonlat_to_tile(lon_min, lat_min, TERRAIN_ZOOM)
    tile_max_x, tile_min_y = _lonlat_to_tile(lon_max, lat_max, TERRAIN_ZOOM)
    x_values = range(tile_min_x, tile_max_x + 1)
    y_values = range(tile_min_y, tile_max_y + 1)

    rows = []
    for y in y_values:
        row_tiles = []
        for x in x_values:
            row_tiles.append(_read_terrarium_tile(TERRAIN_ZOOM, x, y))
        rows.append(np.hstack(row_tiles))
    dem = np.vstack(rows).astype(np.float32)

    left, top = _tile_to_lonlat(tile_min_x, tile_min_y, TERRAIN_ZOOM)
    right, bottom = _tile_to_lonlat(tile_max_x + 1, tile_max_y + 1, TERRAIN_ZOOM)
    transform = from_bounds(left, bottom, right, top, dem.shape[1], dem.shape[0])

    profile = {
        "driver": "GTiff",
        "height": dem.shape[0],
        "width": dem.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": np.nan,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dem, 1)
    return out_path


def _read_terrarium_tile(z: int, x: int, y: int) -> np.ndarray:
    cache_path = RAW_DIR / f"terrarium_{z}_{x}_{y}.png"
    if not cache_path.exists():
        url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        cache_path.write_bytes(response.content)
    image = np.asarray(Image.open(cache_path).convert("RGB"), dtype=np.float32)
    return (image[:, :, 0] * 256.0 + image[:, :, 1] + image[:, :, 2] / 256.0) - 32768.0


def _lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    lat_rad = math.radians(lat)
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _tile_to_lonlat(x: int, y: int, z: int) -> tuple[float, float]:
    n = 2**z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def _build_vectors(dem_path: Path) -> tuple[Path, Path]:
    study_area_path = DATA_DIR / "demo2_study_area.shp"
    fault_path = DATA_DIR / "demo2_faults.shp"
    if study_area_path.exists() and fault_path.exists():
        return study_area_path, fault_path

    with rasterio.open(dem_path) as src:
        bounds_poly = box(*src.bounds)
        bounds_gdf = gpd.GeoDataFrame({"name": ["dem_bounds"]}, geometry=[bounds_poly], crs=src.crs)
        bounds_projected = bounds_gdf.to_crs(TARGET_CRS)
        minx, miny, maxx, maxy = bounds_projected.total_bounds
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        width = (maxx - minx) * 0.72
        height = (maxy - miny) * 0.48
        rotated_box = rotate(
            box(center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2),
            28.0,
            origin="centroid",
        )

    study_gdf = gpd.GeoDataFrame({"name": ["demo2_rotated_study_area"]}, geometry=[rotated_box], crs=TARGET_CRS)
    study_gdf.to_file(study_area_path, encoding="utf-8")

    fault_uri = f"zip://{QFAULT_ZIP.resolve()}!{QFAULT_MEMBER}"
    fault_wgs84 = gpd.read_file(fault_uri, bbox=BBOX_WGS84)
    fault_projected = fault_wgs84.to_crs(TARGET_CRS)
    fault_clipped = gpd.clip(fault_projected, study_gdf)
    if fault_clipped.empty:
        raise RuntimeError("Demo2 selected area contains no USGS fault lines.")
    fault_clipped.to_file(fault_path, encoding="utf-8")
    return study_area_path, fault_path


def _run_alignment_check(dem_path: Path, study_area_path: Path, fault_path: Path) -> dict[str, object]:
    with rasterio.open(dem_path) as src:
        study_gdf = gpd.read_file(study_area_path).to_crs(src.crs)
        clipped, clipped_transform = mask(src, study_gdf.geometry, crop=True, nodata=np.nan)
        clipped_dem = clipped[0].astype(np.float32)
        clipped_profile = src.profile.copy()
        clipped_profile.update(
            height=clipped_dem.shape[0],
            width=clipped_dem.shape[1],
            transform=clipped_transform,
            dtype="float32",
            nodata=np.nan,
        )

    # Reproject clipped DEM to projected CRS so FastScape spacing and fault rasterization use metric units.
    projected_dem_path = DATA_DIR / "demo2_clipped_projected_dem.tif"
    _write_projected_dem(clipped_dem, clipped_profile, projected_dem_path)
    with rasterio.open(projected_dem_path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    spacing = (abs(profile["transform"].a) + abs(profile["transform"].e)) / 2.0
    rotation_angle = calculate_shp_rotation_angle(str(study_area_path))
    ksp = create_erosion_field(
        shape=dem.shape,
        base_k_sp=1.0e-6,
        fault_k_sp=5.0e-6,
        fault_shp_path=str(fault_path),
        study_area_shp_path=str(study_area_path),
        rotation_angle=0,
        border_width=2,
        raster_transform=profile["transform"],
        raster_crs=profile["crs"],
    )
    rotated_profile = build_rotated_profile_from_study_area(profile, str(study_area_path), spacing=spacing)
    rotated_dem = fill_Nan(reproject_array_to_profile(dem, profile, rotated_profile, resampling=Resampling.bilinear))
    rotated_ksp = fill_Nan(reproject_array_to_profile(ksp, profile, rotated_profile, resampling=Resampling.bilinear))

    _save_array(DATA_DIR / "demo2_rotated_dem.npy", rotated_dem)
    _save_array(DATA_DIR / "demo2_rotated_ksp.npy", rotated_ksp)
    _plot_alignment(dem, ksp, rotated_dem, rotated_ksp, profile, rotated_profile, study_area_path, fault_path)

    return {
        "dem_shape": dem.shape,
        "rotated_shape": rotated_dem.shape,
        "rotation_angle": rotation_angle,
        "spacing": spacing,
        "rotated_dem_has_nan": bool(np.isnan(rotated_dem).any()),
        "rotated_ksp_has_nan": bool(np.isnan(rotated_ksp).any()),
        "figures": sorted(str(path.relative_to(ROOT)) for path in OUTPUT_DIR.glob("*.png")),
    }


def _write_projected_dem(dem: np.ndarray, profile: dict, out_path: Path) -> None:
    from rasterio.warp import calculate_default_transform, reproject

    src_crs = CRS.from_user_input(profile["crs"])
    dst_crs = CRS.from_user_input(TARGET_CRS)
    with rasterio.io.MemoryFile() as memfile:
        temp_profile = profile.copy()
        temp_profile.update(driver="GTiff", count=1)
        with memfile.open(**temp_profile) as src:
            src.write(dem, 1)
            transform, width, height = calculate_default_transform(
                src_crs, dst_crs, src.width, src.height, *src.bounds, resolution=120.0
            )
            dst_profile = src.profile.copy()
            dst_profile.update(crs=dst_crs, transform=transform, width=width, height=height, nodata=np.nan)
            projected = np.full((height, width), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=projected,
                src_transform=src.transform,
                src_crs=src_crs,
                src_nodata=np.nan,
                dst_transform=transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    with rasterio.open(out_path, "w", **dst_profile) as dst:
        dst.write(projected, 1)


def _plot_alignment(
    dem: np.ndarray,
    ksp: np.ndarray,
    rotated_dem: np.ndarray,
    rotated_ksp: np.ndarray,
    profile: dict,
    rotated_profile: dict,
    study_area_path: Path,
    fault_path: Path,
) -> None:
    fault = gpd.read_file(fault_path).to_crs(profile["crs"])
    study = gpd.read_file(study_area_path).to_crs(profile["crs"])
    fault_rot = fault.to_crs(rotated_profile["crs"])
    study_rot = study.to_crs(rotated_profile["crs"])

    _plot_map_overlay(dem, profile, study, fault, OUTPUT_DIR / "01_source_dem_fault_study_area.png")
    _plot_matrix_overlay(dem, ksp, OUTPUT_DIR / "02_source_dem_ksp_matrix.png")
    _plot_matrix_overlay(rotated_dem, rotated_ksp, OUTPUT_DIR / "03_rotated_dem_ksp_matrix.png")
    _plot_rotated_map_overlay(
        rotated_dem,
        rotated_profile,
        study_rot,
        fault_rot,
        OUTPUT_DIR / "04_rotated_dem_fault_study_area_georef.png",
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    axes[0, 0].imshow(dem, cmap="terrain")
    axes[0, 0].set_title(f"Source DEM {dem.shape}")
    axes[0, 1].imshow(ksp, cmap="magma")
    axes[0, 1].set_title("Source Ksp from real faults")
    axes[1, 0].imshow(rotated_dem, cmap="terrain")
    axes[1, 0].set_title(f"Rotated filled DEM {rotated_dem.shape}")
    axes[1, 1].imshow(rotated_ksp, cmap="magma")
    axes[1, 1].set_title("Rotated Ksp")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(OUTPUT_DIR / "05_alignment_overview.png", dpi=180)
    plt.close(fig)


def _plot_map_overlay(dem: np.ndarray, profile: dict, study: gpd.GeoDataFrame, fault: gpd.GeoDataFrame, path: Path) -> None:
    bounds = rasterio.transform.array_bounds(dem.shape[0], dem.shape[1], profile["transform"])
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.imshow(dem, cmap="terrain", extent=(bounds[0], bounds[2], bounds[1], bounds[3]), origin="upper")
    study.boundary.plot(ax=ax, edgecolor="cyan", linewidth=2)
    fault.plot(ax=ax, color="black", linewidth=0.8)
    ax.set_title("Source georeferenced DEM + real USGS faults + rotated study area")
    ax.set_aspect("equal")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_matrix_overlay(dem: np.ndarray, ksp: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.imshow(dem, cmap="terrain", origin="upper")
    boosted = np.ma.masked_where(ksp <= np.nanmedian(ksp), ksp)
    ax.imshow(boosted, cmap="autumn", alpha=0.75, origin="upper")
    ax.set_title("Matrix view: DEM with Ksp/fault pixels")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_rotated_map_overlay(
    dem: np.ndarray,
    profile: dict,
    study: gpd.GeoDataFrame,
    fault: gpd.GeoDataFrame,
    path: Path,
) -> None:
    bounds = rasterio.transform.array_bounds(dem.shape[0], dem.shape[1], profile["transform"])
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.imshow(dem, cmap="terrain", extent=(bounds[0], bounds[2], bounds[1], bounds[3]), origin="upper")
    study.boundary.plot(ax=ax, edgecolor="cyan", linewidth=2)
    fault.plot(ax=ax, color="black", linewidth=0.8)
    ax.set_title("Rotated georeferenced DEM + same real faults/study area")
    ax.set_aspect("equal")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_array(path: Path, array: np.ndarray) -> None:
    np.save(path, np.asarray(array, dtype=np.float32))


def _write_summary(artifacts: dict[str, object]) -> None:
    lines = [
        "# Demo2 Alignment Check",
        "",
        f"- Source DEM: `{DATA_DIR / 'demo2_real_dem.tif'}`",
        f"- Fault source: USGS Quaternary Faults GIS, clipped to `{DATA_DIR / 'demo2_faults.shp'}`",
        f"- Study area: `{DATA_DIR / 'demo2_study_area.shp'}`",
        f"- Source DEM shape: `{artifacts['dem_shape']}`",
        f"- Rotated shape: `{artifacts['rotated_shape']}`",
        f"- Rotation angle: `{artifacts['rotation_angle']:.3f}` degrees",
        f"- Pixel spacing: `{artifacts['spacing']:.2f}` m",
        f"- Rotated DEM has NaN: `{artifacts['rotated_dem_has_nan']}`",
        f"- Rotated Ksp has NaN: `{artifacts['rotated_ksp_has_nan']}`",
        "",
        "Figures:",
    ]
    for fig in artifacts["figures"]:
        lines.append(f"- `{fig}`")
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
