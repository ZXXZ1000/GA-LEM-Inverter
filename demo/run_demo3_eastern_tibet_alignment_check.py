"""Demo3: eastern Tibetan Plateau / Minjiang to Sichuan Basin alignment check.

这个脚本不是主优化入口，只用于肉眼检查空间对齐：
- 下载/生成一个小范围真实 DEM；
- 从 GMT 中文手册 CN-faults 数据中裁出岷江-龙门山-四川盆地西缘断层；
- 创建一个旋转研究区 polygon；
- 按当前主流程旋转 DEM，并在最终 georeferenced grid 上重新生成 Ksp；
- 输出可视化图和 summary，检查是否错位、变形、残留 NoData。
"""

from __future__ import annotations

import math
import os
import sys
import zipfile
import json
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
from shapely.ops import transform as shapely_transform
from shapely.geometry import LineString, Point, box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ga_lem_inverter.pipeline.data import (
    build_rotated_profile_from_study_area,
    calculate_shp_rotation_angle,
    reproject_array_to_profile,
)
from ga_lem_inverter.pipeline.erosion import create_erosion_field
from ga_lem_inverter.workflows.main_inversion import create_model_grid_erosion_field, fill_Nan


DATA_DIR = ROOT / "demo" / "data" / "demo3"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = ROOT / "demo" / "outputs" / "demo3"

CN_FAULTS_URL = "https://github.com/gmt-china/china-geospatial-data/releases/download/v0.4.0/china-geospatial-data-UTF8.zip"
CN_FAULTS_ZIP = RAW_DIR / "china-geospatial-data-UTF8.zip"
CN_FAULTS_EXTRACT_DIR = RAW_DIR / "china-geospatial-data"
CN_FAULTS_RELATIVE_PATH = "china-geospatial-data-UTF8/CN-faults.gmt"

# Eastern Tibetan Plateau: Minjiang - Longmenshan - western Sichuan Basin.
BBOX_WGS84 = (102.35, 30.05, 105.65, 33.35)
TARGET_CRS = "EPSG:32648"
TERRAIN_ZOOM = 8
LONGMENSHAN_PARALLEL_ANGLE_DEGREES = 38.0
THERMO_JSON = DATA_DIR / "thermo_samples_raw.json"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _download_faults_if_needed()
    dem_path = _build_real_dem()
    study_area_path, fault_path = _build_vectors(dem_path)
    artifacts = _run_alignment_check(dem_path, study_area_path, fault_path)
    _write_summary(artifacts)
    print(f"Demo3 alignment outputs: {OUTPUT_DIR}")


def _download_faults_if_needed() -> None:
    if not (CN_FAULTS_ZIP.exists() and zipfile.is_zipfile(CN_FAULTS_ZIP)):
        with requests.get(CN_FAULTS_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            with CN_FAULTS_ZIP.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    if not (CN_FAULTS_EXTRACT_DIR / CN_FAULTS_RELATIVE_PATH).exists():
        CN_FAULTS_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(CN_FAULTS_ZIP) as archive:
            archive.extractall(CN_FAULTS_EXTRACT_DIR)


def _build_real_dem() -> Path:
    """Build a small GeoTIFF from Mapzen Terrarium real elevation tiles."""
    out_path = DATA_DIR / "demo3_real_dem.tif"
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
    study_area_path = DATA_DIR / "demo3_study_area.shp"
    fault_path = DATA_DIR / "demo3_faults.shp"
    if study_area_path.exists() and fault_path.exists():
        return study_area_path, fault_path

    with rasterio.open(dem_path) as src:
        bounds_projected = _study_bounds_from_thermo_samples(src)
        minx, miny, maxx, maxy = bounds_projected.total_bounds
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        width = max((maxx - minx) * 1.25, 85_000.0)
        height = max((maxy - miny) * 1.65, 70_000.0)
        rotated_box = rotate(
            box(center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2),
            LONGMENSHAN_PARALLEL_ANGLE_DEGREES,
            origin="centroid",
        )

    study_gdf = gpd.GeoDataFrame({"name": ["demo3_rotated_study_area"]}, geometry=[rotated_box], crs=TARGET_CRS)
    study_gdf.to_file(study_area_path, encoding="utf-8")

    fault_wgs84 = _load_fault_sources()
    fault_projected = fault_wgs84.to_crs(TARGET_CRS)
    fault_clipped = gpd.clip(fault_projected, study_gdf)
    if fault_clipped.empty:
        raise RuntimeError("Demo3 selected area contains no CN-faults lines.")
    fault_clipped.to_file(fault_path, encoding="utf-8")
    return study_area_path, fault_path


def _study_bounds_from_thermo_samples(src: rasterio.io.DatasetReader) -> gpd.GeoDataFrame:
    if THERMO_JSON.exists():
        samples = json.loads(THERMO_JSON.read_text(encoding="utf-8"))
        points = [
            Point(float(item["Long"]), float(item["Lat"]))
            for item in samples
            if item.get("Long") is not None and item.get("Lat") is not None
        ]
        if points:
            sample_gdf = gpd.GeoDataFrame({"kind": ["thermo_samples"] * len(points)}, geometry=points, crs="EPSG:4326")
            return sample_gdf.to_crs(TARGET_CRS)

    bounds_poly = box(*src.bounds)
    return gpd.GeoDataFrame({"kind": ["dem_bounds"]}, geometry=[bounds_poly], crs=src.crs).to_crs(TARGET_CRS)


def _load_fault_sources() -> gpd.GeoDataFrame:
    cn_faults = CN_FAULTS_EXTRACT_DIR / CN_FAULTS_RELATIVE_PATH
    if not cn_faults.exists():
        raise RuntimeError(f"CN-faults 数据缺失: {cn_faults}")
    return _read_cn_faults_gmt(cn_faults, BBOX_WGS84)


def _read_cn_faults_gmt(path: Path, bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    records = []
    current_meta: dict[str, str] = {}
    current_coords: list[tuple[float, float]] = []

    def flush() -> None:
        nonlocal current_meta, current_coords
        if len(current_coords) >= 2:
            line = LineString(current_coords)
            min_lon, min_lat, max_lon, max_lat = line.bounds
            if max_lon >= bbox[0] and min_lon <= bbox[2] and max_lat >= bbox[1] and min_lat <= bbox[3]:
                records.append(
                    {
                        **current_meta,
                        "source": "CN-faults / GMT China",
                        "display_name": current_meta.get("name_en") or current_meta.get("name_cn") or "unnamed_fault",
                        "geometry": line,
                    }
                )
        current_meta = {}
        current_coords = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                continue
            if line.startswith("# @D"):
                parts = line.removeprefix("# @D").split("|")
                current_meta = {
                    "zone_cn": _clean_gmt_field(parts[0]) if len(parts) > 0 else "",
                    "zone_en": _clean_gmt_field(parts[1]) if len(parts) > 1 else "",
                    "name_cn": _clean_gmt_field(parts[2]) if len(parts) > 2 else "",
                    "name_en": _clean_gmt_field(parts[3]) if len(parts) > 3 else "",
                    "feature_cn": _clean_gmt_field(parts[6]) if len(parts) > 6 else "",
                    "feature_en": _clean_gmt_field(parts[7]) if len(parts) > 7 else "",
                    "age": _clean_gmt_field(parts[8]) if len(parts) > 8 else "",
                    "reference": _clean_gmt_field(parts[10]) if len(parts) > 10 else "",
                }
                continue
            if line.startswith("#"):
                continue
            pieces = line.split()
            if len(pieces) >= 2:
                try:
                    lon = float(pieces[0])
                    lat = float(pieces[1])
                    current_coords.append((lon, lat))
                except ValueError:
                    continue
    flush()

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    if gdf.empty:
        return gdf
    return gdf.cx[bbox[0] : bbox[2], bbox[1] : bbox[3]].copy()


def _clean_gmt_field(value: str) -> str:
    return value.strip().strip('"')


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
    projected_dem_path = DATA_DIR / "demo3_clipped_projected_dem.tif"
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
    rotated_ksp = create_model_grid_erosion_field(
        shape=rotated_dem.shape,
        base_k_sp=1.0e-6,
        fault_k_sp=5.0e-6,
        fault_shp_path=str(fault_path),
        study_area_shp_path=str(study_area_path),
        dem_profile=rotated_profile,
        border_width=2,
    )

    _save_array(DATA_DIR / "demo3_rotated_dem.npy", rotated_dem)
    _save_array(DATA_DIR / "demo3_rotated_ksp.npy", rotated_ksp)
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
    _plot_matrix_overlay(
        rotated_dem,
        rotated_ksp,
        OUTPUT_DIR / "03_rotated_dem_ksp_matrix.png",
        flip_vertical=True,
        flip_horizontal=True,
    )
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
    axes[1, 0].imshow(np.fliplr(np.flipud(rotated_dem)), cmap="terrain")
    axes[1, 0].set_title(f"Rotated filled DEM {rotated_dem.shape}")
    axes[1, 1].imshow(np.fliplr(np.flipud(rotated_ksp)), cmap="magma")
    axes[1, 1].set_title("Rotated-grid Ksp")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(OUTPUT_DIR / "05_alignment_overview.png", dpi=180)
    plt.close(fig)


def _plot_map_overlay(
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
    _annotate_faults(ax, fault)
    ax.set_title("Source georeferenced DEM + CN-faults + study area")
    ax.set_aspect("equal")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_matrix_overlay(
    dem: np.ndarray,
    ksp: np.ndarray,
    path: Path,
    flip_vertical: bool = False,
    flip_horizontal: bool = False,
) -> None:
    display_dem = np.flipud(dem) if flip_vertical else dem
    display_ksp = np.flipud(ksp) if flip_vertical else ksp
    display_dem = np.fliplr(display_dem) if flip_horizontal else display_dem
    display_ksp = np.fliplr(display_ksp) if flip_horizontal else display_ksp
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.imshow(display_dem, cmap="terrain", origin="upper")
    boosted = np.ma.masked_where(display_ksp <= np.nanmedian(display_ksp), display_ksp)
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
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.imshow(np.fliplr(np.flipud(dem)), cmap="terrain", origin="upper")
    study_pixel = _to_pixel_geometries(
        study,
        profile["transform"],
        height=dem.shape[0],
        width=dem.shape[1],
        flip_vertical=True,
        flip_horizontal=True,
    )
    fault_pixel = _to_pixel_geometries(
        fault,
        profile["transform"],
        height=dem.shape[0],
        width=dem.shape[1],
        flip_vertical=True,
        flip_horizontal=True,
    )
    study_pixel.boundary.plot(ax=ax, edgecolor="cyan", linewidth=2)
    fault_pixel.plot(ax=ax, color="black", linewidth=0.8)
    _annotate_faults(ax, fault_pixel)
    ax.set_xlim(0, dem.shape[1])
    ax.set_ylim(dem.shape[0], 0)
    ax.set_title("Rotated DEM matrix + same faults/study area in pixel coordinates")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _to_pixel_geometries(
    gdf: gpd.GeoDataFrame,
    affine_transform,
    height: int | None = None,
    width: int | None = None,
    flip_vertical: bool = False,
    flip_horizontal: bool = False,
) -> gpd.GeoDataFrame:
    inverse = ~affine_transform

    def project_geometry(geom):
        def project_xy(x, y, z=None):
            col, row = inverse * (x, y)
            if flip_vertical and height is not None:
                row = height - row
            if flip_horizontal and width is not None:
                col = width - col
            return col, row

        return shapely_transform(project_xy, geom)

    pixel_gdf = gdf.copy()
    pixel_gdf["geometry"] = pixel_gdf.geometry.apply(project_geometry)
    pixel_gdf = pixel_gdf.set_crs(None, allow_override=True)
    return pixel_gdf


def _annotate_faults(ax, fault: gpd.GeoDataFrame) -> None:
    name_col = "display_name" if "display_name" in fault.columns else "display_na" if "display_na" in fault.columns else None
    if name_col is None:
        return
    seen: set[str] = set()
    preferred = {
        "Wenchuan-Maowen fault",
        "Beichuan-Yinxiu fault",
        "Anxian-Guanxian fault",
        "Guanxian-Jiangyou fault",
        "Longquanshan fault",
        "Huya fault",
        "Xiaoyudong fault",
    }
    for _, row in fault.iterrows():
        label = str(row.get(name_col, "") or "").strip()
        if not label or label == "unnamed_fault" or label in seen:
            continue
        if label not in preferred and len(seen) >= 12:
            continue
        seen.add(label)
        point = row.geometry.representative_point()
        ax.text(
            point.x,
            point.y,
            label,
            fontsize=7,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.5},
        )


def _save_array(path: Path, array: np.ndarray) -> None:
    np.save(path, np.asarray(array, dtype=np.float32))


def _write_summary(artifacts: dict[str, object]) -> None:
    lines = [
        "# Demo3 Alignment Check",
        "",
        f"- Source DEM: `{DATA_DIR / 'demo3_real_dem.tif'}`",
        "- Region: Eastern Tibetan Plateau, Minjiang - Longmenshan - western Sichuan Basin",
        "- DEM source: Mapzen Terrarium tiles, cached locally",
        f"- Fault source: CN-faults from GMT China geospatial data v0.4.0, clipped to `{DATA_DIR / 'demo3_faults.shp'}`",
        f"- Study area: `{DATA_DIR / 'demo3_study_area.shp'}`",
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
