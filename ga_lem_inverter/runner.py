"""Unified command-line runner for non-code users."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from importlib import import_module
from pathlib import Path
from typing import Callable

from ga_lem_inverter.config import AppConfig, UserConfigError, load_app_config
from ga_lem_inverter.outputs import RunContext, create_run_context, finalize_run, write_metrics


Workflow = Callable[[object, RunContext], dict]

WORKFLOWS: dict[str, tuple[str, str]] = {
    "main": ("ga_lem_inverter.workflows.main_inversion", "run_main_workflow"),
    "synthetic": ("ga_lem_inverter.workflows.synthetic", "run_synthetic_workflow"),
    "k_sensitivity": ("ga_lem_inverter.workflows.k_sensitivity", "run_k_sensitivity_workflow"),
}


def main(default_mode: str | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GA-LEM-Inverter unified runner. 推荐普通用户直接运行: python runner.py",
    )
    parser.add_argument("--config", default="config.ini", help="配置文件路径，默认 config.ini")
    parser.add_argument(
        "--mode",
        choices=sorted(WORKFLOWS),
        help="临时覆盖 [Run] mode；普通用户建议直接改 config.ini。",
    )
    args = parser.parse_args()

    try:
        app_config = load_app_config(args.config)
        if args.mode:
            app_config.parser["Run"]["mode"] = args.mode
            app_config = AppConfig(app_config.parser, app_config.path, args.mode)
        elif default_mode:
            app_config.parser["Run"]["mode"] = default_mode
            app_config = AppConfig(app_config.parser, app_config.path, default_mode)

        context = create_run_context(app_config.path, app_config.parser, app_config.mode)
        _configure_logging(context)
        _print_diagnostics(app_config, context)

        workflow = _load_workflow(app_config.mode)
        metrics = workflow(app_config.parser, context)
        if metrics:
            write_metrics(context, "workflow_metrics.json", _json_safe(metrics))
        finalize_run(context, "success", "运行成功完成。")
        print(f"\n运行完成。结果目录: {context.root}")
        print(f"优先查看: {context.root / 'summary.md'}")
        return 0

    except UserConfigError as exc:
        print("\n配置需要调整：")
        print(f"  {exc}")
        print("\n请打开 config.ini 按中文注释修改后，再运行 python runner.py。")
        return 2
    except KeyboardInterrupt:
        print("\n用户中断运行。")
        return 130
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        context = locals().get("context")
        if isinstance(context, RunContext):
            finalize_run(context, "failed", f"缺少 Python 依赖: {missing}")
        print("\n当前 Python 环境缺少依赖：")
        print(f"  {missing}")
        print("\n请先按 README 运行环境安装脚本，然后在激活后的环境里执行 python runner.py。")
        return 3
    except Exception as exc:
        context = locals().get("context")
        if isinstance(context, RunContext):
            logging.exception("Workflow failed")
            finalize_run(context, "failed", str(exc))
            log_hint = context.logs_dir / "run.log"
            print("\n运行失败：")
            print(f"  {exc}")
            print(f"详细日志: {log_hint}")
            print("如果你不是代码用户，优先检查 config.ini 的输入路径、scale_factor、GA 参数和 DEM 尺寸。")
        else:
            print("\n运行失败：")
            print(f"  {exc}")
            print("程序尚未创建输出目录。请先检查 config.ini 是否存在、路径是否正确。")
            traceback.print_exc()
        return 1


def _load_workflow(mode: str) -> Workflow:
    module_name, function_name = WORKFLOWS[mode]
    module = import_module(module_name)
    return getattr(module, function_name)


def _configure_logging(context: RunContext) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(context.logs_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    root.addHandler(stream_handler)
    context.add_artifact(context.logs_dir / "run.log")


def _print_diagnostics(app_config: AppConfig, context: RunContext) -> None:
    config = app_config.parser
    mode = app_config.mode
    terrain = Path(config.get("Data", "terrain_path", fallback=""))
    fault = config.get("Data", "fault_shp_path", fallback="").strip()
    study_area = config.get("Data", "study_area_shp_path", fallback="").strip()

    if mode == "main" and terrain.exists():
        shape_hint = _dem_shape_hint(terrain)
    elif mode == "main":
        shape_hint = "DEM 文件不存在，稍后会停止并提示"
    else:
        section = "Synthetic" if mode == "synthetic" else "KSensitivity"
        shape_hint = config.get(section, "shape", fallback=config.get("Synthetic", "shape", fallback="64,64"))

    print("\nGA-LEM-Inverter 运行诊断")
    print("=" * 42)
    print(f"实验模式: {mode}")
    print(f"配置文件: {app_config.path}")
    print(f"输出目录: {context.root}")
    print(f"DEM: {terrain} ({'存在' if terrain.exists() else '不存在/本模式可不需要'})")
    print(f"断层 Shapefile: {fault or '未配置，将跳过'}")
    print(f"研究区 Shapefile: {study_area or '未配置，将跳过'}")
    print(f"网格大小: {shape_hint}")
    print(f"scale_factor/K: {config.get('Optimization', 'scale_factor', fallback='')}")
    print(
        "GA 规模: "
        f"population={config.get('Optimization', 'population_size', fallback=config.get('GeneticAlgorithm', 'ga_pop_size', fallback=''))}, "
        f"max_iter={config.get('Optimization', 'max_iterations', fallback=config.get('GeneticAlgorithm', 'ga_max_iter', fallback=''))}"
    )
    print(f"n_jobs: {config.get('Optimization', 'n_jobs', fallback=config.get('GeneticAlgorithm', 'n_jobs', fallback='1'))}")
    print(f"预设: {config.get('Run', 'preset', fallback='demo')}")
    print("=" * 42)


def _dem_shape_hint(path: Path) -> str:
    try:
        import rasterio

        with rasterio.open(path) as src:
            return f"{src.height} x {src.width}"
    except Exception:
        try:
            import numpy as np

            arr = np.load(path)
            return f"{arr.shape[0]} x {arr.shape[1]}"
        except Exception:
            return "无法读取尺寸，运行时会进一步诊断"


def _json_safe(metrics: dict) -> dict:
    safe = {}
    for key, value in metrics.items():
        try:
            if hasattr(value, "item"):
                value = value.item()
            safe[key] = value
        except Exception:
            safe[key] = str(value)
    return safe


if __name__ == "__main__":
    raise SystemExit(main())
