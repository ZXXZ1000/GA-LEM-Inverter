"""FastScape 正演工具入口。

普通用户只需要：
1. 编辑同目录下的 ``forward_config.ini``。
2. 如需复杂时间函数，编辑同目录下的 ``uplift_time.py``。
3. 在终端运行：

    python forward_run.py

默认使用项目根目录下的 ``forward_config.ini``；也支持手动指定：

    python forward_run.py path/to/some_other_config.ini
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from forward_simulator.workflow import run_forward_simulation


def _resolve_config_path() -> Path:
    if len(sys.argv) >= 2:
        return Path(sys.argv[1]).expanduser().resolve()
    here = Path(__file__).resolve().parent
    candidates = [
        here / "forward_config.ini",
        here / "demo" / "forward" / "forward_config_demo.ini",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "未找到 forward_config.ini。请在项目根目录新建一个，或在命令行指定: "
        "python forward_run.py path/to/forward_config.ini"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config_path = _resolve_config_path()
    print(f"使用配置文件: {config_path}")
    context = run_forward_simulation(config_path)
    print("")
    print(f"正演完成。结果目录: {context.root}")
    print(f"优先查看: {context.root / 'summary.md'}")


if __name__ == "__main__":
    main()
