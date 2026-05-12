"""Backward-compatible wrapper.

推荐入口是:

    python runner.py

然后在 config.ini 的 [Run] 中设置 mode = k_sensitivity。
"""

from ga_lem_inverter.runner import main


if __name__ == "__main__":
    raise SystemExit(main(default_mode="k_sensitivity"))
