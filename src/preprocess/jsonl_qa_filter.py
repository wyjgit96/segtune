# jsonl_filter.py
# 功能：读取一个 JSONL 文件，按各指标的“分位数阈值”过滤样本。若某行在任一指标上低于该指标的分位数阈值，则剔除；全部指标达标才保留。
# 用法：python jsonl_filter.py input.jsonl output.jsonl
# 说明：将阈值（分位数，0~100）在下方全局变量中配置；阈值 p 的含义是“保留 >= 该指标 p 分位数 的样本”。
#      例如 p=5 表示剔除位于该指标底部 5% 的样本；p=95 表示仅保留该指标前 5%（非常严格）。

import json
import argparse
from typing import Any, Dict, List, Optional

# ——全局配置：指标与分位数阈值（单位：百分位，0~100）——
# 如果想给所有指标同一个阈值，可设置 DEFAULT_PERCENTILE，并让 THRESHOLDS=None。
DEFAULT_PERCENTILE = 5  # 示例：剔除每个指标的底部 5%
THRESHOLDS: Optional[Dict[str, float]] = None
# # 若要分别设置每个指标，请取消注释并按需修改（不需要就保持 THRESHOLDS=None 使用 DEFAULT_PERCENTILE）。
# THRESHOLDS = {
#     "audiobox-aesthetics.CE": 5,
#     "audiobox-aesthetics.CU": 5,
#     "audiobox-aesthetics.PC": 5,
#     "audiobox-aesthetics.PQ": 5,
#     "songeval.Coherence": 5,
#     "songeval.Musicality": 5,
#     "songeval.Memorability": 5,
#     "songeval.Clarity": 5,
#     "songeval.Naturalness": 5,
# }

# ——需要参与过滤的全部指标（键路径用“.”分隔）——
METRICS = [
    "audiobox-aesthetics.CE",
    "audiobox-aesthetics.CU",
    "audiobox-aesthetics.PC",
    "audiobox-aesthetics.PQ",
    "songeval.Coherence",
    "songeval.Musicality",
    "songeval.Memorability",
    "songeval.Clarity",
    "songeval.Naturalness",
]


def get_in(d: Dict[str, Any], path: str) -> Any:
    """按点路径取值，缺失返回 None。"""
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and x == x  # 排除 NaN


def percentile_value(sorted_vals: List[float], p: float) -> float:
    """
    计算分位数值（简单线性插值版本）。
    p: 0~100，返回使得 >=p% 样本在右侧的分位值。
    """
    n = len(sorted_vals)
    if n == 0:
        return float("inf")  # 无数据则后续比较恒为 False（视为无法达标）
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    # 采用“(n-1)*p/100”的位置并线性插值
    pos = (n-1) * (p/100.0)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1-frac) + sorted_vals[hi] * frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="输入 JSONL 文件路径")
    ap.add_argument("output", help="输出（过滤后）JSONL 文件路径")
    args = ap.parse_args()

    # 读取所有行
    rows: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # 为每个指标收集数值（忽略缺失/非数值），并计算对应的“达标下界值”（该指标的分位数阈值）
    per_metric_values: Dict[str, List[float]] = {m: [] for m in METRICS}
    for r in rows:
        for m in METRICS:
            v = get_in(r, m)
            if is_number(v):
                per_metric_values[m].append(float(v))

    # 计算各指标阈值（实际数值下界）
    metric_percentiles: Dict[str, float] = {}
    for m in METRICS:
        sorted_vals = sorted(per_metric_values[m])
        p = (THRESHOLDS[m] if THRESHOLDS and m in THRESHOLDS else DEFAULT_PERCENTILE)
        metric_percentiles[m] = percentile_value(sorted_vals, p)

    # 过滤：每行在所有指标上均需 v >= 阈值
    kept: List[Dict[str, Any]] = []
    for r in rows:
        ok = True
        for m in METRICS:
            v = get_in(r, m)
            # 缺失或非数值视为不达标；数值需 >= 该指标的分位下界
            if not is_number(v) or float(v) < metric_percentiles[m]:
                ok = False
                break
        if ok:
            kept.append(r)

    # 写回
    with open(args.output, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 可选：打印简要统计到控制台
    total, remain = len(rows), len(kept)
    print(f"总计 {total} 行，保留 {remain} 行（{(remain/total*100 if total else 0):.2f}%）。")


if __name__ == "__main__":
    main()
