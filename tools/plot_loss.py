#!/usr/bin/env python3
"""
解析 train_smolvlm.log，绘制 loss 曲线并保存为 PNG。

用法:
    python tools/plot_loss.py runs/simvla_libero_small/train_smolvlm.log
    python tools/plot_loss.py runs/simvla_libero_small/train_smolvlm.log --out loss.png
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# 匹配形如:
#   14:12:19 | INFO | __main__ | [20/200000] loss=0.1234 lr_core=...
#   [iter 1000] loss=0.1234
#   step=1000, loss=0.1234
LOG_PATTERNS = [
    # SimVLA 格式: [step/total] loss=value
    re.compile(r"\[(\d+)/\d+\]\s+loss=([\d.eE+\-]+)"),
    # 通用格式
    re.compile(r"iter[=:\s]+(\d+).*?loss[=:\s]+([\d.eE+\-]+)", re.IGNORECASE),
    re.compile(r"step[=:\s]+(\d+).*?loss[=:\s]+([\d.eE+\-]+)", re.IGNORECASE),
    re.compile(r"\[(\d+)\].*?loss[=:\s]+([\d.eE+\-]+)", re.IGNORECASE),
]


def parse_log(log_path: str):
    steps, losses = [], []
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            for pat in LOG_PATTERNS:
                m = pat.search(line)
                if m:
                    try:
                        steps.append(int(m.group(1)))
                        losses.append(float(m.group(2)))
                    except ValueError:
                        pass
                    break
    return np.array(steps), np.array(losses)


def smooth(y, window=50):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def plot(steps, losses, out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(12, 5))

    # 原始曲线（浅色）
    ax.plot(steps, losses, alpha=0.25, color="#4C8BF5", linewidth=0.8, label="raw")

    # 平滑曲线
    win = max(1, len(losses) // 100)
    if len(losses) >= win * 2:
        s_losses = smooth(losses, win)
        s_steps = steps[win - 1: win - 1 + len(s_losses)]
        ax.plot(s_steps, s_losses, color="#E84545", linewidth=1.8, label=f"smooth (w={win})")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 标注最低点
    if len(losses) > 0:
        idx = np.argmin(losses)
        ax.annotate(
            f"min={losses[idx]:.4f}\n@iter {steps[idx]}",
            xy=(steps[idx], losses[idx]),
            xytext=(steps[idx], losses[idx] + (losses.max() - losses.min()) * 0.15),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"),
            color="gray",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"图片保存到: {out_path}")
    print(f"共解析 {len(steps)} 条记录")
    if len(losses) > 0:
        print(f"loss 范围: {losses.min():.4f} ~ {losses.max():.4f}")
        print(f"最新 loss: {losses[-1]:.4f}  (iter {steps[-1]})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="日志文件路径")
    parser.add_argument("--out", default="", help="输出 PNG 路径（默认和日志同目录）")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"错误: 找不到文件 {log_path}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or str(log_path.with_suffix(".png"))

    steps, losses = parse_log(str(log_path))
    if len(steps) == 0:
        print("未解析到任何 loss 数据，请检查日志格式。")
        print("前几行内容：")
        with open(log_path) as f:
            for i, line in enumerate(f):
                print(f"  {line.rstrip()}")
                if i > 10:
                    break
        sys.exit(1)

    plot(steps, losses, out_path, title=f"Training Loss — {log_path.name}")


if __name__ == "__main__":
    main()
