#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 服务器上没有图形界面，用 Agg 后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOG_FILE = "training_log.txt"   # 你的日志文件名
PLOT_EVERY = 100                # 每 500 个 epoch 取一个平均点
OUT_FILE = "loss_curve.png"     # 输出图片名


def load_losses(log_file):
    losses = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                loss = float(obj["loss"])
                losses.append(loss)
            except Exception as e:
                # 有脏行就跳过
                print("Skip line:", line, "Error:", e)
                continue
    return losses


def compute_chunked_losses(losses, chunk_size):
    """按 chunk_size（比如 100）划分，并对每一段取平均"""
    losses = np.asarray(losses, dtype=float)
    n_chunks = len(losses) // chunk_size
    if n_chunks == 0:
        return [], []

    # 只保留能整除的部分
    losses = losses[: n_chunks * chunk_size]
    # 形状变成 (n_chunks, chunk_size)，对 axis=1 求平均
    chunk_losses = losses.reshape(n_chunks, chunk_size).mean(axis=1)

    # x 轴用真实的 epoch 编号：100, 200, 300, ...
    epochs = np.arange(1, n_chunks + 1) * chunk_size
    return epochs, chunk_losses


def plot_loss(epochs, chunk_losses, out_file):
    plt.figure()
    fig, ax = plt.subplots()

    # 模仿老师的 showPlot：只画一个折线
    ax.plot(epochs, chunk_losses)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Loss (avg every {PLOT_EVERY} epochs)")
    ax.set_title("Training Loss Curve")

    # y 轴用 MultipleLocator，和你原来 showPlot 里的风格类似
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)

    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_file}")


def main():
    losses = load_losses(LOG_FILE)
    if not losses:
        print("No loss data found in log file.")
        return

    epochs, chunk_losses = compute_chunked_losses(losses, PLOT_EVERY)
    if len(chunk_losses) == 0:
        print("Not enough data to form one chunk.")
        return

    plot_loss(epochs, chunk_losses, OUT_FILE)


if __name__ == "__main__":
    main()
