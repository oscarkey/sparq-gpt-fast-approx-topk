"""Benchmarks the speed of SparQ relative to the dense baseline."""

# Copyright (c) Graphcore 2024
# All rights reserved.
# This source code is licensed under the BSD-3 license,
# see the LICENSE file in the root directory of this source tree.

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from benchmark import Benchmark, run_or_load
from sparq import RKForCompressionRatio, SparQArgs
from theoretical_speedups import h100, speedup_theoretical_time_in_attn

expected_gpu = "H100"

base_config = dict(model="llama27bchat", quant=None, compile=True, gpu=expected_gpu)
prompt_lengths = [16] + [4096 * x for x in [1, 2, 4, 6, 8, 10]]

benchmarks = {}
benchmarks["dense"] = [
    Benchmark(**base_config, attention="dense", prompt_length=prompt_length)
    for prompt_length in prompt_lengths
]
benchmarks["SparQ (torch.topk)"] = [
    Benchmark(
        **base_config,
        attention="sparq",
        prompt_length=prompt_length,
        sparq=SparQArgs(rk=RKForCompressionRatio(8)),
    )
    for prompt_length in prompt_lengths
]
benchmarks[f"SparQ (approx top-k, $k_b=1$)"] = [
    Benchmark(
        **base_config,
        attention="sparq",
        prompt_length=prompt_length,
        sparq=SparQArgs(
            rk=RKForCompressionRatio(8), approx_topk_j=1, approx_topk_mtb=False
        ),
    )
    for prompt_length in prompt_lengths
]
benchmarks[f"SparQ (approx top-k, $k_b=2$)"] = [
    Benchmark(
        **base_config,
        attention="sparq",
        prompt_length=prompt_length,
        sparq=SparQArgs(
            rk=RKForCompressionRatio(8), approx_topk_j=2, approx_topk_mtb=False
        ),
    )
    for prompt_length in prompt_lengths
]
results = {k: [run_or_load(b) for b in bs] for k, bs in benchmarks.items()}

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{amsfonts}")
plt.rc("font", family="serif", serif="CMU Serif")
squashed_legend_params = {
    "handlelength": 1.0,
    "handletextpad": 0.5,
    "labelspacing": 0.3,
    "borderaxespad": 0.2,
    "borderpad": 0.25,
    "columnspacing": 0.7,
}

fig, axes = plt.subplots(ncols=2, figsize=(7, 2.0))
time_ax: Axes = axes[0]
speedup_ax: Axes = axes[1]
labels_done = False
colors = [
    (0.0, 0.0, 0.0),
    (0.92907237, 0.68878959, 0.50411509),
    (0.75861834, 0.25356035, 0.40663694),
    (0.29408557, 0.13721193, 0.38442775),
]
markers = ["o", "s", "^", "P", "X"]
time_ax_handles = []

for i, ((label, rs), color, marker) in enumerate(zip(results.items(), colors, markers)):
    means = [r.secs_per_token_mean for r in rs]
    stds = [r.secs_per_token_std for r in rs]
    (line,) = time_ax.plot(
        prompt_lengths, means, marker=marker, label=label, color=color
    )
    time_ax_handles.append(line)

    if label != "dense":
        speedups = [
            dense_r.secs_per_token_mean / sparq_t
            for dense_r, sparq_t in zip(results["dense"], means)
        ]
        speedup_ax.plot(prompt_lengths, speedups, marker=marker, color=color)

        print(label)
        for prompt_length, speedup in zip(prompt_lengths, speedups):
            print(f"{prompt_length}, {speedup:.2f}")
        print()

    if i == len(results) - 1:
        (theoretical_speedup_line,) = speedup_ax.plot(
            prompt_lengths,
            [
                speedup_theoretical_time_in_attn(
                    b, sr, platform=h100, model_config_name="7B"
                )
                for b, sr in zip(benchmarks[label], rs)
            ],
            color="black",
            linestyle="--",
            label="theoretical max speedup",
        )

time_ax.set_xlabel("prompt length")
time_ax.set_ylabel("secs per token")
time_ax.set_xlim(left=min(prompt_lengths))
time_ax.set_ylim(bottom=0)
time_ax.set_xticks([16, 20_000, 40_000])
legend_1 = time_ax.legend(
    handles=time_ax_handles[:2], loc="upper left", **squashed_legend_params
)
legend_2 = time_ax.legend(
    handles=time_ax_handles[2:], loc="lower right", **squashed_legend_params
)
time_ax.add_artist(legend_1)

speedup_ax.axhline(1.0, color="black")
speedup_ax.set_xlabel("prompt length")
speedup_ax.set_ylabel("speedup over dense")
speedup_ax.set_xlim(left=min(prompt_lengths))
speedup_ax.set_yticks([1, 1.5, 2.0])
speedup_ax.set_xticks([16, 20_000, 40_000])
speedup_ax.legend(loc="upper left", **squashed_legend_params)

plt.tight_layout()
figure_dir = Path("figures")
figure_dir.mkdir(exist_ok=True)
plt.savefig(figure_dir / "speedup_benchmark.png", dpi=300)
plt.savefig(figure_dir / "speedup_benchmark.pdf")
plt.close()
