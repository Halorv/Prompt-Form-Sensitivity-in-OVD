#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 workbuddy3.md 的三条改动生成可追溯的证据文件。

产出两个文件：
  results/RUNNER_UP_MARGINS.json      —— 附录 A.5 的「模板 vs 次差 prompt」逐类差距
  results/THRESHOLD_SENSITIVITY.json  —— 附录里的阈值敏感性计数

审计（paper-claim-audit）只能核对"能在结果文件里找到的数字"。
这两组数字原先只存在于对话里，直接写进论文会在审计时被标为 untraceable。
"""
import json
import os

CLASSES = ["person", "car", "dog", "cat", "chair", "cup", "bottle", "book"]


def find(rel):
    for d in ("results", "results/archive_superseded", "results/archive_80query"):
        p = os.path.join(d, rel)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(rel)


# ---------------------------------------------------------------- A.5 差距表
glip = json.load(open(find("glip_8class_ap50_summary.json"), encoding="utf-8"))["results"]
gdino = json.load(open(find("gdino_v11_summary.json"), encoding="utf-8"))["per_prompt_ap50"]


def margins(src, get, model):
    out = {}
    for c in CLASSES:
        ap = get(src[c])
        order = sorted(ap, key=ap.get)
        tpl = f"a photo of a {c}"
        rank = order.index(tpl) + 1
        second = ap[order[1]] if rank == 1 else ap[order[0]]
        out[c] = {
            "template_ap50": ap[tpl],
            "second_worst_ap50": second,
            "margin_template_minus_second": ap[tpl] - second,
            "rank": rank,
            "n_prompts": len(ap),
            "tied_within_0.005": abs(ap[tpl] - second) <= 0.005,
        }
    n_worst = sum(1 for v in out.values() if v["rank"] == 1)
    return {
        "model": model,
        "n_classes": len(CLASSES),
        "n_template_worst": n_worst,
        "n_tied_within_0.005": sum(1 for v in out.values() if v["rank"] == 1 and v["tied_within_0.005"]),
        "per_class": out,
    }


margins_out = {
    "schema": "runner_up_margins_v1",
    "description": (
        "Per-class AP50 of the CLIP template and of the runner-up prompt, used by "
        "the ordinal 'worst prompt form' claim (Appendix A.5). Derived from the "
        "same per-prompt AP50 table as PER_PROMPT_AP50.json."
    ),
    "sources": {
        "GLIP": "results/glip_8class_ap50_summary.json",
        "GDino": "results/archive_superseded/gdino_v11_summary.json",
    },
    "models": {
        "GLIP": margins(glip, lambda v: {p: x["ap50"] for p, x in v["per_prompt"].items()}, "GLIP-T"),
        "GDino": margins(gdino, lambda v: {p: x["ap50_all"] for p, x in v.items()}, "Grounding DINO"),
    },
}

# ---------------------------------------------------- 阈值敏感性
C = json.load(open("results/CANONICAL_8CLASS.json", encoding="utf-8"))["models"]


def classify(db, dl, lo, hi):
    if abs(db) < 0.01:
        return "N/A"
    if db <= -0.01:
        return "NEX"
    r = dl / db
    if r < lo:
        return "TAG"
    if r <= hi:
        return "MIX"
    return "TS"


sens = {}
for lo, hi in [(0.30, 0.70), (0.25, 0.75), (0.20, 0.80)]:
    key = f"{lo:.2f}/{hi:.2f}"
    sens[key] = {}
    for m, per in C.items():
        counts = {}
        per_class = {}
        for c, v in per.items():
            k = classify(v["delta_bare"], v["delta_len"], lo, hi)
            counts[k] = counts.get(k, 0) + 1
            per_class[c] = k
        sens[key][m] = {"counts": counts, "per_class": per_class}

sens_out = {
    "schema": "threshold_sensitivity_v1",
    "description": (
        "Mechanism-label counts under three threshold settings, supporting the "
        "sensitivity paragraph in the appendix. Rule: |delta_bare| < 0.01 -> N/A; "
        "delta_bare <= -0.01 -> NEX; otherwise ratio = delta_len/delta_bare is "
        "binned as <lo -> TAG, [lo,hi] -> MIX, >hi -> TS."
    ),
    "source": "results/CANONICAL_8CLASS.json",
    "thresholds": ["0.30/0.70", "0.25/0.75", "0.20/0.80"],
    "results": sens,
}

for path, obj in [("results/RUNNER_UP_MARGINS.json", margins_out),
                  ("results/THRESHOLD_SENSITIVITY.json", sens_out)]:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print("wrote", path)

print("\n--- 模板最差计数 ---")
for m, v in margins_out["models"].items():
    print(f"  {m:6s} 最差 {v['n_template_worst']}/8   并列(≤0.005) {v['n_tied_within_0.005']}")

print("\n--- 阈值敏感性 ---")
for t, models in sens["0.30/0.70"].items():
    pass
for t in sens_out["thresholds"]:
    row = {m: sens[t][m]["counts"] for m in sens[t]}
    print(f"  {t}: {row}")
