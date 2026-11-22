#!/usr/bin/env python3
"""Emit results/ORACLE_TABLE.json — explicit aggregates for paper Table 2.

`m1_v2_v7_analysis.json` stores only per-class AP50 dicts plus `S3_mean_eval`.
The paper quotes five aggregates (S0/S1/S2-DefA/S2-DefB/S3) and four gaps that
exist in NO stored field, so a claim audit reading the raw file cannot trace
them and may report them as missing evidence.

The aggregates are the unweighted mean over the 8 classes — the same rule the
analysis script uses for `S3_mean_eval` (m1_v2_v7_analysis.py line 299).
"""
import json

d = json.load(open("results/m1_v2_v7_analysis.json"))

SERIES = {
    "S0": ("S0_eval", "Default CLIP template 'a photo of a X'"),
    "S1": ("S1_eval", "Per-class best static prompt (selection split, hindsight)"),
    "S2_DefA": ("S2_defA_oracle", "Per-image oracle, strict definition"),
    "S2_DefB": ("S2_defB_oracle", "Per-image oracle, loose definition"),
    "S3": ("S3_eval", "Global bare class name"),
}

strategies, per_class = {}, {}
for name, (key, desc) in SERIES.items():
    vals = d[key]
    per_class[name] = vals
    strategies[name] = {
        "description": desc,
        "source_field": key,
        "mean_ap50": sum(vals.values()) / len(vals),
        "n_classes": len(vals),
    }

# Sanity: reproduce the field the original script did store.
assert abs(strategies["S3"]["mean_ap50"] - d["S3_mean_eval"]) < 1e-9, "S3 mismatch"

gaps = {
    "S2_DefB_minus_S3": strategies["S2_DefB"]["mean_ap50"] - strategies["S3"]["mean_ap50"],
    "S2_DefA_minus_S3": strategies["S2_DefA"]["mean_ap50"] - strategies["S3"]["mean_ap50"],
    "S2_DefB_minus_S1": strategies["S2_DefB"]["mean_ap50"] - strategies["S1"]["mean_ap50"],
    "S3_minus_S1": strategies["S3"]["mean_ap50"] - strategies["S1"]["mean_ap50"],
}

out = {
    "schema": "oracle_table_v1",
    "description": (
        "Explicit aggregates for paper Table 2 (oracle upper bound). "
        "Values are the unweighted mean of the per-class AP50 dicts stored in "
        "m1_v2_v7_analysis.json, matching that script's own S3_mean_eval rule."
    ),
    "source": "results/m1_v2_v7_analysis.json",
    "aggregation": "unweighted mean over the 8 COCO classes",
    "split": d.get("split"),
    "strategies": strategies,
    "gaps": gaps,
    "per_class": per_class,
}

with open("results/ORACLE_TABLE.json", "w") as f:
    json.dump(out, f, indent=2)

for k, v in strategies.items():
    print(f"{k:9s} {v['mean_ap50']:.6f}")
for k, v in gaps.items():
    print(f"{k:18s} {v*100:+.4f} AP points")
print("\nwrote results/ORACLE_TABLE.json")
