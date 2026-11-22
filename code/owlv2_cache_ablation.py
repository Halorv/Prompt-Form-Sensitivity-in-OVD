#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OWLv2 缓存推理路径 vs 完整前向 —— 单变量对照
=============================================

背景
----
`m6_40class.py` 为了加速，手写了一条「缓存视觉特征」的推理路径：

    image_embedder(img)  → image_feats, feature_map      （每张图只算一次）
    text_model(text)     → pooled → text_projection → L2 归一化
    class_predictor(image_feats, query_embeds, query_mask) → pred_logits
    box_predictor(image_feats, feature_map)               → pred_boxes

这条路径**手工复刻**了 `Owlv2Model.forward` 的内部步骤。如果复刻有偏差
（例如 text_projection 做两次、归一化缺失或多余、query_mask 构造不同），
结果就会和官方的 `model(**inputs)` 不同。

论文中 OWLv2 的全部数字（Table 1、41 类实验）都来自这条缓存路径，
而 `alignment_test_72.py` 验证的是 v10 的检测结果 —— 它没有覆盖这条路径。

本脚本只做一件事：对同一张图、同一条 prompt，
比较「完整前向」与「缓存路径」的 logits 是否一致。

用法
----
    python3 owlv2_cache_ablation.py --quick          # 5 张图，30 秒
    python3 owlv2_cache_ablation.py --n-images 50

判据
----
    max|Δlogit| < 1e-4   → 缓存路径正确，0.775 vs 0.672 另有原因
    max|Δlogit| > 0.01   → 缓存路径复刻有误，论文 OWLv2 数字需要重跑
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor


def center_to_corners(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


# ---------------------------------------------------------------- 完整前向
def forward_full(model, processor, image, text, device):
    """官方路径：一条 prompt 一次完整前向。"""
    inputs = processor(text=[[text]], images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    return out.logits[0][:, 0], out.pred_boxes[0]


# ---------------------------------------------------------------- 缓存路径
# 以下三个函数逐字复刻 experiments/m6_40class.py 的实现，
# 以便复现论文所用的那条路径。
def cache_image_features(model, processor, image, device):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        image_embeds, vision_outputs = model.image_embedder(pixel_values=pixel_values)
    b, h, w, d = image_embeds.shape
    image_feats = image_embeds.reshape(b, h * w, d)
    return image_feats, image_embeds


def run_from_cache(model, processor, image_feats, feature_map, text, device,
                   apply_projection=True, apply_norm=True):
    """缓存路径。两个开关用来定位到底哪一步导致偏差。"""
    text_inputs = processor(text=[text], return_tensors="pt")
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)

    with torch.no_grad():
        text_outputs = model.owlv2.text_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        text_embeds = text_outputs[1]                     # pooled

        if apply_projection:
            text_embeds = model.owlv2.text_projection(text_embeds)
        if apply_norm:
            text_embeds = text_embeds / torch.linalg.norm(
                text_embeds, dim=-1, keepdim=True)

        query_embeds = text_embeds.reshape(1, input_ids.shape[0], text_embeds.shape[-1])
        input_ids_r = input_ids.reshape(1, input_ids.shape[0], input_ids.shape[-1])
        query_mask = input_ids_r[..., 0] > 0

        pred_logits, _ = model.class_predictor(image_feats, query_embeds, query_mask)
        pred_boxes = model.box_predictor(image_feats, feature_map)

    return pred_logits[0][:, 0], pred_boxes[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--n-images", type=int, default=5)
    ap.add_argument("--img-ids-from", default="results/m1_v2_raw_detections.json")
    ap.add_argument("--img-dir", default=os.environ.get("COCO_IMG_DIR", "data/val2017"))
    ap.add_argument("--prompts", default="person,person.,a photo of a person,there is a person",
                    help="逗号分隔。注意 Grounding DINO 用句点，OWLv2 不用；这里只是测路径一致性")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results/owlv2_cache_ablation.json")
    args = ap.parse_args()
    if args.quick:
        args.n_images = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}")

    print(f"[model] loading {args.model} ...")
    processor = Owlv2Processor.from_pretrained(args.model)
    model = Owlv2ForObjectDetection.from_pretrained(
        args.model, torch_dtype=torch.float32).to(device).eval()
    print("[model] ready")

    # 图像
    if os.path.isfile(args.img_ids_from):
        d = json.load(open(args.img_ids_from))
        ids = (d.get("selected_img_ids") or [])[: args.n_images]
        print(f"[ids] {len(ids)} 张图来自 {args.img_ids_from}")
    else:
        ids = []
        for fn in sorted(os.listdir(args.img_dir))[: args.n_images]:
            ids.append(int(os.path.splitext(fn)[0]))
        print(f"[ids] {len(ids)} 张图来自目录（未找到 {args.img_ids_from}）")

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]

    # 三种缓存配置，用来定位差异来源
    configs = [
        ("cache(projection+norm)", dict(apply_projection=True,  apply_norm=True)),
        ("cache(projection only)", dict(apply_projection=True,  apply_norm=False)),
        ("cache(norm only)      ", dict(apply_projection=False, apply_norm=True)),
        ("cache(neither)        ", dict(apply_projection=False, apply_norm=False)),
    ]
    stats = {name: {"max": 0.0, "sum": 0.0, "n": 0} for name, _ in configs}
    box_stats = {name: 0.0 for name, _ in configs}

    for idx, img_id in enumerate(ids, 1):
        path = os.path.join(args.img_dir, f"{img_id:012d}.jpg")
        if not os.path.isfile(path):
            print(f"  [skip] {path}")
            continue
        image = Image.open(path).convert("RGB")
        image_feats, feature_map = cache_image_features(model, processor, image, device)

        for text in prompts:
            logits_full, boxes_full = forward_full(model, processor, image, text, device)
            for name, kw in configs:
                logits_c, boxes_c = run_from_cache(
                    model, processor, image_feats, feature_map, text, device, **kw)
                dl = (logits_full - logits_c).abs()
                s = stats[name]
                s["max"] = max(s["max"], float(dl.max()))
                s["sum"] += float(dl.sum())
                s["n"] += dl.numel()
                box_stats[name] = max(box_stats[name],
                                      float((boxes_full - boxes_c).abs().max()))

        print(f"  [{idx}/{len(ids)}] done")

    print("\n" + "=" * 78)
    print("完整前向  vs  缓存路径  —— logits 差异")
    print("=" * 78)
    print(f"{'配置':<26} {'max|Δlogit|':>14} {'mean|Δlogit|':>14} {'max|Δbox|':>12}")
    print("-" * 78)
    for name, _ in configs:
        s = stats[name]
        mean = s["sum"] / max(s["n"], 1)
        print(f"{name:<26} {s['max']:>14.3e} {mean:>14.3e} {box_stats[name]:>12.3e}")

    best = min(configs, key=lambda kv: stats[kv[0]]["max"])
    print("\n结论：")
    if stats[best[0]]["max"] < 1e-4:
        print(f"  「{best[0]}」与完整前向一致（<1e-4）。")
        if best[1] != configs[0][1]:
            print("  ⚠️ 但 m6 用的是 projection+norm —— 说明当前实现与官方不符，"
                  "论文 OWLv2 数字需要重跑。")
        else:
            print("  缓存路径复刻正确，0.775 vs 0.672 的差异另有来源。")
    else:
        print("  所有配置都与完整前向不一致，需要进一步检查 image_embedder 的用法。")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({name: stats[name] | {"max_box_diff": box_stats[name]}
               for name, _ in configs},
              open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
