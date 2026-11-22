#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OWLv2 批量大小单变量对照实验
================================

回答两个问题：

  Q1  结构层面：把 N 条 prompt 一次前向，与逐条前向，同一个 (图像位置, prompt)
      对的 logits 是否相同？
  Q2  指标层面：如果 logits 不同，AP50 会差多少？

实验设计（严格单变量）
----------------------
A 臂：model(**processor(text=[prompts], images=img))          # N 条一起
B 臂：model(**processor(text=[[p]],     images=img)) 逐条     # 每条单独

两臂使用完全相同的模型实例、同一张图、同一组 prompt、同一套后处理，
唯一的变量就是文本批量大小。

用法
----
# 先花 30 秒做个冒烟测试（5 张图，只比 logits）
python3 owlv2_batch_ablation.py --quick

# 完整跑（100 张图，含 AP50）
python3 owlv2_batch_ablation.py --n-images 100 --classes person \
    --img-ids-from results/m1_v2_raw_detections.json

# 只用 logits 对比，不算 AP50（快很多）
python3 owlv2_batch_ablation.py --n-images 50 --skip-ap

判据
----
  max|Δlogits| < 1e-4  →  结构预测成立，批量不影响输出
                          若此时 AP50 仍差很多，说明差异来自流水线而非批量
  max|Δlogits| > 1e-2  →  架构里存在未发现的耦合，需要深挖
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

from transformers import Owlv2ForObjectDetection, Owlv2Processor


# ---------------------------------------------------------------- prompts
# 十种 prompt 包装（{c} 替换为类名）。与论文的形式一致，但同义/复数类
# 在这里统一成通用模板，因为我们只关心「同一组 prompt 在两种批量下的差异」。
PROMPT_TEMPLATES = [
    "{c}",
    "a {c}",
    "a photo of a {c}",
    "a {c} in the scene",
    "the {c}",
    "there is a {c}",
    "a {c} standing",
    "an individual {c}",
    "a large {c}",
    "a small {c}",
]


def center_to_corners(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def postprocess(logits_1d, pred_boxes, img_hw, threshold, max_dets):
    """与 experiments/m6_40class.py 完全一致的后处理。"""
    h, w = img_hw
    scale = float(max(h, w))

    scores = torch.sigmoid(logits_1d).cpu().numpy()
    boxes = (center_to_corners(pred_boxes) * scale).cpu().numpy()

    mask = scores > threshold
    if not mask.any():
        return []
    fb, fs = boxes[mask], scores[mask]
    if len(fs) > max_dets:
        order = np.argsort(-fs)[:max_dets]
        fb, fs = fb[order], fs[order]

    out = []
    for i in range(len(fs)):
        x1, y1, x2, y2 = fb[i]
        out.append({
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(fs[i]),
        })
    return out


def load_image_ids(args):
    """优先使用论文实验所用的 500 张图，保证可比。"""
    if args.img_ids_from and os.path.isfile(args.img_ids_from):
        d = json.load(open(args.img_ids_from))
        ids = d.get("selected_img_ids") or d.get("img_ids") or []
        if ids:
            print(f"[ids] 从 {args.img_ids_from} 读取 {len(ids)} 个图像 ID")
            return [int(i) for i in ids[: args.n_images]]

    from pycocotools.coco import COCO

    coco = COCO(args.coco_ann)
    ids = sorted(coco.getImgIds())
    rng = np.random.RandomState(42)
    pick = sorted(rng.choice(ids, size=min(args.n_images, len(ids)), replace=False).tolist())
    print(f"[ids] 未指定 --img-ids-from，随机抽取 {len(pick)} 个（seed=42）")
    return pick


def coco_ap50(coco_gt, detections, img_ids, cat_id):
    """在给定图像子集上计算某一类的 AP50。"""
    if not detections:
        return float("nan")
    try:
        coco_dt = coco_gt.loadRes(detections)
    except Exception as e:
        print(f"  [warn] loadRes 失败：{e}")
        return float("nan")

    from pycocotools.cocoeval import COCOeval
    import contextlib
    import io

    E = COCOeval(coco_gt, coco_dt, "bbox")
    E.params.imgIds = sorted(img_ids)
    E.params.catIds = [cat_id]
    E.evaluate()
    E.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):   # summarize() 会打印 15 行
        E.summarize()
    # stats[1] = AP@IoU=0.50, all areas, maxDets=100；-1 表示该子集无 GT
    ap = float(E.stats[1])
    return ap if ap >= 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--classes", default="person",
                    help="逗号分隔，默认 person")
    ap.add_argument("--img-ids-from", default="results/m1_v2_raw_detections.json")
    ap.add_argument("--coco-ann", default=os.environ.get("COCO_ANN_FILE", "data/annotations/instances_val2017.json"))
    ap.add_argument("--img-dir", default=os.environ.get("COCO_IMG_DIR", "data/val2017"))
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--max-dets", type=int, default=100)
    ap.add_argument("--batch-all-classes", action="store_true",
                    help="把所有类别的 prompt 合成一个更长的大批量（默认不做）")
    ap.add_argument("--skip-ap", action="store_true", help="只比 logits，不算 AP50")
    ap.add_argument("--quick", action="store_true", help="5 张图 + 跳过 AP50")
    ap.add_argument("--out", default="results/owlv2_batch_ablation.json")
    args = ap.parse_args()

    if args.quick:
        args.n_images = 5
        args.skip_ap = True
        args.batch_all_classes = False

    if args.batch_all_classes and not args.skip_ap:
        # 多类合并成一个大批量时，检测结果的 prompt→类别映射会错位，
        # 这种情况下只做 logits 对比。
        print("[warn] --batch-all-classes 只做 logits 对比，自动开启 --skip-ap")
        args.skip_ap = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}  torch={torch.__version__}")

    # ---------------- model ----------------
    print(f"[model] loading {args.model} ...")
    t0 = time.time()
    processor = Owlv2Processor.from_pretrained(args.model)
    model = Owlv2ForObjectDetection.from_pretrained(
        args.model, torch_dtype=torch.float32
    ).to(device).eval()
    print(f"[model] ready in {time.time() - t0:.1f}s | "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    # 每个类一组 prompt；--batch-all-classes 时把多类合并成一个大批量
    prompts_per_class = {c: [t.format(c=c) for t in PROMPT_TEMPLATES] for c in classes}
    if args.batch_all_classes and len(classes) > 1:
        flat = [p for c in classes for p in prompts_per_class[c]]
    else:
        flat = prompts_per_class[classes[0]]
    N = len(flat)
    print(f"[plan] N = {N} 条 prompt / 类：{classes}")

    # ---------------- data ----------------
    img_ids = load_image_ids(args)

    coco_gt = None
    cat_ids = {}
    if not args.skip_ap:
        from pycocotools.coco import COCO
        print(f"[coco] loading annotations {args.coco_ann} ...")
        coco_gt = COCO(args.coco_ann)
        for c in classes:
            ids = coco_gt.getCatIds(catNms=[c])
            if not ids:
                print(f"[warn] 类别 {c} 不在 COCO 中，跳过 AP50")
            cat_ids[c] = ids[0] if ids else None

    # ---------------- accumulators ----------------
    diff_logits_max = 0.0
    diff_logits_mean_acc = 0.0
    diff_boxes_max = 0.0
    n_compared = 0
    n_pixels_flipped = 0     # |Δlogit| > 1e-3 的元素数
    n_total = 0

    dets_A = {c: {j: [] for j in range(len(prompts_per_class[c]))} for c in classes}
    dets_B = {c: {j: [] for j in range(len(prompts_per_class[c]))} for c in classes}

    used_ids = []
    t_start = time.time()

    for k, img_id in enumerate(img_ids, 1):
        path = os.path.join(args.img_dir, f"{img_id:012d}.jpg")
        if not os.path.isfile(path):
            continue
        image = Image.open(path).convert("RGB")
        hw = (image.height, image.width)
        used_ids.append(img_id)

        # ---- A 臂：N 条 prompt 一次前向 ----
        inputs_a = processor(text=[flat], images=image, return_tensors="pt")
        inputs_a = {kk: vv.to(device) for kk, vv in inputs_a.items()}
        with torch.no_grad():
            out_a = model(**inputs_a)
        logits_a = out_a.logits[0]        # [P, N]
        boxes_a = out_a.pred_boxes[0]     # [P, 4]

        # ---- B 臂：逐条前向 ----
        singles = []
        boxes_b = None
        for j, p in enumerate(flat):
            inp_b = processor(text=[[p]], images=image, return_tensors="pt")
            inp_b = {kk: vv.to(device) for kk, vv in inp_b.items()}
            with torch.no_grad():
                out_b = model(**inp_b)
            singles.append(out_b.logits[0][:, 0])   # [P]
            if j == 0:
                boxes_b = out_b.pred_boxes[0]
        logits_b = torch.stack(singles, dim=-1)     # [P, N]

        # ---- 比较 ----
        dl = (logits_a - logits_b).abs()
        diff_logits_max = max(diff_logits_max, float(dl.max()))
        diff_logits_mean_acc += float(dl.mean())
        n_compared += 1
        n_pixels_flipped += int((dl > 1e-3).sum())
        n_total += int(dl.numel())
        diff_boxes_max = max(diff_boxes_max, float((boxes_a - boxes_b).abs().max()))

        # ---- 收集检测（用于 AP50）----
        if not args.skip_ap:
            if args.batch_all_classes and len(classes) > 1:
                raise SystemExit("--batch-all-classes 与 --skip-ap 互斥用法，请分别跑")
            c = classes[0]
            for j in range(N):
                for arm, dets_src in (("A", logits_a), ("B", logits_b)):
                    dets = postprocess(dets_src[:, j], boxes_a if arm == "A" else boxes_b,
                                       hw, args.threshold, args.max_dets)
                    store = dets_A if arm == "A" else dets_B
                    for d in dets:
                        store[c][j].append(
                            {"image_id": img_id, "category_id": cat_ids[c],
                             "bbox": d["bbox"], "score": d["score"]}
                        )

        if k % 10 == 0 or k == len(img_ids):
            print(f"  [{k}/{len(img_ids)}] max|Δlogit| so far = {diff_logits_max:.3e}")

    elapsed = time.time() - t_start
    print(f"\n[done] {len(used_ids)} 张图，用时 {elapsed:.1f}s")

    # ---------------- 汇总：logits ----------------
    mean_dl = diff_logits_mean_acc / max(n_compared, 1)
    print("\n" + "=" * 78)
    print("Q1  结构层面：批量 vs 单条，同一个 (图像位置, prompt) 的 logits")
    print("=" * 78)
    print(f"  比较元素数        : {n_total:,}")
    print(f"  max |Δlogit|      : {diff_logits_max:.3e}")
    print(f"  mean|Δlogit|      : {mean_dl:.3e}")
    print(f"  |Δlogit|>1e-3 的数 : {n_pixels_flipped:,}  "
          f"({100.0 * n_pixels_flipped / max(n_total, 1):.4f}%)")
    print(f"  max |Δbox|        : {diff_boxes_max:.3e}")

    if diff_logits_max < 1e-4:
        verdict_q1 = "两臂数值一致 → 结构预测成立，批量大小不影响单个 (位置, prompt) 的分数"
    elif diff_logits_max < 1e-2:
        verdict_q1 = "存在小幅差异，可能来自 batch 形状导致的浮点差异，需要看 AP50 是否受影响"
    else:
        verdict_q1 = "存在显著差异 → 架构中存在未发现的耦合，必须深挖"
    print(f"\n  判定：{verdict_q1}")

    # ---------------- 汇总：AP50 ----------------
    report = {
        "model": args.model,
        "classes": classes,
        "n_images": len(used_ids),
        "prompts": flat,
        "logits": {
            "max_abs_diff": diff_logits_max,
            "mean_abs_diff": mean_dl,
            "n_elements": n_total,
            "n_gt_1e-3": n_pixels_flipped,
            "max_box_diff": diff_boxes_max,
            "verdict": verdict_q1,
        },
    }

    if not args.skip_ap and coco_gt is not None:
        print("\n" + "=" * 78)
        print("Q2  指标层面：两臂各自的 AP50")
        print("=" * 78)
        print(f"{'prompt':<28} {'AP50(A 批量)':>14} {'AP50(B 单条)':>14} {'Δ':>10}")
        print("-" * 78)

        per_prompt = []
        for c in classes:
            cid = cat_ids.get(c)
            if cid is None:
                continue
            for j, p in enumerate(prompts_per_class[c]):
                a = coco_ap50(coco_gt, dets_A[c][j], used_ids, cid)
                b = coco_ap50(coco_gt, dets_B[c][j], used_ids, cid)
                d = (a - b) if (a == a and b == b) else float("nan")
                per_prompt.append({"class": c, "prompt": p,
                                   "ap50_A_batched": a, "ap50_B_single": b, "delta": d})
                print(f"{p[:28]:<28} {a:>14.4f} {b:>14.4f} {d:>+10.4f}")

        if per_prompt:
            deltas = [abs(x["delta"]) for x in per_prompt if x["delta"] == x["delta"]]
            if deltas:
                print("-" * 78)
                print(f"{'max |ΔAP50|':<28} {'':>14} {'':>14} {max(deltas):>+10.4f}")
        report["ap50"] = per_prompt

    # ---------------- 最终结论 ----------------
    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)
    if diff_logits_max < 1e-4:
        print("  logits 两臂一致（<1e-4）。")
        print("  → 批量大小本身不改变输出，0.775 vs 0.672 的差异不能归因于批量。")
        print("  → 下一步：在同一脚本内复现两条流水线，逐个排除预处理/索引/后处理差异。")
    else:
        print("  logits 两臂存在差异，见上方数值。")
        print("  → 先确认差异是否足以改变 AP50（看 Q2 表），再决定是否需要隔离到具体算子。")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[out] 结果已写入 {args.out}")


if __name__ == "__main__":
    main()
