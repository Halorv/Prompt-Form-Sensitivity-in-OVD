#!/usr/bin/env python3
"""
Weighted AP50 — pure NumPy implementation that replicates pycocotools COCOeval
exactly (IoU, matching, 101-point interpolation), but supports per-image weights
for correct bootstrap resampling.

When all weights=1, this produces identical results to pycocotools COCOeval
(max|diff| < 1e-6 verified on 48 model×class×prompt combinations).

The key difference from pycocotools: when bootstrap resamples 500 images with
replacement, ~316 unique images survive. pycocotools' COCOeval.evaluate() does
  p.imgIds = list(np.unique(p.imgIds))
which deduplicates the image IDs, collapsing each image's weight to 1 regardless
of how many times it was sampled. This makes CIs artificially narrow.

weighted_ap50 takes a weights dict {img_id: count} and multiplies each image's
TP/FP contribution by its weight, and uses sum(weights * n_gt_per_image) as
the denominator. This correctly implements bootstrap with replacement.
"""
import numpy as np


def _bbox_to_xyxy(bbox):
    """[x, y, w, h] -> (x1, y1, x2, y2)"""
    x, y, w, h = bbox
    return x, y, x + w, y + h


def _iou_matrix(det_bboxes, gt_bboxes, gt_iscrowd):
    """Compute IoU between all detections and all GTs for one image.

    Args:
        det_bboxes: list of [x,y,w,h]
        gt_bboxes:  list of [x,y,w,h]
        gt_iscrowd: list of 0/1

    Returns:
        ious: np.ndarray shape (n_det, n_gt)
    """
    n_det = len(det_bboxes)
    n_gt = len(gt_bboxes)
    if n_det == 0 or n_gt == 0:
        return np.zeros((n_det, n_gt))

    ious = np.zeros((n_det, n_gt), dtype=np.float64)
    for i in range(n_det):
        dx, dy, dw, dh = det_bboxes[i]
        dx1, dy1, dx2, dy2 = dx, dy, dx + dw, dy + dh
        d_area = dw * dh
        for j in range(n_gt):
            gx, gy, gw, gh = gt_bboxes[j]
            gx1, gy1, gx2, gy2 = gx, gy, gx + gw, gy + gh

            ix1 = max(dx1, gx1)
            iy1 = max(dy1, gy1)
            ix2 = min(dx2, gx2)
            iy2 = min(dy2, gy2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih

            g_area = gw * gh
            if gt_iscrowd[j]:
                # crowd: IoU = inter / (det_area + eps)
                ious[i, j] = inter / (d_area + np.spacing(1))
            else:
                union = d_area + g_area - inter
                ious[i, j] = inter / (union + np.spacing(1))
    return ious


def _evaluate_img(det_list, gt_list, gt_iscrowd, ious, iou_thr,
                 area_rng=(0.0, 1e10), max_det=100):
    """Evaluate one image, one category, one IoU threshold.

    Replicates pycocotools evaluateImg:
    - Sort GTs: ignore-last (iscrowd → ignore=1, so they go last)
    - Sort dets: by score descending (mergesort for stability)
    - Cap dets at max_det
    - Greedy matching: for each det (in score order), find best GT
      - Skip GTs already matched (unless crowd)
      - If matched to regular GT and encountering ignore GT, stop
      - Track: dtMatches (which GT each det matched), gtMatches (which det each GT matched)
    - Unmatched dets outside area range → ignore

    Returns:
        dt_scores_sorted: scores in descending order (after max_det cap)
        dt_tp: array of 0/1 (1 = matched to non-ignore GT)
        dt_fp: array of 0/1 (1 = not matched, not ignored)
        gt_ignore: array of 0/1 for each GT (in sorted order)
        n_gt_valid: count of non-ignore GTs
    """
    # Sort GTs: ignore last (iscrowd=1 → ignore=1 → goes to end)
    gt_ignore_flags = [1 if c else 0 for c in gt_iscrowd]
    # Also apply area range filter for ignore
    for j in range(len(gt_list)):
        if gt_ignore_flags[j] == 0:
            # check area range
            gw, gh = gt_list[j][2], gt_list[j][3]
            area = gw * gh
            if area < area_rng[0] or area > area_rng[1]:
                gt_ignore_flags[j] = 1

    gt_sort = np.argsort(gt_ignore_flags, kind='mergesort')
    gt_sorted = [gt_list[i] for i in gt_sort]
    gt_ig_sorted = [gt_ignore_flags[i] for i in gt_sort]

    # Sort dets: highest score first (mergesort for stability, matching pycocotools)
    dt_scores = np.array([-d['score'] for d in det_list])
    dt_sort = np.argsort(dt_scores, kind='mergesort')
    # Cap at max_det
    dt_sort = dt_sort[:max_det]
    dt_sorted = [det_list[i] for i in dt_sort]
    dt_scores_sorted = [det_list[i]['score'] for i in dt_sort]

    n_det = len(dt_sorted)
    n_gt = len(gt_sorted)

    # Reorder ious to match sorted GT order
    if ious is not None and ious.size > 0:
        ious_sorted = ious[:, gt_sort]
        # But dets are also sorted differently, so we need to use the right indices
        # ious was computed with original det order. We need to map dt_sort.
        # Actually, ious is (n_det_orig, n_gt_orig). After gt_sort reorder, it's
        # (n_det_orig, n_gt_sorted). We need to index by dt_sort.
        ious_sorted = ious_sorted[dt_sort] if len(dt_sort) > 0 else ious_sorted
    else:
        ious_sorted = ious

    gtm = np.zeros(n_gt, dtype=np.int64)  # gt matched to which det
    dtm = np.zeros(n_det, dtype=np.int64)  # det matched to which gt
    dt_ig = np.zeros(n_det, dtype=np.float64)

    if ious_sorted is not None and ious_sorted.size > 0:
        for dind in range(n_det):
            iou = min(iou_thr, 1.0 - 1e-10)
            m = -1
            for gind in range(n_gt):
                # if this gt already matched and not crowd, continue
                if gtm[gind] > 0 and not gt_ig_sorted[gind]:
                    continue
                # if dt matched to reg gt, and on ignore gt, stop
                if m > -1 and gt_ig_sorted[m] == 0 and gt_ig_sorted[gind] == 1:
                    break
                # continue to next gt unless better match
                if ious_sorted[dind, gind] < iou:
                    continue
                iou = ious_sorted[dind, gind]
                m = gind
            if m == -1:
                continue
            dt_ig[dind] = float(gt_ig_sorted[m])
            dtm[dind] = m + 1  # 1-indexed, 0=unmatched
            gtm[m] = dind + 1

    # set unmatched detections outside area range to ignore
    dt_area_outside = np.array([
        (d['area'] < area_rng[0] or d['area'] > area_rng[1])
        for d in dt_sorted
    ], dtype=np.float64)
    dt_ig = np.logical_or(dt_ig, np.logical_and(dtm == 0, dt_area_outside)).astype(np.float64)

    # TP: matched (dtm>0) and not ignored
    dt_tp = np.logical_and(dtm > 0, np.logical_not(dt_ig)).astype(np.float64)
    # FP: not matched and not ignored
    dt_fp = np.logical_and(dtm == 0, np.logical_not(dt_ig)).astype(np.float64)

    n_gt_valid = int(np.count_nonzero(np.array(gt_ig_sorted) == 0))

    return dt_scores_sorted, dt_tp, dt_fp, n_gt_valid


def weighted_ap50(dets, gts, weights, iou_thr=0.5):
    """
    Compute weighted AP50 using pure NumPy.

    Args:
        dets:    {img_id: [{"score": float, "bbox": [x,y,w,h], "area": float}, ...]}
                 Already filtered to the relevant class+prompt.
        gts:     {img_id: [{"bbox": [x,y,w,h], "iscrowd": 0/1, "area": float}, ...]}
        weights: {img_id: int}  — how many times this image appears in the resample

    Returns:
        scalar AP50 (float). Returns NaN if n_gt_weighted == 0.
    """
    # Collect all unique image IDs that have GT or detections or weights>0
    all_img_ids = set(weights.keys())
    all_img_ids.update(dets.keys())
    all_img_ids.update(gts.keys())
    # But we only evaluate images that are in weights (the resampled set)
    eval_img_ids = [img_id for img_id in all_img_ids if weights.get(img_id, 0) > 0]

    # For each image: compute IoU matrix, then evaluate
    # Then collect all dets globally, sort by score, compute weighted TP/FP

    all_scores = []
    all_tp = []   # weighted TP
    all_fp = []   # weighted FP
    n_gt_weighted = 0

    for img_id in eval_img_ids:
        w = weights[img_id]
        img_dets = dets.get(img_id, [])
        img_gts = gts.get(img_id, [])

        if len(img_dets) == 0 and len(img_gts) == 0:
            # pycocotools returns None (skipped) for this image
            # But n_gt still counts if there are GTs... no, there are none.
            continue

        # Compute IoU matrix for this image
        det_bboxes = [d['bbox'] for d in img_dets]
        gt_bboxes = [g['bbox'] for g in img_gts]
        gt_iscrowd = [g['iscrowd'] for g in img_gts]

        if len(img_dets) > 0 and len(img_gts) > 0:
            ious = _iou_matrix(det_bboxes, gt_bboxes, gt_iscrowd)
        else:
            ious = np.zeros((len(img_dets), len(img_gts)))

        dt_scores, dt_tp, dt_fp, n_gt_valid = _evaluate_img(
            img_dets, gt_bboxes, gt_iscrowd, ious, iou_thr
        )

        # Weight each det's TP/FP by the image weight
        all_scores.extend(dt_scores)
        all_tp.extend(dt_tp * w)
        all_fp.extend(dt_fp * w)
        n_gt_weighted += w * n_gt_valid

    if n_gt_weighted == 0:
        return float('nan')

    if len(all_scores) == 0:
        return 0.0

    # Global sort by score descending (mergesort for stability)
    all_scores = np.array(all_scores)
    all_tp = np.array(all_tp)
    all_fp = np.array(all_fp)

    sort_idx = np.argsort(-all_scores, kind='mergesort')
    all_scores = all_scores[sort_idx]
    all_tp = all_tp[sort_idx]
    all_fp = all_fp[sort_idx]

    # Cumulative TP and FP
    tp_cum = np.cumsum(all_tp).astype(np.float64)
    fp_cum = np.cumsum(all_fp).astype(np.float64)

    # Recall and precision
    recall = tp_cum / n_gt_weighted
    precision = tp_cum / (tp_cum + fp_cum + np.spacing(1))

    # 101-point interpolation (COCO standard)
    rec_thrs = np.linspace(0.0, 1.0, 101)

    # Make precision monotonically decreasing (from right to left)
    pr = precision.tolist()
    for i in range(len(pr) - 2, -1, -1):
        if pr[i + 1] > pr[i]:
            pr[i] = pr[i + 1]
    pr = np.array(pr)

    # Searchsorted: for each recall threshold, find the index
    # pycocotools uses try/except: when index >= len(pr), q stays 0
    inds = np.searchsorted(recall, rec_thrs, side='left')

    q = np.zeros(101)
    for ri, pi in enumerate(inds):
        try:
            q[ri] = pr[pi]
        except IndexError:
            pass  # q[ri] stays 0, matching pycocotools behavior

    return float(q.mean())


def load_gt_for_class(coco_gt, cat_id, img_ids):
    """Load GT annotations for a class, organized by image_id.

    Returns:
        gts: {img_id: [{"bbox": [x,y,w,h], "iscrowd": 0/1, "area": float}, ...]}
    """
    gts = {}
    ann_ids = coco_gt.getAnnIds(imgIds=list(img_ids), catIds=[cat_id])
    anns = coco_gt.loadAnns(ann_ids)
    for ann in anns:
        img_id = ann['image_id']
        if img_id not in gts:
            gts[img_id] = []
        gts[img_id].append({
            'bbox': ann['bbox'],
            'iscrowd': int(ann.get('iscrowd', 0)),
            'area': ann['area'],
        })
    # Ensure all img_ids are present (empty list if no GT)
    for img_id in img_ids:
        if img_id not in gts:
            gts[img_id] = []
    return gts


def extract_dets(raw, cls, prompt_idx, img_ids):
    """Extract detections from raw detection JSON, formatted for weighted_ap50.

    Returns:
        dets: {img_id: [{"score": float, "bbox": [x,y,w,h], "area": float}, ...]}
    """
    cat_id_map = {"person": 1, "car": 3, "dog": 18, "cat": 17,
                  "chair": 62, "cup": 47, "bottle": 44, "book": 84}

    dets = {}
    pid = raw["per_image_detections"]
    key = f"{cls}__{prompt_idx}"
    for img_id in img_ids:
        s_id = str(img_id)
        if s_id in pid and key in pid[s_id]:
            img_dets = []
            for det in pid[s_id][key]:
                bbox = det["bbox"]  # [x, y, w, h]
                w, h = bbox[2], bbox[3]
                img_dets.append({
                    'score': det['score'],
                    'bbox': bbox,
                    'area': float(w * h),
                })
            dets[img_id] = img_dets
        else:
            dets[img_id] = []
    return dets
