#!/usr/bin/env python3
"""
M6: 41-class generalization experiment.

Extends the 8-class prompt sensitivity analysis to 41 classes.
3 prompts per class: bare, template ("a photo of a X"), length ("there is a X").
3 models: GLIP-T, Grounding DINO, OWLv2.
Same 500 images (seed=42) from m1_v2_raw_detections.json.
AP50 via weighted_ap50 (weights=1, equivalent to pycocotools COCOeval).

Class order: orig8 first (cross-validation), then 33 new classes.
Saves per (model, class) to results/m6_40class/<model>/<class>.json
Updates results/m6_40class/<model>_STATUS.json after each class.

Usage:
  python m6_40class.py --model glip --classes person    # smoke test
  python m6_40class.py --model glip                      # all 41 classes
  python m6_40class.py --model gdino --resume            # skip done
  python m6_40class.py --model owlv2                     # all 41
"""
import os, sys, json, time, argparse, random
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

sys.path.insert(0, os.path.dirname(__file__))
from weighted_ap import weighted_ap50, load_gt_for_class

# ── Constants ─────────────────────────────────────────────────────────────
COCO_IMG_DIR    = os.environ.get("COCO_IMG_DIR", "data/val2017")
COCO_ANN_FILE   = os.environ.get("COCO_ANN_FILE", "data/annotations/instances_val2017.json")
OUTPUT_DIR      = "results/m6_40class"
RAW_IMG_FILE    = "results/m1_v2_raw_detections.json"

SCORE_FILTER = 0.01
MAX_DETS     = 100

# ── Prompt generation ────────────────────────────────────────────────────
def make_prompts(class_name):
    """Generate 3 prompts for a class."""
    return {
        "bare":     class_name,
        "template": "a photo of a " + class_name,
        "there_is": "there is a " + class_name,
    }

# ── GLIP-T model ─────────────────────────────────────────────────────────
STANDALONE_DIR = os.environ.get("GLIP_ROOT", "third_party/glip")
GLIP_WEIGHTS    = os.environ.get("GLIP_WEIGHTS", "weights/glip_tiny_model_o365_goldg_cc_sbu.pth")
BERT_PATH = "bert-base-uncased"

PIXEL_MEAN = torch.tensor([103.530, 116.280, 123.675]).view(3, 1, 1)
PIXEL_STD  = torch.tensor([57.375, 57.120, 58.395]).view(3, 1, 1)

def _glip_norm(t):
    return (t[[2, 1, 0]] * 255.0 - PIXEL_MEAN) / PIXEL_STD

def _glip_preprocess(path, min_size=800, max_size=1333):
    img = Image.open(path).convert("RGB")
    ow, oh = img.size
    s = min_size / min(ow, oh)
    if max(ow, oh) * s > max_size:
        s = max_size / max(ow, oh)
    nw, nh = int(round(ow * s)), int(round(oh * s))
    img = img.resize((nw, nh), Image.BILINEAR)
    from torchvision.transforms import functional as TF
    return _glip_norm(TF.to_tensor(img)), (ow, oh), (nh, nw)

import torch.nn as nn

class _RPN(nn.Module):
    def __init__(self):
        super().__init__()
        from modules import VLDyHead, AnchorGenerator
        self.head = VLDyHead(num_convs=6, skip_stages={0})
        self.anchor_generator = AnchorGenerator()

class _VisTex(nn.Module):
    def __init__(self):
        super().__init__()
        from modules import Backbone, LanguageBackbone, ConvMapModule
        self.backbone = Backbone()
        self.language_backbone = LanguageBackbone()
        self.rpn = _RPN()
        self.fcto768 = ConvMapModule()

def _build_positive_map(tok):
    ids = tok["input_ids"][0]
    pmap, phrase, pos = {}, 0, []
    for idx in range(1, len(ids)):
        t = ids[idx].item()
        if t == 0:
            break
        if t in (1012, 102):
            if pos:
                phrase += 1
                pmap[phrase] = pos
                pos = []
        else:
            pos.append(idx)
        if phrase >= 79:
            break
    return pmap

def _glip_run_inference(model, tokenizer, img_t, orig_w, orig_h,
                        text_prompt, device="cuda"):
    from modules import to_image_list, ATSSPostProcessor
    batch, sizes = to_image_list([img_t], 32)
    batch = batch.to(device)
    resized_h, resized_w = sizes[0]

    caption = text_prompt.strip() + ". "
    tk = tokenizer(caption, max_length=256, padding="max_length",
                   return_tensors="pt", truncation=True)
    tk = {k: v.to(device) for k, v in tk.items()}
    pm = _build_positive_map(tk)

    with torch.no_grad():
        vfeat = model.backbone(batch)
        lang = model.language_backbone(tk)
        logits, bbox_reg, ctr, dot = model.rpn.head(vfeat, lang)
        anchors = model.rpn.anchor_generator(sizes, vfeat)
        results = ATSSPostProcessor()(bbox_reg, ctr, anchors, logits, dot, pm)

    pred = results[0]
    scale_x = orig_w / resized_w
    scale_y = orig_h / resized_h

    sc = pred.get_field("scores").cpu().numpy()
    boxes = pred.bbox.cpu().numpy()
    boxes[:, 0] *= scale_x
    boxes[:, 2] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 3] *= scale_y

    if len(sc) > MAX_DETS:
        order = np.argsort(-sc)
        sc, boxes = sc[order[:MAX_DETS]], boxes[order[:MAX_DETS]]

    dets = []
    for i in range(len(sc)):
        dets.append({
            "bbox": [float(boxes[i][0]), float(boxes[i][1]),
                     float(boxes[i][2] - boxes[i][0]),
                     float(boxes[i][3] - boxes[i][1])],
            "score": float(sc[i]),
        })
    return dets

def _load_glip_model(device="cuda"):
    sys.path.insert(0, STANDALONE_DIR)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    model = _VisTex()
    ck = torch.load(GLIP_WEIGHTS, map_location="cpu", weights_only=False)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in ck["model"].items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    miss_real = [m for m in miss if "relative_position_index" not in m]
    print(f"GLIP-T loaded: {len(sd)} keys | missing {len(miss_real)} unexpected {len(unexp)}")
    model = model.to(device).eval()
    return model, tokenizer

# ── Grounding DINO model ────────────────────────────────────────────────
GDINO_MODEL = "IDEA-Research/grounding-dino-tiny"

def _center_to_corners(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - w/2, cx + w/2, cy + h/2], dim=-1)

def _gdino_run_inference(model, processor, img_pil, prompt_text, device="cuda"):
    caption = prompt_text.strip().lower()
    if not caption.endswith("."):
        caption = caption + "."

    inputs = processor(text=caption, images=img_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    pred_boxes = outputs.pred_boxes[0]

    probs = torch.sigmoid(logits)
    scores = probs.max(dim=-1).values

    boxes_corners = _center_to_corners(pred_boxes)
    h, w = img_pil.height, img_pil.width
    scale_fct = torch.tensor([w, h, w, h], dtype=torch.float32, device=device)
    boxes_abs = boxes_corners * scale_fct

    scores_np = scores.cpu().numpy()
    boxes_np = boxes_abs.cpu().numpy()

    mask = scores_np > SCORE_FILTER
    if not mask.any():
        return []
    fb, fs = boxes_np[mask], scores_np[mask]
    if len(fs) > MAX_DETS:
        order = np.argsort(-fs)[:MAX_DETS]
        fb, fs = fb[order], fs[order]
    dets = []
    for i in range(len(fs)):
        x1, y1, x2, y2 = fb[i]
        dets.append({
            "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
            "score": float(fs[i]),
        })
    return dets

def _load_gdino_model(device="cuda"):
    os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    processor = AutoProcessor.from_pretrained(GDINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_MODEL, torch_dtype=torch.float32)
    model = model.to(device).eval()
    print(f"GDino loaded: {sum(p.numel() for p in model.parameters()):,} params")
    return model, processor

# ── OWLv2 model ──────────────────────────────────────────────────────────
OWLV2_MODEL = "google/owlv2-base-patch16-ensemble"

def _owlv2_run_inference_single(model, processor, img_pil, text, device="cuda"):
    """Run OWLv2 inference with a SINGLE text query.

    IMPORTANT: OWLv2 processes text queries jointly (text self-attention +
    cross-attention). Using a different number of text queries (e.g., 3 vs 80)
    changes the text embeddings and produces different scores/boxes, even though
    pred_boxes is query-independent in theory. To ensure reproducibility and
    consistency with the original v10 protocol (which used 80 queries but is
    not reproducible per-class), we run EACH prompt as an independent
    single-query forward pass. This eliminates text-batch-size interference.

    This matches how GLIP and GDino are run: each prompt is a separate forward.
    """
    inputs = processor(text=[text], images=img_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]       # [N, 1]
    pred_boxes = outputs.pred_boxes[0]  # [N, 4]
    boxes_corners = _center_to_corners(pred_boxes)
    h, w = img_pil.height, img_pil.width
    scale = float(max(h, w))
    boxes_abs = boxes_corners * scale
    boxes_np = boxes_abs.cpu().numpy()

    scores = torch.sigmoid(logits[:, 0]).cpu().numpy()  # [N]
    mask = scores > SCORE_FILTER
    if not mask.any():
        return []
    fb, fs = boxes_np[mask], scores[mask]
    if len(fs) > MAX_DETS:
        order = np.argsort(-fs)[:MAX_DETS]
        fb, fs = fb[order], fs[order]
    dets = []
    for i in range(len(fs)):
        x1, y1, x2, y2 = fb[i]
        dets.append({
            "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
            "score": float(fs[i]),
        })
    return dets

# ── OWLv2 cached inference (vision features computed once per image) ──
# OWLv2 is CLIP-style dual-encoder: vision_model and text_model are
# independent branches inside Owlv2Model.forward(). Only class_predictor
# mixes vision and text. box_predictor and objectness_predictor are
# vision-only. Therefore caching image_feats and reusing them for each
# text query is mathematically identical to a full forward per prompt,
# but runs the vision encoder only once per image instead of 3×.
# Expected speedup: ~3× (from ~0.17s/img×3 to ~0.17s/img×1 + tiny text).

def _owlv2_cache_image_features(model, processor, img_pil, device="cuda"):
    """Run vision encoder once, return (image_feats, feature_map) on device.

    image_feats:  [1, num_patches, hidden_dim]  (flattened)
    feature_map:   [1, num_patches_h, num_patches_w, hidden_dim]  (spatial)
    """
    pixel_values = processor(images=img_pil, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    with torch.no_grad():
        image_embeds, vision_outputs = model.image_embedder(
            pixel_values=pixel_values)
        # image_embeds shape: [1, H, W, D]
        batch_size, num_patches_h, num_patches_w, hidden_dim = image_embeds.shape
        image_feats = image_embeds.reshape(
            batch_size, num_patches_h * num_patches_w, hidden_dim)
    return image_feats, image_embeds  # image_embeds == feature_map

def _owlv2_run_inference_from_cache(model, processor, image_feats,
                                    feature_map, text, device="cuda"):
    """Run text encoder + class/box/objectness heads using cached image feats.

    This is numerically identical to model(image=..., text=[text]) because
    Owlv2Model.forward runs vision_model and text_model independently and
    only class_predictor combines them. We skip re-running vision_model.
    """
    # Encode the single text query
    text_inputs = processor(text=[text], return_tensors="pt")
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)

    with torch.no_grad():
        # Text branch (mirrors Owlv2Model.forward text path)
        text_outputs = model.owlv2.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True)
        text_embeds = text_outputs[1]               # pooled output
        text_embeds = model.owlv2.text_projection(text_embeds)
        text_embeds = text_embeds / torch.linalg.norm(
            text_embeds, dim=-1, keepdim=True)

        # Reshape for class_predictor: [batch=1, num_queries=1, hidden]
        batch_size = 1
        max_text_queries = input_ids.shape[0] // batch_size
        query_embeds = text_embeds.reshape(
            batch_size, max_text_queries, text_embeds.shape[-1])
        input_ids_r = input_ids.reshape(
            batch_size, max_text_queries, input_ids.shape[-1])
        query_mask = input_ids_r[..., 0] > 0       # [1, 1]

        # Heads (vision-only parts reuse cached image_feats)
        (pred_logits, class_embeds) = model.class_predictor(
            image_feats, query_embeds, query_mask)
        pred_boxes = model.box_predictor(image_feats, feature_map)
        # pred_logits: [1, num_patches, num_queries+1]
        # pred_boxes:  [1, num_patches, 4]

    logits = pred_logits[0]          # [num_patches, num_queries+1]
    boxes = pred_boxes[0]            # [num_patches, 4]
    boxes_corners = _center_to_corners(boxes)
    return logits, boxes_corners

def _owlv2_dets_from_logits(logits, boxes_corners, img_pil):
    """Convert cached-inference logits+boxes to detection list (matches
    _owlv2_run_inference_single output format)."""
    scores = torch.sigmoid(logits[:, 0]).cpu().numpy()
    boxes_np = boxes_corners.cpu().numpy()
    h, w = img_pil.height, img_pil.width
    scale = float(max(h, w))
    boxes_abs = boxes_np * scale
    mask = scores > SCORE_FILTER
    if not mask.any():
        return []
    fb, fs = boxes_abs[mask], scores[mask]
    if len(fs) > MAX_DETS:
        order = np.argsort(-fs)[:MAX_DETS]
        fb, fs = fb[order], fs[order]
    dets = []
    for i in range(len(fs)):
        x1, y1, x2, y2 = fb[i]
        dets.append({
            "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
            "score": float(fs[i]),
        })
    return dets

def _load_owlv2_model(device="cuda"):
    os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    processor = Owlv2Processor.from_pretrained(OWLV2_MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(
        OWLV2_MODEL, torch_dtype=torch.float32)
    model = model.to(device).eval()
    print(f"OWLv2 loaded: {sum(p.numel() for p in model.parameters()):,} params")
    return model, processor

# ── Common: detection → weighted_ap50 format ────────────────────────────
def dets_to_weighted_format(raw_dets, img_ids):
    """Convert list of {bbox, score} per image to weighted_ap50 format.
    
    raw_dets: {img_id: [{bbox, score}, ...]}
    Returns: {img_id: [{score, bbox, area}, ...]}
    """
    result = {}
    for img_id in img_ids:
        img_dets = raw_dets.get(img_id, [])
        formatted = []
        for d in img_dets:
            w, h = d["bbox"][2], d["bbox"][3]
            formatted.append({
                "score": d["score"],
                "bbox": d["bbox"],
                "area": float(w * h),
            })
        result[img_id] = formatted
    return result

def compute_ap50s(dets_per_prompt, gts, img_ids):
    """Compute AP50 for each prompt using weighted_ap50 with weights=1."""
    weights = {img_id: 1 for img_id in img_ids}
    ap50s = {}
    for prompt_name, dets in dets_per_prompt.items():
        formatted = dets_to_weighted_format(dets, img_ids)
        ap = weighted_ap50(formatted, gts, weights)
        ap50s[prompt_name] = ap
    return ap50s

# ── Status file management ──────────────────────────────────────────────
def load_status(model_name):
    status_file = os.path.join(OUTPUT_DIR, f"{model_name}_STATUS.json")
    if os.path.exists(status_file):
        with open(status_file) as f:
            return json.load(f)
    return {"model": model_name, "classes_done": [], "class_results": {}}

def save_status(status, model_name):
    status_file = os.path.join(OUTPUT_DIR, f"{model_name}_STATUS.json")
    status["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)

def save_class_result(model_name, class_name, result):
    model_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    out_file = os.path.join(model_dir, f"{class_name}.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="M6: 41-class generalization")
    parser.add_argument("--model", required=True, choices=["glip","gdino","owlv2"])
    parser.add_argument("--classes", nargs="*", help="specific classes to run")
    parser.add_argument("--resume", action="store_true", help="skip done classes")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-raw", action="store_true",
                        help="save per-image raw detections to results/m6_40class/<model>_raw/<class>.json")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = args.device

    # ── Load 500 image IDs from m1_v2_raw_detections ─────────────────────
    with open(RAW_IMG_FILE) as f:
        m1_raw = json.load(f)
    selected_img_ids = m1_raw["selected_img_ids"]
    print(f"Using {len(selected_img_ids)} images (seed=42)")

    # ── Load 41-class list ───────────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, "class_list_41.json")) as f:
        class_list = json.load(f)["classes"]

    # Order: orig8 first, then new classes by n_gt desc
    orig8_order = ["person","car","dog","cat","chair","cup","bottle","book"]
    new_classes = [c for c in class_list if c["name"] not in orig8_order]
    new_classes.sort(key=lambda c: -c["n_gt"])
    orig8_entries = [c for c in class_list if c["name"] in orig8_order]
    orig8_entries.sort(key=lambda c: orig8_order.index(c["name"]))
    all_classes = orig8_entries + new_classes

    # Filter to specific classes if requested
    if args.classes:
        all_classes = [c for c in all_classes if c["name"] in args.classes]

    print(f"Classes to run: {len(all_classes)}")
    for c in all_classes:
        print(f"  {c['name']:<15} cat_id={c['cat_id']:>3} n_gt={c['n_gt']:>5} {c['type']}")

    # ── Resume: skip done classes ────────────────────────────────────────
    status = load_status(args.model)
    if args.resume:
        done_set = set(status["classes_done"])
        before = len(all_classes)
        all_classes = [c for c in all_classes if c["name"] not in done_set]
        print(f"Resume: skipping {before - len(all_classes)} done classes, "
              f"{len(all_classes)} remaining")

    if not all_classes:
        print("Nothing to do. All classes completed.")
        return

    # ── Load COCO GT ──────────────────────────────────────────────────────
    print("Loading COCO val2017 annotations...", flush=True)
    coco_gt = COCO(COCO_ANN_FILE)
    img_infos = coco_gt.loadImgs(selected_img_ids)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\nLoading {args.model} model...", flush=True)
    t0 = time.time()
    if args.model == "glip":
        model, tokenizer = _load_glip_model(device)
    elif args.model == "gdino":
        model, processor = _load_gdino_model(device)
    elif args.model == "owlv2":
        model, processor = _load_owlv2_model(device)
    print(f"Model loaded in {time.time()-t0:.1f}s\n", flush=True)

    # ── Run per class ─────────────────────────────────────────────────────
    for ci, class_info in enumerate(all_classes):
        cls_name = class_info["name"]
        cat_id = class_info["cat_id"]
        n_gt_expected = class_info["n_gt"]
        prompts = make_prompts(cls_name)
        prompt_names = ["bare", "template", "there_is"]

        print(f"\n{'='*60}", flush=True)
        print(f"[{ci+1}/{len(all_classes)}] {cls_name} (cat_id={cat_id}, "
              f"n_gt={n_gt_expected})", flush=True)
        print(f"  bare:     {prompts['bare']}", flush=True)
        print(f"  template: {prompts['template']}", flush=True)
        print(f"  there_is: {prompts['there_is']}", flush=True)

        # ── Load GT for this class ────────────────────────────────────────
        gts = load_gt_for_class(coco_gt, cat_id, selected_img_ids)

        # ── Run inference ─────────────────────────────────────────────────
        # dets_per_prompt: {prompt_name: {img_id: [dets]}}
        dets_per_prompt = {pn: {} for pn in prompt_names}
        t0 = time.time()

        for ii, img_info in enumerate(img_infos):
            img_id = img_info["id"]
            img_path = os.path.join(COCO_IMG_DIR, img_info["file_name"])

            if args.model == "glip":
                img_t, (ow, oh), _ = _glip_preprocess(img_path)
                img_t = img_t.to(device)
                for pn in prompt_names:
                    dets = _glip_run_inference(
                        model, tokenizer, img_t, ow, oh,
                        prompts[pn], device)
                    dets_per_prompt[pn][img_id] = dets

            elif args.model == "gdino":
                img_pil = Image.open(img_path).convert("RGB")
                for pn in prompt_names:
                    dets = _gdino_run_inference(
                        model, processor, img_pil,
                        prompts[pn], device)
                    dets_per_prompt[pn][img_id] = dets

            elif args.model == "owlv2":
                img_pil = Image.open(img_path).convert("RGB")
                # Cache vision features once per image, reuse for all 3 prompts
                image_feats, feature_map = _owlv2_cache_image_features(
                    model, processor, img_pil, device)
                for pn in prompt_names:
                    logits, boxes_corners = _owlv2_run_inference_from_cache(
                        model, processor, image_feats, feature_map,
                        prompts[pn], device)
                    dets = _owlv2_dets_from_logits(logits, boxes_corners, img_pil)
                    dets_per_prompt[pn][img_id] = dets

            if (ii + 1) % 50 == 0 or (ii + 1) == len(img_infos):
                elapsed = time.time() - t0
                eta = elapsed / (ii + 1) * len(img_infos)
                print(f"  {ii+1}/{len(img_infos)} images | "
                      f"{elapsed:.1f}s | ETA {eta:.1f}s", flush=True)

        elapsed_inf = time.time() - t0

        # ── Compute AP50 ──────────────────────────────────────────────────
        print(f"  Computing AP50...", flush=True)
        ap50s = compute_ap50s(dets_per_prompt, gts, selected_img_ids)

        n_dets = {}
        for pn in prompt_names:
            total = sum(len(dets_per_prompt[pn].get(i, [])) for i in selected_img_ids)
            n_dets[pn] = total

        ap_bare    = ap50s["bare"]
        ap_tmpl    = ap50s["template"]
        ap_this    = ap50s["there_is"]
        delta_bare = ap_bare - ap_tmpl
        delta_len  = ap_this - ap_tmpl
        ratio      = delta_len / delta_bare if abs(delta_bare) > 1e-8 else float('nan')

        print(f"  AP50(bare)={ap_bare:.4f}  AP50(template)={ap_tmpl:.4f}  "
              f"AP50(there_is)={ap_this:.4f}", flush=True)
        print(f"  Δ_bare={delta_bare:+.4f}  Δ_len={delta_len:+.4f}  "
              f"ratio={ratio:.4f}", flush=True)
        print(f"  n_gt={n_gt_expected}  n_det(bare)={n_dets['bare']}  "
              f"n_det(tmpl)={n_dets['template']}  "
              f"n_det(this)={n_dets['there_is']}", flush=True)
        print(f"  Inference: {elapsed_inf:.1f}s "
              f"({elapsed_inf/len(img_infos):.3f}s/img)", flush=True)

        # ── Save per-class result ─────────────────────────────────────────
        result = {
            "model": args.model,
            "class": cls_name,
            "cat_id": cat_id,
            "type": class_info["type"],
            "n_gt": n_gt_expected,
            "num_images": len(selected_img_ids),
            "seed": 42,
            # Protocol metadata: OWLv2 uses single-query per prompt (1 text
            # query per forward) with vision feature caching. GLIP/GDino
            # inherently run single-prompt per forward (no query batching).
            "protocol": "single_query" if args.model == "owlv2" else "single_forward",
            "num_text_queries": 1 if args.model == "owlv2" else 1,
            "vision_feature_cache": args.model == "owlv2",
            "prompts": {
                pn: {
                    "text": prompts[pn],
                    "ap50": ap50s[pn],
                    "n_detections": n_dets[pn],
                }
                for pn in prompt_names
            },
            "ap_bare": ap_bare,
            "ap_template": ap_tmpl,
            "ap_there_is": ap_this,
            "delta_bare": delta_bare,
            "delta_len": delta_len,
            "ratio": ratio,
            "n_detections_80query_reference": 15470,
            "n_detections_single_query": n_dets["bare"] if args.model == "owlv2" else None,
            "inference_time_seconds": elapsed_inf,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        save_class_result(args.model, cls_name, result)

        # ── Update status ─────────────────────────────────────────────────
        if cls_name not in status["classes_done"]:
            status["classes_done"].append(cls_name)
        status["class_results"][cls_name] = {
            "ap_bare": ap_bare,
            "ap_template": ap_tmpl,
            "ap_there_is": ap_this,
            "delta_bare": delta_bare,
            "delta_len": delta_len,
            "ratio": ratio,
            "n_gt": n_gt_expected,
            "status": "done",
            "timestamp": result["timestamp"],
        }
        status["classes_total"] = len(class_list)
        status["classes_completed"] = len(status["classes_done"])
        save_status(status, args.model)

        print(f"  Saved: {OUTPUT_DIR}/{args.model}/{cls_name}.json", flush=True)

        # ── Save raw detections if requested ───────────────────────────────
        if args.save_raw:
            raw_dir = os.path.join(OUTPUT_DIR, f"{args.model}_raw")
            os.makedirs(raw_dir, exist_ok=True)
            raw_file = os.path.join(raw_dir, f"{cls_name}.json")
            raw_data = {
                "model": args.model,
                "class": cls_name,
                "cat_id": cat_id,
                "selected_img_ids": [int(i) for i in selected_img_ids],
                "prompt_bank": {
                    pn: prompts[pn] for pn in prompt_names
                },
                "num_images": len(selected_img_ids),
                "seed": 42,
                "per_image_detections": {
                    str(img_id): {
                        pn: dets_per_prompt[pn].get(img_id, [])
                        for pn in prompt_names
                    }
                    for img_id in selected_img_ids
                },
            }
            with open(raw_file, "w") as f:
                json.dump(raw_data, f)
            print(f"  Raw saved: {raw_file}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"DONE: {args.model} — {len(status['classes_done'])}/"
          f"{len(class_list)} classes completed", flush=True)
    print(f"{'='*60}", flush=True)

    # Free GPU
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
