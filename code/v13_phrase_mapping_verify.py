#!/usr/bin/env python3
"""
v13 Step 2: Verify Grounding DINO phrase mapping.

Goal: determine whether GD's multi-word phrase failure is
  (a) model's real behavior, or
  (b) our implementation bug (phrase→token mapping error)

Method:
  1. Use OFFICIAL post_process_grounded_object_detection on 5 images
     with caption = "car . a photo of a car ."
     → two phrases sent simultaneously through official phrase splitting
  2. Extract per-phrase scores and boxes from official post-processing
  3. Also run our v11 bypass method on the same images for comparison
  4. Print input_ids, token→phrase mapping, and per-phrase detection quality

This script does NOT need GPU for AP50 computation — it just runs
5 images through the model and prints the raw outputs.
"""
import os, sys, json, time
import torch
import numpy as np
from PIL import Image

os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

COCO_IMG_DIR = os.environ.get("COCO_IMG_DIR", "data/val2017")
COCO_ANN_FILE = os.environ.get("COCO_ANN_FILE", "data/annotations/instances_val2017.json")
MODEL_NAME = "IDEA-Research/grounding-dino-tiny"

# 5 test images — pick from our v11 selected set
with open("results/gdino_v11_raw_detections.json") as f:
    gdino_raw = json.load(f)
selected_img_ids = gdino_raw["selected_img_ids"][:5]

# Also load COCO for GT
from pycocotools.coco import COCO
coco_gt = COCO(COCO_ANN_FILE)


def center_to_corners(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def run_official_multi_phrase(model, processor, img_pil, caption, device="cuda"):
    """Use official post_process_grounded_object_detection with multi-phrase caption."""
    inputs = processor(text=caption, images=img_pil, return_tensors="pt")
    inputs_dev = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs_dev)

    # Official post-processing
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs_dev["input_ids"],
        box_threshold=0.01,
        text_threshold=0.01,
        target_sizes=torch.tensor([[img_pil.height, img_pil.width]]).to(device),
    )
    return results[0], inputs, outputs


def run_bypass_single(model, processor, img_pil, prompt_text, device="cuda"):
    """Our v11 bypass method: max over ALL tokens, single prompt."""
    caption = prompt_text.strip().lower()
    if not caption.endswith("."):
        caption = caption + "."

    inputs = processor(text=caption, images=img_pil, return_tensors="pt")
    inputs_dev = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs_dev)

    logits = outputs.logits[0]
    pred_boxes = outputs.pred_boxes[0]

    probs = torch.sigmoid(logits)
    scores = probs.max(dim=-1).values

    boxes_corners = center_to_corners(pred_boxes)
    h, w = img_pil.height, img_pil.width
    scale_fct = torch.tensor([w, h, w, h], dtype=torch.float32, device=device)
    boxes_abs = boxes_corners * scale_fct

    scores_np = scores.cpu().numpy()
    boxes_np = boxes_abs.cpu().numpy()

    # Also print per-token scores for top queries
    probs_np = probs.cpu().numpy()

    return scores_np, boxes_np, probs_np, inputs


def decode_input_ids(processor, input_ids):
    """Decode input_ids to tokens for inspection."""
    tokens = []
    for tid in input_ids[0].cpu().tolist():
        token = processor.tokenizer.convert_ids_to_tokens(tid)
        tokens.append(token)
    return tokens


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    print("Loading model...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    )
    model = model.to(device).eval()
    print(f"Model loaded in {t0-time.time():.1f}s\n")

    # Test captions
    test_cases = [
        ("car", "car ."),
        ("template", "a photo of a car ."),
        ("multi_phrase", "car . a photo of a car ."),
    ]

    all_results = {}

    for img_idx, img_id in enumerate(selected_img_ids):
        img_info = coco_gt.loadImgs([img_id])[0]
        img_path = os.path.join(COCO_IMG_DIR, img_info["file_name"])
        img_pil = Image.open(img_path).convert("RGB")
        print(f"\n{'='*80}")
        print(f"Image {img_idx+1}/5: id={img_id}, file={img_info['file_name']}, "
              f"size={img_pil.size}")
        print(f"{'='*80}")

        img_results = {}

        for case_name, caption in test_cases:
            print(f"\n--- Case: {case_name} | caption='{caption}' ---")

            # Decode input_ids
            inputs = processor(text=caption, images=img_pil, return_tensors="pt")
            input_ids = inputs["input_ids"]
            tokens = decode_input_ids(processor, input_ids)
            print(f"  input_ids: {input_ids[0].cpu().tolist()}")
            print(f"  tokens:    {tokens}")

            if case_name == "multi_phrase":
                # Official multi-phrase path
                result_official, inputs_full, outputs = run_official_multi_phrase(
                    model, processor, img_pil, caption, device
                )

                # Print what phrases were detected
                phrases_official = result_official.get("labels", [])
                scores_official = result_official.get("scores", [])
                boxes_official = result_official.get("boxes", [])

                print(f"\n  OFFICIAL post_process_grounded_object_detection:")
                print(f"    phrases field: {result_official.get('labels', [])[:20]}")
                print(f"    n_detections: {len(scores_official)}")

                # Count detections per phrase
                from collections import Counter
                phrase_counts = Counter(phrases_official)
                print(f"    phrase distribution: {dict(phrase_counts)}")

                # Top-10 detections per phrase
                for phrase_text in ["car", "a photo of a car"]:
                    mask = [p == phrase_text for p in phrases_official]
                    phrase_scores = scores_official[mask].cpu().numpy() if isinstance(scores_official, torch.Tensor) else np.array(scores_official)[mask]
                    if len(phrase_scores) > 0:
                        print(f"\n    Phrase='{phrase_text}': {len(phrase_scores)} detections")
                        print(f"      top-5 scores: {sorted(phrase_scores, reverse=True)[:5]}")
                        print(f"      score stats: min={phrase_scores.min():.4f} "
                              f"max={phrase_scores.max():.4f} "
                              f"mean={phrase_scores.mean():.4f}")
                    else:
                        print(f"\n    Phrase='{phrase_text}': 0 detections")

                # Also get per-token scores from the raw logits
                logits = outputs.logits[0]  # [num_queries, seq_len]
                probs = torch.sigmoid(logits).cpu().numpy()
                print(f"\n  RAW LOGITS ANALYSIS (multi-phrase):")
                print(f"    logits shape: {logits.shape}")
                # For each token position, print the max query score
                for tok_idx, tok in enumerate(tokens):
                    if tok in ['[CLS]', '[SEP]', '[PAD]']:
                        continue
                    col = probs[:, tok_idx]
                    print(f"    token[{tok_idx}]='{tok}': "
                          f"max={col.max():.4f} "
                          f"mean={col.mean():.4f} "
                          f"#queries>{0.5}={int((col > 0.5).sum())}")

                # Print the phrase→token mapping from the processor
                # The processor's post_process_grounded_object_detection
                # uses tokenized_phrase = tokenized_caption.split by '.'
                # Let's check what phrases the processor sees
                print(f"\n  PHRASE SPLITTING:")
                phrases_from_caption = [p.strip() for p in caption.split('.') if p.strip()]
                print(f"    caption: '{caption}'")
                print(f"    split by '.': {phrases_from_caption}")

                # Map phrases to token positions
                token_list = tokens
                current_pos = 0
                for phrase in phrases_from_caption:
                    phrase_tokens = processor.tokenizer.encode(
                        phrase, add_special_tokens=False
                    )
                    decoded = [processor.tokenizer.convert_ids_to_tokens(t)
                               for t in phrase_tokens]
                    # Find this token sequence in the full token list
                    # (accounting for [CLS] prefix)
                    search_start = 1  # skip [CLS]
                    found = False
                    for start in range(search_start, len(token_list) - len(phrase_tokens) + 1):
                        match = all(
                            processor.tokenizer.convert_tokens_to_ids(token_list[start + j]) == phrase_tokens[j]
                            for j in range(len(phrase_tokens))
                        )
                        if match:
                            print(f"    phrase='{phrase}' → tokens[{start}:{start+len(phrase_tokens)}] = "
                                  f"{token_list[start:start+len(phrase_tokens)]}")
                            found = True
                            break
                    if not found:
                        print(f"    phrase='{phrase}' → NOT FOUND in token list!")

                img_results["multi_phrase"] = {
                    "caption": caption,
                    "input_ids": input_ids[0].cpu().tolist(),
                    "tokens": tokens,
                    "phrases_split": phrases_from_caption,
                    "official_detections": {
                        "n_total": len(scores_official),
                        "phrase_counts": dict(phrase_counts),
                    }
                }

            elif case_name == "car":
                # Single phrase: "car."
                scores_np, boxes_np, probs_np, inputs_single = run_bypass_single(
                    model, processor, img_pil, "car", device
                )
                print(f"\n  BYPASS (v11 method) for 'car.':")
                print(f"    logits shape: {probs_np.shape}")
                mask = scores_np > 0.01
                print(f"    detections above 0.01: {mask.sum()}")
                if mask.sum() > 0:
                    top_idx = np.argsort(-scores_np[mask])[:5]
                    masked_scores = scores_np[mask]
                    print(f"    top-5 scores: {sorted(masked_scores, reverse=True)[:5]}")

                # Per-token analysis
                print(f"\n  PER-TOKEN SCORES (bypass, 'car.'):")
                for tok_idx, tok in enumerate(tokens):
                    if tok in ['[CLS]', '[SEP]', '[PAD]']:
                        continue
                    col = probs_np[:, tok_idx]
                    print(f"    token[{tok_idx}]='{tok}': "
                          f"max={col.max():.4f} mean={col.mean():.4f} "
                          f"#queries>{0.5}={int((col > 0.5).sum())}")

            elif case_name == "template":
                # Single phrase: "a photo of a car."
                scores_np, boxes_np, probs_np, inputs_single = run_bypass_single(
                    model, processor, img_pil, "a photo of a car", device
                )
                print(f"\n  BYPASS (v11 method) for 'a photo of a car.':")
                print(f"    logits shape: {probs_np.shape}")
                mask = scores_np > 0.01
                print(f"    detections above 0.01: {mask.sum()}")
                if mask.sum() > 0:
                    top_idx = np.argsort(-scores_np[mask])[:5]
                    masked_scores = scores_np[mask]
                    print(f"    top-5 scores: {sorted(masked_scores, reverse=True)[:5]}")

                # Per-token analysis
                print(f"\n  PER-TOKEN SCORES (bypass, 'a photo of a car.'):")
                for tok_idx, tok in enumerate(tokens):
                    if tok in ['[CLS]', '[SEP]', '[PAD]']:
                        continue
                    col = probs_np[:, tok_idx]
                    print(f"    token[{tok_idx}]='{tok}': "
                          f"max={col.max():.4f} mean={col.mean():.4f} "
                          f"#queries>{0.5}={int((col > 0.5).sum())}")

        all_results[str(img_id)] = img_results

    # ── Summary comparison ──────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("SUMMARY: Official multi-phrase vs bypass single-phrase")
    print(f"{'='*80}")
    print()
    print("For each image, comparing 'car' phrase detections from:")
    print("  (A) Official multi-phrase: caption='car . a photo of a car .'")
    print("  (B) Bypass single-phrase: caption='car.'")
    print("  (C) Bypass single-phrase: caption='a photo of a car.'")
    print()
    print("If (A) gives similar quality for 'car' and 'a photo of a car' phrases,")
    print("  and (B)/(C) show 'car' >> 'a photo of a car', → our bypass has a bug")
    print("If (A) also shows 'car' >> 'a photo of a car', → model's real behavior")

    # Save results
    # Strip non-serializable
    serializable = {}
    for img_id, data in all_results.items():
        serializable[img_id] = {}
        for case, case_data in data.items():
            if isinstance(case_data, dict):
                serializable[img_id][case] = {
                    k: v for k, v in case_data.items()
                    if isinstance(v, (str, int, float, list, dict, bool))
                }

    with open("results/v13_phrase_mapping_verification.json", "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to results/v13_phrase_mapping_verification.json")

    # Free GPU
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
