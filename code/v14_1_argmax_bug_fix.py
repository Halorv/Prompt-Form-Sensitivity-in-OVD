#!/usr/bin/env python3
"""
v14.1: Print the FULL 8-position probs vector for query #0 to resolve
the question: does 0.5534 come from [CLS]/[SEP] or from token 'a' (pos 1)?

Also fix the dict-key-collision bug in all_token_scores.
"""
import os, sys, json, torch, numpy as np
from PIL import Image
from collections import Counter

os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_NAME = "IDEA-Research/grounding-dino-tiny"
COCO_IMG_DIR = os.environ.get("COCO_IMG_DIR", "data/val2017")
COCO_ANN_FILE = os.environ.get("COCO_ANN_FILE", "data/annotations/instances_val2017.json")
TARGET_IMG_ID = 267537
CAPTION = "a photo of a car."

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model = model.to(device).eval()
    print("Model loaded.\n")

    from pycocotools.coco import COCO
    coco = COCO(COCO_ANN_FILE)
    img_info = coco.loadImgs([TARGET_IMG_ID])[0]
    img_path = os.path.join(COCO_IMG_DIR, img_info["file_name"])
    img_pil = Image.open(img_path).convert("RGB")
    print(f"Image: id={TARGET_IMG_ID}, file={img_info['file_name']}, size={img_pil.size}\n")

    # Tokenize
    inputs = processor(text=CAPTION, images=img_pil, return_tensors="pt")
    inputs_dev = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs_dev["input_ids"]
    tokens = [processor.tokenizer.convert_ids_to_tokens(tid) for tid in input_ids[0].cpu().tolist()]

    print(f"Caption: '{CAPTION}'")
    print(f"input_ids: {input_ids[0].cpu().tolist()}")
    print(f"tokens:    {tokens}")
    print(f"  idx:     {'  '.join(str(i) for i in range(len(tokens)))}")
    print()

    # Forward
    with torch.no_grad():
        outputs = model(**inputs_dev)

    logits = outputs.logits[0]          # [num_queries, seq_len]
    probs = torch.sigmoid(logits)        # [num_queries, seq_len]
    per_query_max, per_query_argmax = probs.max(dim=-1)  # [num_queries]

    # Sort queries by score
    raw_scores = per_query_max.cpu().numpy()
    top5_idx = np.argsort(-raw_scores)[:5]

    print("=" * 90)
    print("FULL 8-POSITION PROBS VECTOR FOR TOP-5 QUERIES")
    print("=" * 90)

    for rank, qi in enumerate(top5_idx):
        qi = int(qi)
        tok_probs = probs[qi].cpu().numpy()

        print(f"\n  Query #{qi} (rank {rank+1}): max-over-tokens score = {raw_scores[qi]:.6f}")
        print(f"  {'pos':>4s} | {'token':>8s} | {'prob':>10s} | {'is_max?':>7s}")
        print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*10}-+-{'-'*7}")

        max_pos = int(np.argmax(tok_probs))
        for j in range(len(tokens)):
            is_max = "  <-- MAX" if j == max_pos else ""
            print(f"  {j:>4d} | {tokens[j]:>8s} | {tok_probs[j]:>10.6f} | {is_max}")

        # Check if max is [CLS] or [SEP]
        if tokens[max_pos] in ['[CLS]', '[SEP]']:
            print(f"  *** WARNING: max token is special token '{tokens[max_pos]}' at position {max_pos}!")
        else:
            print(f"  Max token = '{tokens[max_pos]}' at position {max_pos} (NOT a special token)")

    # Now show the BUG: dict key collision
    print("\n" + "=" * 90)
    print("BUG DEMONSTRATION: Dict key collision on duplicate tokens")
    print("=" * 90)

    qi = 0  # query #0
    tok_probs = probs[qi].cpu().numpy()

    print(f"\nQuery #{qi} probs vector:")
    for j in range(len(tokens)):
        print(f"  pos {j}: token='{tokens[j]}'  prob={tok_probs[j]:.6f}")

    print(f"\nDict comprehension (buggy code):")
    buggy_dict = {tokens[j]: float(tok_probs[j])
                  for j in range(len(tokens))
                  if tokens[j] not in ['[CLS]', '[SEP]', '[PAD]']
                  and tok_probs[j] > 0.01}
    print(f"  {buggy_dict}")
    print(f"  NOTE: 'a' appears twice in tokens (pos 1 and pos 4)")
    print(f"  Dict stores only the LAST value for 'a': {buggy_dict.get('a', 'N/A')}")
    print(f"  But actual value at pos 1 ('a'): {tok_probs[1]:.6f}")
    print(f"  And actual value at pos 4 ('a'): {tok_probs[4]:.6f}")

    print(f"\nFixed: use position-indexed keys")
    fixed_dict = {f"{tokens[j]}@{j}": float(tok_probs[j])
                  for j in range(len(tokens))
                  if tokens[j] not in ['[CLS]', '[SEP]', '[PAD]']
                  and tok_probs[j] > 0.01}
    print(f"  {fixed_dict}")

    print(f"\n{'='*90}")
    print("CONCLUSION")
    print(f"{'='*90}")
    max_pos = int(np.argmax(tok_probs))
    print(f"  Score 0.5534 for query #0 comes from position {max_pos} = '{tokens[max_pos]}'")
    print(f"  [CLS] (pos 0) prob = {tok_probs[0]:.6f}")
    print(f"  [SEP] (pos 7) prob = {tok_probs[7]:.6f}")
    if tokens[max_pos] not in ['[CLS]', '[SEP]']:
        print(f"  --> The max is a REGULAR token ('{tokens[max_pos]}'), NOT a special token.")
        print(f"  --> The contradiction in the report was caused by dict key collision:")
        print(f"      'a' at pos 1 (prob={tok_probs[1]:.4f}) was overwritten in the dict")
        print(f"      by 'a' at pos 4 (prob={tok_probs[4]:.4f}).")
    else:
        print(f"  --> The max IS a special token! HF's max-over-all-tokens includes [CLS]/[SEP].")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    main()
