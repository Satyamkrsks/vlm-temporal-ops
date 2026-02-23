"""
Phase 4: Evaluation Script
Computes OCA, tIoU@0.5, and AA@1 on 30 held-out clips from U0108.
"""

import os
import re
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm

from data_pipeline import (
    load_annotations, find_video, normalize_video, extract_clip,
    adaptive_sample, OPERATION_CLASSES, PROCEDURAL_GRAMMAR,
    FPS, CLIP_FRAMES, TARGET_SIZE, NUM_SAMPLE_FRAMES, BOUNDARY_MARGIN_F,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def tiou(p_start, p_end, g_start, g_end) -> float:
    """Temporal Intersection over Union."""
    inter = max(0, min(p_end, g_end) - max(p_start, g_start))
    union = max(0, max(p_end, g_end) - min(p_start, g_start))
    return inter / union if union > 0 else 0.0


def compute_metrics(preds: list[dict], gts: list[dict]) -> dict:
    """Compute OCA, tIoU@0.5, AA@1 over parallel prediction/GT lists."""
    assert len(preds) == len(gts)
    n = len(preds)

    oca_ok   = 0
    tiou_ok  = 0
    aa1_ok   = 0
    tiou_vals = []
    per_class = {op: {"tp": 0, "total": 0} for op in OPERATION_CLASSES}

    for pred, gt in zip(preds, gts):
        # OCA
        p_op = pred.get("dominant_operation", "Unknown")
        g_op = gt["operation"]
        per_class[g_op]["total"] += 1
        if p_op == g_op:
            oca_ok += 1
            per_class[g_op]["tp"] += 1

        # tIoU
        seg   = pred.get("temporal_segment", {})
        ps, pe = seg.get("start_frame", 0), seg.get("end_frame", 0)
        gs, ge = gt["local_start"], gt["local_end"]
        if ps < pe:
            t = tiou(ps, pe, gs, ge)
            tiou_vals.append(t)
            if t >= 0.5:
                tiou_ok += 1

        # AA@1
        p_next = pred.get("anticipated_next_operation", "Unknown")
        g_next = gt["next_operation"]
        if p_next == g_next:
            aa1_ok += 1

    return {
        "OCA":       round(oca_ok  / n, 4),
        "tIoU@0.5":  round(tiou_ok / n, 4),
        "AA@1":      round(aa1_ok  / n, 4),
        "mean_tIoU": round(float(np.mean(tiou_vals)) if tiou_vals else 0.0, 4),
        "n_clips":   n,
        "per_class_OCA": {
            op: round(d["tp"] / d["total"], 3) if d["total"] > 0 else None
            for op, d in per_class.items()
        },
    }


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model(model_name: str, lora_ckpt: Optional[str] = None):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    proc = AutoProcessor.from_pretrained(model_name)
    kwargs = dict(
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    if torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        )

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **kwargs)

    if lora_ckpt and Path(lora_ckpt).exists():
        from peft import PeftModel
        logger.info(f"Loading LoRA: {lora_ckpt}")
        model = PeftModel.from_pretrained(model, lora_ckpt)
        model = model.merge_and_unload()

    model.eval()
    return model, proc


EVAL_PROMPT = (
    "Analyze this warehouse packaging video clip.\n"
    "Available operations: " + ", ".join(OPERATION_CLASSES) + "\n\n"
    "Respond ONLY with JSON:\n"
    '{"dominant_operation":"<name>",'
    '"temporal_segment":{"start_frame":<int>,"end_frame":<int>},'
    '"anticipated_next_operation":"<name>","confidence":<float>}'
)


def infer_clip(model, proc, video: str, clip_start: int, clip_end: int) -> dict:
    """Run inference on one clip."""
    from qwen_vl_utils import process_vision_info
    from PIL import Image

    raw = extract_clip(Path(video), clip_start, clip_end)
    if raw is None or len(raw) < 2:
        return {"dominant_operation": "Unknown",
                "temporal_segment":   {"start_frame": 0, "end_frame": 0},
                "anticipated_next_operation": "Unknown", "confidence": 0.0}

    idx  = adaptive_sample(raw, n=NUM_SAMPLE_FRAMES)
    imgs = [Image.fromarray(raw[i]).resize(TARGET_SIZE) for i in idx]

    messages = [{"role": "user", "content": [
        *[{"type": "image", "image": im} for im in imgs],
        {"type": "text", "text": EVAL_PROMPT},
    ]}]

    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = proc(text=[text], images=img_inputs, videos=vid_inputs,
                  padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    new  = out[:, inputs["input_ids"].shape[1]:]
    resp = proc.batch_decode(new, skip_special_tokens=True)[0]
    return _parse(resp, len(raw))


def _parse(text: str, total: int) -> dict:
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            r = json.loads(m.group())
            if r.get("dominant_operation") not in OPERATION_CLASSES:
                r["dominant_operation"] = "Unknown"
            if r.get("anticipated_next_operation") not in OPERATION_CLASSES:
                r["anticipated_next_operation"] = PROCEDURAL_GRAMMAR.get(
                    r.get("dominant_operation", "Unknown"), "Unknown")
            seg = r.get("temporal_segment", {})
            r["temporal_segment"] = {
                "start_frame": int(max(0, seg.get("start_frame", 0))),
                "end_frame":   int(min(total-1, seg.get("end_frame", total-1))),
            }
            r["confidence"] = float(max(0.0, min(1.0, r.get("confidence", 0.5))))
            return r
    except Exception:
        pass
    return {"dominant_operation": "Unknown",
            "temporal_segment":   {"start_frame": 0, "end_frame": total-1},
            "anticipated_next_operation": "Unknown", "confidence": 0.1}


# ─── Test Clip Builder ────────────────────────────────────────────────────────

def build_test_clips(data_root: Path, subject: str = "U0108", n: int = 30) -> list[dict]:
    from data_pipeline import _find_sessions

    cache   = data_root / "_cache"
    sessions = _find_sessions(data_root, subject)
    clips   = []

    for sess in sessions:
        if len(clips) >= n:
            break

        src  = find_video(data_root, subject, sess)
        if src is None:
            continue
        norm = cache / subject / sess / "normalized.mp4"
        if not normalize_video(src, norm):
            continue

        anns = load_annotations(data_root, subject, sess)
        for i, ann in enumerate(anns):
            if len(clips) >= n:
                break
            op_start = ann["start_frame"]
            op_end   = ann["end_frame"]
            op       = ann["operation"]
            next_ann = anns[i + 1] if i + 1 < len(anns) else None
            next_op  = next_ann["operation"] if next_ann else PROCEDURAL_GRAMMAR.get(op, "Unknown")

            c_start = max(0, op_start - BOUNDARY_MARGIN_F)
            c_end   = c_start + CLIP_FRAMES
            loc_s   = max(0, op_start - c_start)
            loc_e   = min(CLIP_FRAMES - 1, op_end - c_start)

            clips.append({
                "clip_id":       f"{subject}_{sess}_op{i:04d}_{op.replace(' ','_')}",
                "video":         str(norm),
                "clip_start":    c_start,
                "clip_end":      c_end,
                "operation":     op,
                "next_operation": next_op,
                "local_start":   loc_s,
                "local_end":     loc_e,
            })

    clips = sorted(clips, key=lambda x: x["clip_id"])[:n]
    logger.info(f"Built {len(clips)} test clips from {subject}")
    return clips


# ─── Evaluation Runner ────────────────────────────────────────────────────────

def run_eval(model, proc, clips: list[dict], label: str):
    preds    = []
    per_clip = []

    for clip in tqdm(clips, desc=f"Eval [{label}]"):
        t0 = time.time()
        pred = infer_clip(model, proc, clip["video"], clip["clip_start"], clip["clip_end"])
        ms   = (time.time() - t0) * 1000

        t = tiou(pred["temporal_segment"]["start_frame"],
                 pred["temporal_segment"]["end_frame"],
                 clip["local_start"], clip["local_end"])

        preds.append(pred)
        per_clip.append({
            "clip_id":     clip["clip_id"],
            "gt_op":       clip["operation"],
            "pred_op":     pred["dominant_operation"],
            "gt_next":     clip["next_operation"],
            "pred_next":   pred.get("anticipated_next_operation"),
            "tiou":        round(t, 4),
            "oca_ok":      pred["dominant_operation"] == clip["operation"],
            "aa1_ok":      pred.get("anticipated_next_operation") == clip["next_operation"],
            "latency_ms":  round(ms, 1),
        })

    metrics = compute_metrics(preds, clips)
    return metrics, per_clip


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",            type=Path, default=Path("/data/openpack"))
    parser.add_argument("--base_model",           type=str,  default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--finetuned_checkpoint", type=str,  default=None)
    parser.add_argument("--n_clips",              type=int,  default=30)
    parser.add_argument("--subject",              type=str,  default="U0108")
    parser.add_argument("--output",               type=str,  default="results.json")
    args = parser.parse_args()

    clips = build_test_clips(args.data_root, args.subject, args.n_clips)
    if not clips:
        logger.error("No clips found. Check --data_root.")
        return

    results  = {}
    all_detail = {}

    # ── Base model ────────────────────────────────────────────────────────────
    logger.info("=== Base Model ===")
    bm, bp = load_model(args.base_model)
    bm_metrics, bm_detail = run_eval(bm, bp, clips, "base")
    results["base_model"] = {k: v for k, v in bm_metrics.items()}
    all_detail["base_model"] = bm_detail

    logger.info(f"OCA={bm_metrics['OCA']}  tIoU={bm_metrics['tIoU@0.5']}  AA@1={bm_metrics['AA@1']}")

    del bm, bp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Fine-tuned model ──────────────────────────────────────────────────────
    if args.finetuned_checkpoint and Path(args.finetuned_checkpoint).exists():
        logger.info("=== Fine-tuned Model ===")
        fm, fp = load_model(args.base_model, lora_ckpt=args.finetuned_checkpoint)
        fm_metrics, fm_detail = run_eval(fm, fp, clips, "finetuned")
        results["finetuned_model"] = {k: v for k, v in fm_metrics.items()}
        all_detail["finetuned_model"] = fm_detail

        logger.info(f"OCA={fm_metrics['OCA']}  tIoU={fm_metrics['tIoU@0.5']}  AA@1={fm_metrics['AA@1']}")

        d_oca  = fm_metrics["OCA"]      - bm_metrics["OCA"]
        d_tiou = fm_metrics["tIoU@0.5"] - bm_metrics["tIoU@0.5"]
        d_aa1  = fm_metrics["AA@1"]     - bm_metrics["AA@1"]

        results["deltas"] = {"ΔOCA": round(d_oca,4), "ΔtIoU@0.5": round(d_tiou,4), "ΔAA@1": round(d_aa1,4)}

        if d_aa1 <= 0:
            logger.warning("⚠  AA@1 did not improve — model failed temporal learning!")

        del fm, fp

    # Save
    Path(args.output).write_text(json.dumps(results, indent=2))
    logger.info(f"Saved → {args.output}")

    detail_path = Path(args.output).with_suffix(".details.json")
    detail_path.write_text(json.dumps(all_detail, indent=2))
    logger.info(f"Saved details → {detail_path}")


if __name__ == "__main__":
    main()