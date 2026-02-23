"""
Phase 2: OpenPack Data Pipeline
Motion-magnitude adaptive frame sampling for temporal boundary clips.

Usage:
  python data_pipeline.py --data_root /data/openpack --output_dir ./training_data_samples
  python data_pipeline.py --data_root /data/openpack --output_dir ./shards --mode shards
"""

import os
import io
import json
import math
import hashlib
import tarfile
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Iterator

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

OPERATION_CLASSES = [
    "Box Setup", "Inner Packing", "Tape", "Put Items",
    "Pack", "Wrap", "Label", "Final Check", "Idle", "Unknown"
]

OP_ID_TO_NAME = {
    100: "Box Setup",
    200: "Inner Packing",
    300: "Tape",
    400: "Put Items",
    500: "Pack",
    600: "Wrap",
    700: "Label",
    800: "Final Check",
    900: "Idle",
    0:   "Unknown",
}

OP_NAME_TO_ID = {v: k for k, v in OP_ID_TO_NAME.items()}

PROCEDURAL_GRAMMAR = {
    "Box Setup":     "Inner Packing",
    "Inner Packing": "Put Items",
    "Put Items":     "Pack",
    "Pack":          "Tape",
    "Tape":          "Label",
    "Label":         "Final Check",
    "Final Check":   "Idle",
    "Wrap":          "Label",
    "Idle":          "Box Setup",
    "Unknown":       "Unknown",
}

TRAIN_SUBJECTS = ["U0101", "U0102", "U0103", "U0104", "U0105", "U0106"]
VAL_SUBJECTS   = ["U0107"]
TEST_SUBJECTS  = ["U0108"]

FPS               = 25           # Kinect RGB framerate
CLIP_DURATION_S   = 5.0          # Seconds per clip
CLIP_FRAMES       = int(CLIP_DURATION_S * FPS)   # = 125
NUM_SAMPLE_FRAMES = 8            # Frames fed to VLM
TARGET_SIZE       = (336, 336)   # Qwen2.5-VL native resolution
BOUNDARY_MARGIN_S = 0.5          # Overlap around boundaries (seconds)
BOUNDARY_MARGIN_F = int(BOUNDARY_MARGIN_S * FPS)  # = 12 frames


# ─── Annotation Loading ───────────────────────────────────────────────────────

def load_annotations(data_root: Path, subject: str, session: str) -> list[dict]:
    """
    Load frame-level operation annotations for one subject/session.
    Tries: JSON file → openpack-toolkit → synthetic fallback
    """
    ann_path = data_root / subject / session / "annotation" / "openpack-operations.json"

    if ann_path.exists():
        return _load_from_json(ann_path)

    try:
        import openpack_toolkit as opt
        return _load_from_toolkit(data_root, subject, session)
    except Exception:
        pass

    logger.warning(f"No annotation found for {subject}/{session} — using synthetic")
    return _synthetic_annotations(subject, session)


def _load_from_json(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)

    result = []
    for entry in data.get("annotations", []):
        op_id    = entry.get("id", 0)
        start_ms = entry.get("start_time", 0)
        end_ms   = entry.get("end_time",   0)
        result.append({
            "operation":   OP_ID_TO_NAME.get(op_id, "Unknown"),
            "start_frame": int(start_ms * FPS / 1000),
            "end_frame":   int(end_ms   * FPS / 1000),
        })
    return result


def _load_from_toolkit(data_root: Path, subject: str, session: str) -> list[dict]:
    import openpack_toolkit as opt
    cfg   = opt.configs.load(data_root / "config.yaml")
    annot = opt.load_annotation(cfg, subject, session)
    result = []
    for row in annot.itertuples():
        op_id = getattr(row, "operation_id", 0)
        result.append({
            "operation":   OP_ID_TO_NAME.get(op_id, "Unknown"),
            "start_frame": int(getattr(row, "start_frame", 0)),
            "end_frame":   int(getattr(row, "end_frame",   0)),
        })
    return result


def _synthetic_annotations(subject: str, session: str) -> list[dict]:
    """Deterministic synthetic annotations for testing pipeline without real data."""
    seed = int(hashlib.md5(f"{subject}{session}".encode()).hexdigest()[:8], 16)
    rng  = np.random.default_rng(seed)

    sequence = [
        "Box Setup", "Inner Packing", "Put Items", "Pack",
        "Tape", "Label", "Final Check", "Idle"
    ]
    mean_dur = {
        "Box Setup": 8.0, "Inner Packing": 12.0, "Put Items": 20.0,
        "Pack": 15.0, "Tape": 10.0, "Label": 5.0,
        "Final Check": 6.0, "Idle": 4.0,
    }

    anns  = []
    frame = 0
    for _ in range(3):                       # 3 packaging cycles
        for op in sequence:
            dur_s = rng.exponential(mean_dur[op])
            dur_f = int(dur_s * FPS)
            anns.append({
                "operation":   op,
                "start_frame": frame,
                "end_frame":   frame + dur_f,
            })
            frame += dur_f
    return anns


# ─── Video Utilities ──────────────────────────────────────────────────────────

def find_video(data_root: Path, subject: str, session: str) -> Optional[Path]:
    """Search standard OpenPack directory structure for Kinect RGB video."""
    candidates = [
        data_root / subject / session / "kinect" / "color" / "video.mp4",
        data_root / subject / session / "kinect" / "color.mp4",
        data_root / subject / session / "video.mp4",
        data_root / subject / session / "color.avi",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Glob fallback
    hits = (list((data_root / subject / session).glob("**/*.mp4")) +
            list((data_root / subject / session).glob("**/*.avi")))
    return hits[0] if hits else None


def normalize_video(src: Path, dst: Path) -> bool:
    """Normalize to 25fps, 336×336 H.264 using ffmpeg. Skips if dst exists."""
    if dst.exists():
        return True

    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale={TARGET_SIZE[0]}:{TARGET_SIZE[1]},fps={FPS}",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-an",
        str(dst)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error(f"ffmpeg error: {r.stderr[:300]}")
        return False
    return True


def extract_clip(video: Path, start: int, end: int) -> Optional[np.ndarray]:
    """Extract frames [start, end] as numpy array (N, H, W, 3)."""
    try:
        from decord import VideoReader, cpu
        vr    = VideoReader(str(video), ctx=cpu(0))
        total = len(vr)
        s     = max(0, min(start, total - 1))
        e     = max(s,  min(end,   total - 1))
        idx   = list(range(s, e + 1))
        return vr.get_batch(idx).asnumpy() if idx else None
    except Exception:
        return _extract_cv2(video, start, end)


def _extract_cv2(video: Path, start: int, end: int) -> Optional[np.ndarray]:
    import cv2
    cap    = cv2.VideoCapture(str(video))
    frames = []
    for i in range(start, end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames) if frames else None


# ─── Motion-Magnitude Adaptive Sampling ──────────────────────────────────────
#
# Diagram:
#   Op:     [──── Pack ───────────────][──── Tape ──────]
#   Motion:  ░░░░░░░░░░░░░░▓▓▓░░░░░░▓▓▓▓▓░░░░░░░░░░
#                               ▲ boundary (high motion)
#   Uniform: ×   ×   ×   ×   ×   ×   ×   ×  (misses spike)
#   Adaptive:×       ×       × × ×   ×   ×  (captures spike)

def adaptive_sample(
    frames: np.ndarray,
    n:      int = NUM_SAMPLE_FRAMES,
    boundary: Optional[int] = None,
) -> list[int]:
    """
    Select n frame indices from clip using motion-magnitude priority.

    Always includes frames 0 and T-1.
    Optionally weights toward a known boundary frame.
    """
    T = len(frames)
    if T <= n:
        return list(range(T))

    # Grayscale inter-frame difference as motion proxy
    gray    = frames.mean(axis=-1).astype(np.float32)   # (T, H, W)
    scores  = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        scores[t] = np.abs(gray[t] - gray[t - 1]).mean()

    # Optional boundary Gaussian boost
    if boundary is not None:
        sigma  = T * 0.1
        t_vals = np.arange(T, dtype=np.float32)
        boost  = np.exp(-0.5 * ((t_vals - boundary) / sigma) ** 2)
        scores = scores + 0.5 * scores.max() * boost

    mandatory = {0, T - 1}
    if boundary is not None:
        mandatory.add(max(0, min(T - 1, boundary)))

    # Zero out mandatory so they don't compete
    tmp = scores.copy()
    for i in mandatory:
        tmp[i] = -1

    extra   = max(0, n - len(mandatory))
    top_idx = np.argsort(tmp)[::-1][:extra]
    result  = sorted(mandatory | set(top_idx.tolist()))

    return result[:n]


# ─── Training Pair Builder ────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are analyzing a warehouse packaging operation video clip. "
    "Identify the current operation, its temporal boundaries, and predict the next operation "
    "based on procedural workflow knowledge.\n\n"
    f"Available classes: {', '.join(OPERATION_CLASSES)}\n\n"
    "Respond ONLY with JSON:\n"
    "{\n"
    '  "dominant_operation": "<name>",\n'
    '  "temporal_segment": {"start_frame": <int>, "end_frame": <int>},\n'
    '  "anticipated_next_operation": "<name>",\n'
    '  "confidence": <float>\n'
    "}"
)


def make_pair(
    subject:   str,
    session:   str,
    ann:       dict,
    next_ann:  Optional[dict],
    indices:   list[int],
    images:    list[Image.Image],
    loc_start: int,
    loc_end:   int,
) -> dict:
    op      = ann["operation"]
    next_op = next_ann["operation"] if next_ann else PROCEDURAL_GRAMMAR.get(op, "Unknown")
    clip_id = f"{subject}_{session}_op{OP_NAME_TO_ID.get(op,0):04d}"

    return {
        "clip_id":               clip_id,
        "subject":               subject,
        "session":               session,
        "operation":             op,
        "next_operation":        next_op,
        "sampled_frame_indices": indices,
        "system_prompt":         SYSTEM_PROMPT,
        "target_json": {
            "dominant_operation":        op,
            "temporal_segment":          {"start_frame": loc_start, "end_frame": loc_end},
            "anticipated_next_operation": next_op,
            "confidence":                1.0,
        },
        "frames": images,
    }


# ─── Subject Iterator ─────────────────────────────────────────────────────────

def iter_subject(data_root: Path, subject: str, frame_cache: Path) -> Iterator[dict]:
    """Yield training pairs for all sessions of a subject."""
    sessions = _find_sessions(data_root, subject)
    logger.info(f"{subject}: {len(sessions)} sessions found")

    for session in sessions:
        src_video = find_video(data_root, subject, session)
        if src_video is None:
            logger.warning(f"No video: {subject}/{session}")
            continue

        norm_path = frame_cache / subject / session / "normalized.mp4"
        if not normalize_video(src_video, norm_path):
            continue

        anns = load_annotations(data_root, subject, session)
        if not anns:
            continue

        for i, ann in enumerate(anns):
            op    = ann["operation"]
            next_ = anns[i + 1] if i + 1 < len(anns) else None

            # Undersample boring classes
            if op in ("Unknown", "Idle") and np.random.random() > 0.2:
                continue

            op_start   = ann["start_frame"]
            op_end     = ann["end_frame"]
            clip_start = max(0, op_start - BOUNDARY_MARGIN_F)
            clip_end   = clip_start + CLIP_FRAMES
            loc_start  = max(0, op_start - clip_start)
            loc_end    = min(CLIP_FRAMES - 1, op_end - clip_start)

            raw = extract_clip(norm_path, clip_start, clip_end)
            if raw is None or len(raw) < 4:
                continue

            idx  = adaptive_sample(raw, n=NUM_SAMPLE_FRAMES, boundary=loc_start)
            imgs = [Image.fromarray(raw[j]).resize(TARGET_SIZE) for j in idx]

            yield make_pair(subject, session, ann, next_, idx, imgs, loc_start, loc_end)


def _find_sessions(data_root: Path, subject: str) -> list[str]:
    d = data_root / subject
    if d.exists():
        return sorted(p.name for p in d.iterdir() if p.is_dir())
    return ["S0500"]


# ─── Output Writers ───────────────────────────────────────────────────────────

def save_samples(pairs: list[dict], out_dir: Path):
    """Save up to 20 examples for reviewer verification."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, pair in enumerate(pairs[:20]):
        op_slug = pair["operation"].replace(" ", "_")
        d = out_dir / f"example_{i:03d}_{op_slug}"
        d.mkdir(exist_ok=True)

        for j, img in enumerate(pair["frames"]):
            img.save(d / f"frame_{j:02d}.jpg", quality=90)

        meta = {k: v for k, v in pair.items() if k != "frames"}
        (d / "metadata.json").write_text(json.dumps(meta, indent=2))
        (d / "target_response.json").write_text(json.dumps(pair["target_json"], indent=2))
        (d / "system_prompt.txt").write_text(pair["system_prompt"])

    logger.info(f"Saved {min(20, len(pairs))} sample examples → {out_dir}")


def save_webdataset(pairs_iter: Iterator[dict], out_dir: Path, split: str = "train"):
    """Write pairs to WebDataset .tar shards (200 MB each)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    MAX_SHARD = 200 * 1024 * 1024   # 200 MB

    shard_idx  = 0
    shard_size = 0
    n_total    = 0

    def open_shard():
        p = out_dir / f"{split}-{shard_idx:05d}.tar"
        return tarfile.open(p, "w"), p

    tar, cur_path = open_shard()

    for pair in pairs_iter:
        cid = pair["clip_id"]

        # Inner tar of JPEG frames
        fb = io.BytesIO()
        with tarfile.open(fileobj=fb, mode="w") as ft:
            for j, img in enumerate(pair["frames"]):
                ib = io.BytesIO()
                img.save(ib, format="JPEG", quality=90)
                raw = ib.getvalue()
                ti  = tarfile.TarInfo(name=f"frame_{j:02d}.jpg")
                ti.size = len(raw)
                ft.addfile(ti, io.BytesIO(raw))

        meta    = {k: v for k, v in pair.items() if k != "frames"}
        mb      = json.dumps(meta, indent=2).encode()
        pb      = pair["system_prompt"].encode()
        fb_val  = fb.getvalue()

        for name, data in [(f"{cid}.frames.tar", fb_val),
                           (f"{cid}.json",        mb),
                           (f"{cid}.txt",         pb)]:
            ti      = tarfile.TarInfo(name=name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
            shard_size += len(data)

        n_total += 1

        if shard_size >= MAX_SHARD:
            tar.close()
            logger.info(f"Shard {cur_path.name}: {shard_size/1e6:.0f} MB, {n_total} samples")
            shard_idx  += 1
            shard_size  = 0
            tar, cur_path = open_shard()

    tar.close()
    logger.info(f"Done. {n_total} pairs written to {out_dir}")
    return n_total


def build_hf_dataset(data_root: Path, subjects: list[str], frame_cache: Path):
    """Build HuggingFace Dataset for SFTTrainer."""
    from datasets import Dataset

    records = []
    for subject in subjects:
        for pair in iter_subject(data_root, subject, frame_cache):
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": pair["system_prompt"]}],
                },
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image"} for _ in pair["frames"]],
                        {"type": "text", "text": "Analyze this warehouse operation video clip."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": json.dumps(pair["target_json"])}],
                },
            ]
            records.append({
                "clip_id":       pair["clip_id"],
                "messages":      messages,
                "images":        pair["frames"],
                "operation":     pair["operation"],
                "next_operation": pair["next_operation"],
            })

    return Dataset.from_list(records)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OpenPack data pipeline")
    parser.add_argument("--data_root",   type=Path, default=Path("/data/openpack"))
    parser.add_argument("--output_dir",  type=Path, default=Path("./training_data_samples"))
    parser.add_argument("--frame_cache", type=Path, default=Path("./frame_cache"))
    parser.add_argument("--mode",        choices=["samples", "shards", "hf_dataset"], default="samples")
    parser.add_argument("--split",       choices=["train", "val", "test"], default="train")
    parser.add_argument("--max_clips",   type=int, default=None)
    args = parser.parse_args()

    subjects = {"train": TRAIN_SUBJECTS, "val": VAL_SUBJECTS, "test": TEST_SUBJECTS}[args.split]

    if args.mode == "samples":
        all_pairs = []
        for subj in subjects:
            for pair in tqdm(iter_subject(args.data_root, subj, args.frame_cache), desc=subj):
                all_pairs.append(pair)
                if args.max_clips and len(all_pairs) >= args.max_clips:
                    break
            if args.max_clips and len(all_pairs) >= args.max_clips:
                break
        save_samples(all_pairs, args.output_dir)

    elif args.mode == "shards":
        def gen():
            count = 0
            for subj in subjects:
                for pair in iter_subject(args.data_root, subj, args.frame_cache):
                    yield pair
                    count += 1
                    if args.max_clips and count >= args.max_clips:
                        return
        save_webdataset(gen(), args.output_dir, split=args.split)

    elif args.mode == "hf_dataset":
        ds = build_hf_dataset(args.data_root, subjects, args.frame_cache)
        ds.save_to_disk(str(args.output_dir))
        logger.info(f"HF Dataset: {len(ds)} examples → {args.output_dir}")


if __name__ == "__main__":
    main()