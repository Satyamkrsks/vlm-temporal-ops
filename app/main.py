"""
Phase 1: FastAPI VLM Inference Server
Accepts video uploads, returns JSON temporal operation predictions.
Model: Qwen2.5-VL-2B-Instruct
"""

import os
import json
import re
import tempfile
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─── Globals ──────────────────────────────────────────────────────────────────
_model = None
_processor = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
OPERATION_CLASSES = [
    "Box Setup", "Inner Packing", "Tape", "Put Items",
    "Pack", "Wrap", "Label", "Final Check", "Idle", "Unknown"
]

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

# ─── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="VLM Temporal Operation Intelligence API",
    description="Warehouse packaging operation classification with temporal grounding",
    version="1.0.0",
)


# ─── Pydantic Models ──────────────────────────────────────────────────────────
class TemporalSegment(BaseModel):
    start_frame: int
    end_frame: int


class PredictionResponse(BaseModel):
    clip_id: str
    dominant_operation: str
    temporal_segment: TemporalSegment
    anticipated_next_operation: str
    confidence: float


# ─── Model Loading ────────────────────────────────────────────────────────────
def get_model_and_processor():
    """Lazy-load model once on first request."""
    global _model, _processor

    if _model is None:
        from transformers import (
            Qwen2VLForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        model_name   = os.environ.get("MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")
        lora_path    = os.environ.get("LORA_CHECKPOINT", None)
        use_4bit     = os.environ.get("USE_4BIT", "false").lower() == "true"

        logger.info(f"Loading model: {model_name}")
        _processor = AutoProcessor.from_pretrained(model_name)

        load_kwargs = dict(
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        if use_4bit and torch.cuda.is_available():
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, **load_kwargs
        )

        # Load LoRA adapters if available
        if lora_path and Path(lora_path).exists():
            from peft import PeftModel
            logger.info(f"Loading LoRA adapters from: {lora_path}")
            _model = PeftModel.from_pretrained(_model, lora_path)
            _model = _model.merge_and_unload()

        _model.eval()
        logger.info("Model loaded successfully.")

    return _model, _processor


# ─── Frame Extraction ─────────────────────────────────────────────────────────
def extract_frames(video_path: str, num_frames: int = 8):
    """
    Motion-magnitude adaptive frame sampling.
    Preferentially samples high-motion frames near operation boundaries.
    """
    try:
        from decord import VideoReader, cpu

        vr           = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        if total_frames <= num_frames:
            indices = list(range(total_frames))
            frames  = vr.get_batch(indices).asnumpy()
            return frames, indices, total_frames

        # Sample candidate pool (4x target)
        step       = max(1, total_frames // (num_frames * 4))
        candidates = list(range(0, total_frames, step))
        pool_size  = min(len(candidates), 64)
        pool_frames = vr.get_batch(candidates[:pool_size]).asnumpy()

        # Compute inter-frame motion (grayscale abs-diff)
        gray = pool_frames.mean(axis=-1).astype(np.float32)
        motion_scores = [0.0]
        for i in range(1, len(pool_frames)):
            diff = np.abs(gray[i] - gray[i - 1])
            motion_scores.append(float(diff.mean()))

        # Always include first and last frame
        mandatory = {0, pool_size - 1}
        scores_copy = np.array(motion_scores, dtype=np.float32)
        for idx in mandatory:
            scores_copy[idx] = -1

        # Select top-(N-2) high-motion frames
        remaining = num_frames - len(mandatory)
        top_idx   = np.argsort(scores_copy)[::-1][:max(0, remaining)]
        selected  = sorted(mandatory | set(top_idx.tolist()))

        indices = [candidates[i] for i in selected[:num_frames]]
        frames  = vr.get_batch(sorted(indices)).asnumpy()
        return frames, sorted(indices), total_frames

    except Exception as e:
        logger.warning(f"decord failed ({e}), falling back to cv2")
        import cv2

        cap    = cv2.VideoCapture(video_path)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step   = max(1, total // num_frames)
        indices = list(range(0, total, step))[:num_frames]
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.stack(frames) if frames else np.zeros((1, 336, 336, 3), dtype=np.uint8), indices, total


# ─── Prompt Builder ───────────────────────────────────────────────────────────
def build_inference_prompt() -> str:
    ops = ", ".join(f'"{op}"' for op in OPERATION_CLASSES)
    return f"""You are analyzing a warehouse packaging operation video clip.

Examine the sequence of frames carefully and answer:
1. Which packaging operation is PRIMARILY being performed?
   Choose from: {ops}
2. In which frames does this operation START and END? (integers)
3. What operation is MOST LIKELY to happen NEXT after this clip?

Respond ONLY with JSON in exactly this format:
{{
  "dominant_operation": "<operation name>",
  "temporal_segment": {{"start_frame": <int>, "end_frame": <int>}},
  "anticipated_next_operation": "<operation name>",
  "confidence": <float 0.0-1.0>
}}"""


# ─── Inference ────────────────────────────────────────────────────────────────
def run_inference(video_path: str, clip_id: str) -> dict:
    """Full inference pipeline: extract frames → VLM → parse JSON."""
    from qwen_vl_utils import process_vision_info
    from PIL import Image

    model, processor = get_model_and_processor()
    frames, frame_indices, total_frames = extract_frames(video_path, num_frames=8)

    # Convert numpy frames to PIL Images
    pil_images = [
        Image.fromarray(frame).resize((336, 336))
        for frame in frames
    ]

    # Build Qwen2-VL message format
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in pil_images],
                {"type": "text", "text": build_inference_prompt()},
            ],
        }
    ]

    text_input    = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    # Decode only newly generated tokens
    new_tokens  = output_ids[:, inputs["input_ids"].shape[1]:]
    response    = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    return parse_response(response, total_frames, clip_id)


def parse_response(text: str, total_frames: int, clip_id: str) -> dict:
    """Parse and validate JSON from model response."""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            result = json.loads(match.group())

            # Validate operation class
            if result.get("dominant_operation") not in OPERATION_CLASSES:
                result["dominant_operation"] = "Unknown"

            # Validate next operation
            if result.get("anticipated_next_operation") not in OPERATION_CLASSES:
                result["anticipated_next_operation"] = PROCEDURAL_GRAMMAR.get(
                    result.get("dominant_operation", "Unknown"), "Unknown"
                )

            # Clamp temporal segment
            seg = result.get("temporal_segment", {})
            result["temporal_segment"] = {
                "start_frame": int(max(0, seg.get("start_frame", 0))),
                "end_frame":   int(min(total_frames - 1, seg.get("end_frame", total_frames - 1))),
            }

            # Clamp confidence
            result["confidence"] = float(max(0.0, min(1.0, result.get("confidence", 0.5))))
            result["clip_id"]    = clip_id
            return result

    except Exception as e:
        logger.warning(f"JSON parse failed: {e}. Response was: {text[:200]}")

    # Fallback if parse fails
    return {
        "clip_id":                   clip_id,
        "dominant_operation":        "Unknown",
        "temporal_segment":          {"start_frame": 0, "end_frame": total_frames - 1},
        "anticipated_next_operation": "Unknown",
        "confidence":                 0.1,
    }


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "VLM Temporal Operation Intelligence API", "docs": "/docs"}


@app.get("/health")
async def health():
    return {
        "status":         "ok",
        "cuda_available": torch.cuda.is_available(),
        "model_loaded":   _model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file:    UploadFile = File(...),
    clip_id: Optional[str] = None,
):
    """
    Upload a video file, get temporal operation prediction.

    - **file**: Video file (.mp4, .avi, .mkv, .mov)
    - **clip_id**: Optional clip identifier (defaults to filename stem)
    """
    allowed = {".mp4", ".avi", ".mkv", ".mov"}
    suffix  = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Use: {allowed}"
        )

    if clip_id is None:
        clip_id = Path(file.filename).stem

    # Write upload to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content  = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = run_inference(tmp_path, clip_id)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return JSONResponse(content=result)


@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Batch prediction for multiple clips."""
    results = []
    for file in files:
        clip_id = Path(file.filename).stem
        suffix  = Path(file.filename).suffix.lower()

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            result = run_inference(tmp_path, clip_id)
            results.append(result)
        except Exception as e:
            results.append({"clip_id": clip_id, "error": str(e)})
        finally:
            os.unlink(tmp_path)

    return JSONResponse(content={"predictions": results, "count": len(results)})


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)