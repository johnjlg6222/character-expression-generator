"""Core generation logic: Nano Banana Pro via Replicate + rembg background removal."""

import asyncio
import base64
import io
import os

import httpx
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

MODEL_VERSION = "712e06a8e122fb7c8dae55dcf7ad6a8e717afb7b1c41c889fc8c5132fd42f374"

EMOTIONS = {
    "neutral": "a calm neutral expression, looking straight ahead",
    "happy": "a subtle gentle smile, slight happiness",
    "sad": "a slightly sad expression, subtle melancholy, slight frown",
    "angry": "a slightly angry expression, subtle tension in the jaw, furrowed brows",
    "surprised": "a slightly surprised expression, subtly raised eyebrows, parted lips",
    "disgusted": "a slightly disgusted expression, subtle nose wrinkle, lip curl",
    "fearful": "a slightly fearful expression, subtly widened eyes, tense mouth",
    "confident": "a confident self-assured expression, subtle smirk, chin slightly raised",
    "thoughtful": "a thoughtful pensive expression, eyes slightly narrowed, subtle contemplation",
    "flirty": "a subtly flirtatious expression, slight smirk, warm inviting eyes",
}


def _encode_pil_to_data_uri(pil_img: Image.Image) -> str:
    """Encode a PIL image to a PNG data URI."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _resize_for_model(pil_img: Image.Image, max_side: int = 2048) -> Image.Image:
    """Downscale if larger than max_side, keeping aspect ratio."""
    w, h = pil_img.size
    if max(w, h) <= max_side:
        return pil_img
    scale = max_side / max(w, h)
    new_w = (int(w * scale) // 8) * 8
    new_h = (int(h * scale) // 8) * 8
    return pil_img.resize((new_w, new_h), Image.LANCZOS)


async def _poll_prediction(client: httpx.AsyncClient, prediction: dict, token: str) -> dict:
    """Poll a Replicate prediction until terminal state."""
    if prediction.get("status") in ("succeeded", "failed", "canceled"):
        return prediction
    poll_url = prediction["urls"]["get"]
    headers = {"Authorization": f"Bearer {token}"}
    while True:
        await asyncio.sleep(3)
        resp = await client.get(poll_url, headers=headers)
        resp.raise_for_status()
        prediction = resp.json()
        if prediction["status"] in ("succeeded", "failed", "canceled"):
            return prediction


async def _generate_one(
    client: httpx.AsyncClient,
    ref_uri: str,
    emotion_key: str,
    emotion_desc: str,
    token: str,
    resolution: str,
) -> tuple[str, Image.Image | None, str | None]:
    """Generate a single variant. Returns (emotion_key, pil_image_or_None, error_or_None)."""
    prompt = (
        f"Edit this exact person to have {emotion_desc}. "
        "Keep the exact same face, hair, clothing, necklace, lighting, and background. "
        "Only change the facial expression."
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }

    payload = {
        "version": MODEL_VERSION,
        "input": {
            "image_input": [ref_uri],
            "prompt": prompt,
            "resolution": resolution,
            "aspect_ratio": "match_input_image",
            "output_format": "png",
            "safety_filter_level": "block_only_high",
            "allow_fallback_model": True,
        },
    }

    for attempt in range(3):
        try:
            resp = await client.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            prediction = await _poll_prediction(client, resp.json(), token)

            if prediction["status"] != "succeeded":
                error = prediction.get("error", "Unknown error")
                return emotion_key, None, error

            output_url = prediction["output"]
            if isinstance(output_url, list):
                output_url = output_url[0]
            img_resp = await client.get(output_url)
            img_resp.raise_for_status()
            pil_img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            return emotion_key, pil_img, None

        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 ** (attempt + 1))
            else:
                return emotion_key, None, str(e)

    return emotion_key, None, "All retries exhausted"


def _normalize_lab_to_reference(ref_img: Image.Image, target_img: Image.Image) -> Image.Image:
    """Shift target's LAB mean to match ref's LAB mean (fixes color drift)."""
    import cv2

    ref_arr = np.array(ref_img, dtype=np.float64)
    tgt_arr = np.array(target_img, dtype=np.float64)

    ref_lab = cv2.cvtColor(ref_arr.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)
    tgt_lab = cv2.cvtColor(tgt_arr.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)

    for c in range(3):
        tgt_lab[:, :, c] += ref_lab[:, :, c].mean() - tgt_lab[:, :, c].mean()

    tgt_lab = np.clip(tgt_lab, 0, 255)
    corrected = cv2.cvtColor(tgt_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(corrected)


def _remove_background(pil_img: Image.Image) -> Image.Image:
    """Remove background using rembg, returns RGBA."""
    from rembg import remove
    return remove(pil_img)


async def generate_batch(
    ref_image: Image.Image,
    emotions: list[str],
    resolution: str = "2K",
    remove_bg: bool = True,
    normalize_colors: bool = True,
    progress_callback=None,
) -> list[tuple[str, Image.Image | None, str | None]]:
    """Generate a batch of expression variants.

    Args:
        ref_image: Reference PIL image.
        emotions: List of emotion keys (from EMOTIONS dict).
        resolution: "1K", "2K", or "4K".
        remove_bg: Whether to remove background with rembg.
        normalize_colors: Whether to normalize LAB colors to match first result.
        progress_callback: Optional callback(current, total, message).

    Returns:
        List of (emotion_key, pil_image_or_None, error_or_None).
    """
    token = os.environ.get("REPLICATE_API_TOKEN") or os.environ.get("REPLICATE_API_KEY")
    if not token:
        return [(e, None, "REPLICATE_API_TOKEN not set") for e in emotions]

    # Prepare reference
    ref_resized = _resize_for_model(ref_image)
    ref_uri = _encode_pil_to_data_uri(ref_resized)

    total = len(emotions)
    if progress_callback:
        progress_callback(0, total, "Starting generation...")

    # Generate all variants
    results: list[tuple[str, Image.Image | None, str | None]] = []
    async with httpx.AsyncClient(timeout=300) as client:
        # Run sequentially to avoid overloading the model
        for i, emotion_key in enumerate(emotions):
            desc = EMOTIONS.get(emotion_key, emotion_key)
            if progress_callback:
                progress_callback(i, total, f"Generating {emotion_key}...")

            key, img, err = await _generate_one(
                client, ref_uri, emotion_key, desc, token, resolution
            )
            results.append((key, img, err))

    if progress_callback:
        progress_callback(total, total, "Post-processing...")

    # Normalize colors: use first successful result as anchor
    if normalize_colors:
        anchor = None
        for _, img, err in results:
            if img is not None:
                anchor = img
                break

        if anchor is not None:
            results = [
                (key, _normalize_lab_to_reference(anchor, img) if img is not None else None, err)
                for key, img, err in results
            ]

    # Remove backgrounds
    if remove_bg:
        normalized_results = []
        for key, img, err in results:
            if img is not None:
                img = _remove_background(img)
            normalized_results.append((key, img, err))
        results = normalized_results

    return results


def generate_batch_sync(
    ref_image: Image.Image,
    emotions: list[str],
    resolution: str = "2K",
    remove_bg: bool = True,
    normalize_colors: bool = True,
    progress_callback=None,
) -> list[tuple[str, Image.Image | None, str | None]]:
    """Sync wrapper for generate_batch."""
    return asyncio.run(generate_batch(
        ref_image, emotions, resolution, remove_bg, normalize_colors, progress_callback
    ))
