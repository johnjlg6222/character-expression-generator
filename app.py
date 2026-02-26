"""Gradio UI for Character Expression Generator."""

import os
import tempfile
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from PIL import Image

from generator import EMOTIONS, generate_batch_sync

load_dotenv()


def generate(
    ref_image,
    selected_emotions: list[str],
    resolution: str,
    remove_bg: bool,
    normalize_colors: bool,
    progress=gr.Progress(),
):
    """Main generation function called by the UI."""
    if ref_image is None:
        gr.Warning("Please upload a reference image.")
        return [], "No reference image provided."

    if not selected_emotions:
        gr.Warning("Please select at least one emotion.")
        return [], "No emotions selected."

    token = os.environ.get("REPLICATE_API_TOKEN") or os.environ.get("REPLICATE_API_KEY")
    if not token:
        gr.Warning("REPLICATE_API_TOKEN not set. Add it to your .env file.")
        return [], "REPLICATE_API_TOKEN not set."

    cost_estimate = len(selected_emotions) * 0.04
    progress(0, desc=f"Generating {len(selected_emotions)} variants (~${cost_estimate:.2f})...")

    def progress_callback(current, total, message):
        progress(current / max(total, 1), desc=message)

    pil_ref = Image.fromarray(ref_image) if not isinstance(ref_image, Image.Image) else ref_image
    pil_ref = pil_ref.convert("RGB")

    results = generate_batch_sync(
        ref_image=pil_ref,
        emotions=selected_emotions,
        resolution=resolution,
        remove_bg=remove_bg,
        normalize_colors=normalize_colors,
        progress_callback=progress_callback,
    )

    # Build gallery items and log
    gallery_items = []
    log_lines = []
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    for emotion_key, img, err in results:
        if img is not None:
            out_path = output_dir / f"{emotion_key}.png"
            img.save(str(out_path))
            gallery_items.append((img, emotion_key))
            log_lines.append(f"{emotion_key}: saved to {out_path}")
        else:
            log_lines.append(f"{emotion_key}: FAILED - {err}")

    succeeded = sum(1 for _, img, _ in results if img is not None)
    failed = len(results) - succeeded
    cost = succeeded * 0.04

    summary = (
        f"Generated {succeeded}/{len(results)} variants | "
        f"Cost: ~${cost:.2f} | "
        f"Saved to outputs/\n\n" +
        "\n".join(log_lines)
    )

    return gallery_items, summary


def build_ui():
    """Build the Gradio interface."""
    with gr.Blocks(title="Character Expression Generator") as app:
        gr.Markdown("# Character Expression Generator")
        gr.Markdown(
            "Upload a reference image, select emotions, and generate consistent expression variants "
            "using Nano Banana Pro. ~$0.04/image via Replicate."
        )

        with gr.Row():
            with gr.Column(scale=1):
                ref_input = gr.Image(
                    label="Reference Image",
                    type="numpy",
                    height=400,
                )

                emotion_checkboxes = gr.CheckboxGroup(
                    choices=list(EMOTIONS.keys()),
                    value=["neutral", "happy", "sad"],
                    label="Emotions",
                )

                with gr.Row():
                    resolution = gr.Radio(
                        choices=["1K", "2K", "4K"],
                        value="2K",
                        label="Resolution",
                    )

                with gr.Row():
                    remove_bg = gr.Checkbox(value=True, label="Remove background")
                    normalize_colors = gr.Checkbox(value=True, label="Normalize colors")

                cost_display = gr.Markdown("**Estimated cost: $0.12** (3 images)")

                generate_btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="Generated Variants",
                    columns=3,
                    height=500,
                    object_fit="contain",
                )
                log_output = gr.Textbox(label="Log", lines=6, interactive=False)

        # Update cost estimate when emotions change
        def update_cost(emotions):
            n = len(emotions)
            cost = n * 0.04
            return f"**Estimated cost: ${cost:.2f}** ({n} image{'s' if n != 1 else ''})"

        emotion_checkboxes.change(
            fn=update_cost,
            inputs=[emotion_checkboxes],
            outputs=[cost_display],
        )

        generate_btn.click(
            fn=generate,
            inputs=[ref_input, emotion_checkboxes, resolution, remove_bg, normalize_colors],
            outputs=[gallery, log_output],
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(theme=gr.themes.Soft())
