import os
import time
import gc
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


MODEL_NAME = "microsoft/Florence-2-large"

_processor = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_torch_dtype = torch.float16 if _device == "cuda" else torch.float32


def _load_model():
    global _processor, _model
    if _processor is None or _model is None:
        print("\n" + "=" * 60)
        print("  LOADING MICROSOFT FLORENCE-2 VISION-LANGUAGE MODEL")
        print("=" * 60)
        print(f"  Model: {MODEL_NAME}")
        print(f"  Device: {_device.upper()}")
        print(f"  Precision: {'float16' if _device == 'cuda' else 'float32'}")
        print("  Loading processor...")
        start = time.time()
        _processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print(f"  Processor loaded ({time.time() - start:.1f}s)")

        print("  Loading model weights...")
        start = time.time()
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=_torch_dtype,
            trust_remote_code=True
        ).to(_device)

        # Patch for newer transformers compatibility
        if not hasattr(_model.config, "forced_bos_token_id"):
            _model.config.forced_bos_token_id = 0
        if hasattr(_model.config, "text_config") and not hasattr(_model.config.text_config, "forced_bos_token_id"):
            _model.config.text_config.forced_bos_token_id = 0
        if hasattr(_model, "generation_config") and _model.generation_config is not None:
            if not hasattr(_model.generation_config, "forced_bos_token_id"):
                _model.generation_config.forced_bos_token_id = 0

        _model.eval()
        print(f"  Model loaded ({time.time() - start:.1f}s)")
        print("  Florence-2 ready for inference\n")
    return _processor, _model


IMAGE_TASK = "<MORE_DETAILED_CAPTION>"
FLOWCHART_TASK = "<MORE_DETAILED_CAPTION>"


def caption_image(image_path, content_type="image"):
    try:
        processor, model = _load_model()

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        print(f"      Image dimensions: {width} x {height} px")

        task_prompt = FLOWCHART_TASK if content_type == "flowchart" else IMAGE_TASK
        print(f"      Task: {task_prompt}")

        start = time.time()
        inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(_device, _torch_dtype)

        print(f"      Running inference on {_device.upper()}...")
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
                do_sample=False,
                early_stopping=False
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )

        caption = parsed_answer.get(task_prompt, "").strip()

        # Free inference tensors immediately
        del inputs, generated_ids, generated_text, parsed_answer
        gc.collect()

        elapsed = time.time() - start
        print(f"      Inference complete ({elapsed:.1f}s)")
        print(f"      Caption length: {len(caption)} characters")
        print(f"      Caption preview: {caption[:120]}...")

        return caption

    except Exception as e:
        print(f"      ERROR captioning: {e}")
        return f"Caption generation failed for {os.path.basename(image_path)}"


def caption_all_images(images):
    print("\n" + "=" * 60)
    print("  STAGE: VLM IMAGE CAPTIONING (Microsoft Florence-2)")
    print("=" * 60)
    print(f"  Total visuals to process: {len(images)}")

    flowchart_total = sum(1 for img in images if img["content_type"] == "flowchart")
    image_total = len(images) - flowchart_total
    print(f"  Breakdown: {image_total} images, {flowchart_total} flowcharts")
    print("-" * 60)

    captioned = []
    seen_captions = set()
    duplicates_skipped = 0
    total = len(images)
    start_all = time.time()

    for i, img in enumerate(images):
        print(f"\n  [{i+1}/{total}] Processing: {os.path.basename(img['path'])}")
        print(f"      Type: {img['content_type'].upper()}")
        print(f"      Source: {img['source_file']} (page {img['page']})")

        caption = caption_image(img["path"], img["content_type"])

        # Deduplicate by caption content (same image gets same caption)
        caption_hash = hash(caption.strip().lower())
        if caption_hash in seen_captions:
            duplicates_skipped += 1
            print(f"      SKIPPED (duplicate caption)")
            continue
        seen_captions.add(caption_hash)

        prefix = f"[FLOWCHART on page {img['page']}]" if img["content_type"] == "flowchart" else f"[IMAGE on page {img['page']}]"

        captioned.append({
            "text": f"{prefix} {caption}",
            "content_type": img["content_type"],
            "page": img["page"],
            "source_file": img["source_file"],
            "path": img["path"]
        })

    total_time = time.time() - start_all
    print("\n" + "-" * 60)
    print(f"  CAPTIONING COMPLETE")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Average per image: {total_time / max(1, total):.1f}s")
    print(f"  Captioned: {len(captioned)} unique items")
    if duplicates_skipped > 0:
        print(f"  Deduplication: skipped {duplicates_skipped} duplicate caption(s)")
    print("=" * 60 + "\n")

    # Free memory after captioning batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return captioned


def unload_model():
    """Free VLM memory completely. Call after all captioning is done."""
    global _processor, _model
    if _model is not None:
        del _model
        _model = None
    if _processor is not None:
        del _processor
        _processor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  VLM model unloaded from memory")