import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


CLIP_MODEL_ALIASES = {
    "clip-ViT-B-32": "openai/clip-vit-base-patch32",
    "clip-ViT-B-16": "openai/clip-vit-base-patch16",
    "clip-ViT-L-14": "openai/clip-vit-large-patch14",
    "CLIP B/16": "openai/clip-vit-base-patch16",
    "CLIP L/14": "openai/clip-vit-large-patch14",
}


def resolve_clip_model_name(name: str) -> str:
    return CLIP_MODEL_ALIASES.get(name, name)


def log(message: str) -> None:
    print(f"[evaluation] {message}", flush=True)


def extract_text_embedding(model, text_inputs):
    text_kwargs = {
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs.get("attention_mask"),
    }

    if hasattr(model, "get_text_features"):
        try:
            outputs = model.get_text_features(**text_kwargs)
            if torch.is_tensor(outputs):
                return outputs
            if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
                return outputs.text_embeds
        except Exception:
            pass

    text_outputs = model.text_model(**text_kwargs)
    pooled = text_outputs.pooler_output
    return model.text_projection(pooled)


def extract_image_embedding(model, image_inputs):
    image_kwargs = {"pixel_values": image_inputs["pixel_values"]}

    if hasattr(model, "get_image_features"):
        try:
            outputs = model.get_image_features(**image_kwargs)
            if torch.is_tensor(outputs):
                return outputs
            if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                return outputs.image_embeds
        except Exception:
            pass

    vision_outputs = model.vision_model(**image_kwargs)
    pooled = vision_outputs.pooler_output
    return model.visual_projection(pooled)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default="a lion, animated movie character, high detail 3d model", type=str)
    parser.add_argument('--path', default="./result/baseline/validation", type=str, help="directory containing the evaluation images")
    parser.add_argument('--latest', default='ep0060', type=str)
    parser.add_argument('--mode', default='rgb', type=str)
    parser.add_argument('--clip', default="openai/clip-vit-base-patch32", type=str)

    opt = parser.parse_args()

    model_name = resolve_clip_model_name(opt.clip)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir = Path(opt.path)

    log(f"Starting evaluation")
    log(f"Image directory: {image_dir}")
    log(f"Prompt: {opt.text}")
    log(f"CLIP model: {model_name}")
    log(f"Device: {device}")

    log("Loading CLIP model weights...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    log("Loading CLIP processor...")
    processor = CLIPProcessor.from_pretrained(model_name)
    log("Model and processor are ready")

    log("Encoding text prompt...")
    with torch.no_grad():
        text_inputs = processor(text=[opt.text], return_tensors="pt", padding=True).to(device)
        text_emb = F.normalize(extract_text_embedding(model, text_inputs), dim=-1)
    log(f"Text embedding ready with shape {tuple(text_emb.shape)}")

    scores = []
    output_lines = []

    for view_idx in range(1, 9):
        image_path = image_dir / f"df_{opt.latest}_{view_idx:04d}_{opt.mode}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        log(f"Processing image {view_idx}/8: {image_path.name}")
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            img_emb = F.normalize(extract_image_embedding(model, image_inputs), dim=-1)
            score = float((img_emb @ text_emb.T).squeeze().item())
        scores.append(score)
        line = f"View {view_idx:04d}: {score:.6f}"
        output_lines.append(line)
        print(line)

    avg_score = sum(scores) / len(scores)
    avg_line = f"Average CLIP Similarity: {avg_score:.6f}"
    output_lines.append(avg_line)
    print(avg_line)

    output_path = image_dir / f"clip_score_{opt.latest}_{opt.mode}.csv"
    log(f"Writing CSV results to {output_path}")
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "score"])
        for view_idx, score in enumerate(scores, start=1):
            writer.writerow([f"df_{opt.latest}_{view_idx:04d}_{opt.mode}.png", f"{score:.6f}"])
        writer.writerow(["average", f"{avg_score:.6f}"])
    log(f"Saved results to: {output_path}")
