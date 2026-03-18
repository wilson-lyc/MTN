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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default="a lion, animated movie character, high detail 3d model", type=str)
    parser.add_argument('--path', default="./result/baseline/validation", type=str, help="directory containing the evaluation images")
    parser.add_argument('--latest', default='ep00600', type=str)
    parser.add_argument('--mode', default='rgb', type=str)
    parser.add_argument('--clip', default="openai/clip-vit-base-patch32", type=str)

    opt = parser.parse_args()

    model_name = resolve_clip_model_name(opt.clip)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    with torch.no_grad():
        text_inputs = processor(text=[opt.text], return_tensors="pt", padding=True).to(device)
        text_emb = F.normalize(model.get_text_features(**text_inputs), dim=-1)

    image_dir = Path(opt.path)
    scores = []
    output_lines = []

    for view_idx in range(1, 9):
        image_path = image_dir / f"df_{opt.latest}_{view_idx:04d}_{opt.mode}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            img_emb = F.normalize(model.get_image_features(**image_inputs), dim=-1)
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
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "score"])
        for view_idx, score in enumerate(scores, start=1):
            writer.writerow([f"df_{opt.latest}_{view_idx:04d}_{opt.mode}.png", f"{score:.6f}"])
        writer.writerow(["average", f"{avg_score:.6f}"])
    print(f"Saved results to: {output_path}")
