from sentence_transformers import SentenceTransformer, util
from PIL import Image
import argparse
import sys
from pathlib import Path


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default="", type=str, help="text prompt")
    parser.add_argument('--workspace', default="trial", type=str, help="text prompt")
    parser.add_argument('--latest', default='ep0001', type=str, help="which epoch result you want to use for image path")
    parser.add_argument('--mode', default='rgb', type=str, help="mode of result, color(rgb) or textureless()")
    parser.add_argument('--clip', default="clip-ViT-B-32", type=str, help="CLIP model to encode the img and prompt")

    opt = parser.parse_args()

    #Load CLIP model
    model = SentenceTransformer(f'{opt.clip}')

    #Encode text descriptions
    text_emb = model.encode([f'{opt.text}'])

    image_dir = Path(opt.workspace) / "validation"
    scores = []
    output_lines = []

    for view_idx in range(1, 9):
        image_path = image_dir / f"df_{opt.latest}_{view_idx:04d}_{opt.mode}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img_emb = model.encode(Image.open(image_path))
        cos_scores = util.cos_sim(img_emb, text_emb)
        score = float(cos_scores[0][0].cpu().numpy())
        scores.append(score)
        line = f"View {view_idx:04d}: {score:.6f}"
        output_lines.append(line)
        print(line)

    avg_score = sum(scores) / len(scores)
    avg_line = f"Average CLIP R-Precision: {avg_score:.6f}"
    output_lines.append(avg_line)
    print(avg_line)

    output_path = Path(opt.workspace) / f"r_precision_{opt.latest}_{opt.mode}.txt"
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"Saved results to: {output_path}")
