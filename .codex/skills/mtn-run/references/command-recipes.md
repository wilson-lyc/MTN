# MTN Command Recipes

Use this file when the user needs a concrete command quickly.

## Core Patterns

### 1. Standard text-to-3D training

```bash
python main.py --text "<prompt>" --workspace <workspace> -O
```

Optional additions:
- `--sd_version 2.1`
- `--iters 6000`
- `--seed 3407`

### 2. DeepFloyd-IF training

```bash
python main.py --text "<prompt>" --workspace <workspace> -O --IF
```

Lower-VRAM variant:

```bash
python main.py --text "<prompt>" --workspace <workspace> -O --IF --vram_O
```

### 3. Perp-Neg training

```bash
python main.py -O --text "<prompt>" --workspace <workspace> --iters 6000 --IF --batch_size 1 --perpneg
```

When the user explicitly wants stronger negative guidance:

```bash
python main.py -O --text "<prompt>" --workspace <workspace> --iters 6000 --IF --batch_size 1 --perpneg --negative_w -3.0
```

### 4. Test a finished workspace

```bash
python main.py --workspace <workspace> -O --test
```

### 5. Export a textured mesh

```bash
python main.py --workspace <workspace> -O --test --save_mesh
```

### 6. Open GUI on a trained result

```bash
python main.py --workspace <workspace> -O --test --gui
```

### 7. Save six fixed views

```bash
python main.py --workspace <workspace> -O --six_views
```

### 8. Image-conditioned generation with one image

Text + image:

```bash
python main.py --text "<prompt>" --image <image_path> --workspace <workspace> -O
```

Image only:

```bash
python main.py --image <image_path> --workspace <workspace> -O
```

### 9. Multi-view image-conditioned generation

```bash
python main.py --image_config <views.csv> --workspace <workspace> -O
```

Optional text conditioning:

```bash
python main.py --text "<prompt>" --image_config <views.csv> --workspace <workspace> -O
```

### 10. DMTet finetuning

```bash
python main.py --workspace <workspace> -O --dmtet --init_with <checkpoint_or_mesh>
```

Optional mesh export after finetuning:

```bash
python main.py --workspace <workspace> -O --dmtet --init_with <checkpoint_or_mesh> --save_mesh
```

### 11. Preprocess an input image

```bash
python preprocess_image.py <image_path> --size 256 --border_ratio 0.2
```

Disable recentering:

```bash
python preprocess_image.py <image_path> --size 256 --border_ratio 0.2 --dont_recenter
```

## Parameter Notes

- `-O` is the default fast path for most examples.
- `--workspace` should be unique for new training runs and stable for test/export commands.
- `--vram_O` is a memory optimization, not a quality mode.
- `--save_mesh` is useful only after geometry exists in the workspace.
- `--image` and `--image_config` change MTN's internal guidance behavior even if guidance flags are omitted.
- `--IF` usually implies the user has already accepted the DeepFloyd model license and logged into Hugging Face.

## When To Ask Instead Of Guessing

Ask the user for clarification if any of these are missing:

- prompt text for a text-driven run
- image path or CSV path for image-driven runs
- workspace name for test/export on an existing run
- init checkpoint or mesh path for DMTet when they explicitly request finetuning
