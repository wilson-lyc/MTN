---
name: mtn-run
description: Generate MTN project run commands from user intent. Use when the user wants to run this repository, asks for a training or inference command, wants to convert requirements into a `python main.py` or `python preprocess_image.py` invocation, or needs help choosing flags for text-to-3D, DeepFloyd-IF, image-conditioned generation, testing, GUI preview, mesh export, or DMTet finetuning.
---

# MTN Run

## Overview

Translate the user's goal into a runnable command for this repository.
Prefer concise output: give the final command first, then list only the assumptions or required prerequisites that materially affect execution.

## Workflow

1. Identify the user's task category.
2. Map the task to the smallest command that satisfies it.
3. Add only the flags that are required by the user's request or by MTN's CLI behavior.
4. State blockers separately when the command depends on unavailable inputs, models, or environment setup.

## Classify The Request

- Text-to-3D training: build a `python main.py` command with `--text`, `--workspace`, and usually `-O`.
- IF-guided training: add `--IF`; add `--vram_O` only when the user mentions limited VRAM or asks for the lower-memory variant.
- Image-conditioned generation: use `--image <path>` or `--image_config <csv>`; keep `--text` only if the user wants image+text conditioning.
- Test or export: start from the existing workspace and add `--test`, then optionally `--save_mesh` or `--gui`.
- Six-view render: use `--six_views` instead of `--test` when the user explicitly wants the six fixed views.
- DMTet finetuning: add `--dmtet`, plus `--init_with <ckpt-or-mesh>` when the source asset is specified.
- Image preprocessing: use `python preprocess_image.py ...` when the user asks to prepare an input image before training.

## Build The Command

Use these defaults unless the user requests otherwise:

- Use `python main.py` for train, test, GUI, mesh export, and six-view rendering.
- Use `python preprocess_image.py` for image preprocessing.
- Use `-O` for most practical runs because it expands to `--fp16 --cuda_ray`.
- Keep `--sd_version 2.1` only when the user mentions Stable Diffusion versioning or when including it makes the command clearer.
- Always include `--workspace <name>` for anything that writes outputs.
- Keep prompts quoted.
- Preserve user-provided paths and names exactly.

## Respect CLI Coupling

Account for these MTN-specific behaviors when generating commands:

- `-O` implies `--fp16 --cuda_ray`.
- `-O2` implies `--fp16 --backbone vanilla --progressive_level`.
- `--IF` replaces the default `SD` guidance internally; do not add `--guidance IF` unless the user explicitly asks for guidance list control.
- If the user provides only `--image` or `--image_config` and no text, MTN switches to `zero123` guidance internally.
- If the user provides both text and image, MTN uses `SD` plus `clip` internally and overrides several defaults.
- `--save_mesh` is typically paired with `--test`, `--six_views`, or a finished training workspace.
- `--gui` can be used in test mode or training mode; prefer `--test --gui` when the user wants to inspect a finished run.

## Ask Only For Missing Required Inputs

Ask a concise follow-up only when the command cannot be responsibly completed without one of these:

- text prompt for text-to-3D training
- image path or CSV path for image-conditioned runs
- workspace name when the user is referring to an existing run and no workspace can be inferred
- initialization checkpoint or mesh for DMTet when requested explicitly

If the command is still useful with assumptions, make the assumption explicit and proceed.

## Response Shape

Prefer this format:

```bash
python main.py ...
```

Then add at most three short notes:

- why a non-obvious flag was included
- prerequisite the user must satisfy before running it
- assumption you made to fill in a missing value

## References

Load [references/command-recipes.md](references/command-recipes.md) when you need concrete MTN command templates or parameter groupings.
Re-check `main.py`, `README.md`, or `PARAMETERS_ZH.md` if the repository appears to have diverged from the reference file.
