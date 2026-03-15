# MTN（多尺度 Triplane 网络）
本仓库包含《Progressive Text-to-3D Generation for Automatic 3D Prototyping》论文的官方实现（https://arxiv.org/abs/2309.14600）。
### [论文](https://arxiv.org/abs/2309.14600)

### 视频结果


https://github.com/Texaser/MTN/assets/50570271/bdc776a6-ee2d-43ff-9ee3-21784799d3cb

https://github.com/Texaser/MTN/assets/50570271/197fa808-154b-4671-8446-8350b1e166d6



更多视频请参考：https://www.youtube.com/watch?v=LH6-wKg30FQ

### 使用说明：
0. 确保 `cuda-toolkit` 已正确导出：
```
nvcc -V
```
输出类似于：
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
```
1. 安装依赖：
```
conda create --name MTN python=3.9
conda activate MTN
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c conda-forge gcc=11.2.0 gxx=11.2.0
git clone https://github.com/Texaser/MTN.git
cd MTN
pip install -r requirements.txt --no-build-isolation
```
如果编译失败，你必须**覆盖当前 torch 版本，确保它与你的 `nvcc` 版本匹配**（安装说明可从 https://pytorch.org/get-started/previous-versions/ 获取）。

如果要使用 [DeepFloyd-IF](https://github.com/deep-floyd/IF)，你需要先在 [hugging face](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0) 接受使用条款，并在命令行中执行 `huggingface-cli login` 登录。

2. 开始训练！
```
# 选择 stable-diffusion 版本
python main.py --text "a rabbit, animated movie character, high detail 3d model" --workspace trial -O --sd_version 2.1

# 使用 DeepFloyd-IF 作为指导模型：

python main.py --text "a rabbit, animated movie character, high detail 3d model" --workspace trial -O --IF
python main.py --text "a rabbit, animated movie character, high detail 3d model" --workspace trial -O --IF --vram_O # 需要约 24G GPU 显存
python main.py -O --text "a rabbit, animated movie character, high detail 3d model" --workspace trial_perpneg_if_rabbit --iters 6000 --IF --batch_size 1 --perpneg
python main.py -O --text "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes" --workspace trial_perpneg_if_bunny --iters 6000 --IF --batch_size 1 --perpneg
python main.py -O --text "A high quality photo of a toy motorcycle" --workspace trial_perpneg_if_motorcycle --iters 6000 --IF --batch_size 1 --perpneg

# 下列命令使用了更大的 negative_w 绝对值，因为默认的负权重 -2 不足以让扩散模型生成期望视角
python main.py -O --text "a DSLR photo of a tiger dressed as a doctor" --workspace trial_perpneg_if_tiger --iters 6000 --IF --batch_size 1 --perpneg --negative_w -3.0

# 训练完成后：
# 测试（导出 360 度视频）
python main.py --workspace trial -O --test
# 同时保存网格模型（包含 obj、mtl 和 png 纹理）
python main.py --workspace trial -O --test --save_mesh
# 使用 GUI 测试（可自由控制视角！）
python main.py --workspace trial -O --test --gui
```
### 已测试环境
* Python 3.9、torch 1.13、CUDA 11.5，运行于 V100。
* Python 3.9、torch 1.13、CUDA 11.7，运行于 3090/4090。

### 提示
由于原始代码流程（StableDreamfusion）的原因，训练过程有时会不稳定。这种情况下，你可以尝试将 `lr` 调整为 `3e-4` 或 `5e-4`。将 `lr` 设为 `1e-5` 对模型有效收敛来说通常过小。如果模型训练失败，也可以尝试更换 prompt 或随机种子。

### 已知问题：构建 `nvdiffrast` 时 CUDA 版本不匹配

在某些环境下，**nvdiffrast** 的构建过程可能因严格的 CUDA 版本检查而失败，例如系统检测到的 CUDA Toolkit 版本（如 CUDA 12.8）与 PyTorch 编译时使用的 CUDA 版本（如 CUDA 11.7）不一致。

一些开发者使用过一种临时性绕过方法（**不建议作为永久解决方案**）：跳过 PyTorch 内部的 CUDA 版本检查。

找到你当前 PyTorch 安装中的 `cpp_extension.py` 文件，通常路径为：`~/miniconda3/envs/<your_env_name>/lib/python3.9/site-packages/torch/utils/cpp_extension.py`

你可以通过以下命令确认路径：

```python
python -c "import torch; print(torch.__file__)"
```

打开该文件，找到 `_check_cuda_version` 函数（通常在第 380–400 行附近）。定位到抛出错误的这一行：

```python
raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
```

将其替换为一个空操作，例如：

```python
# raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
print("Warning: CUDA version check temporarily bypassed (build only)")
# 或者直接：
# pass
```

保存文件后，重新尝试安装：
```python
pip install -r requirements.txt --no-build-isolation
```

## Star History
如果这份代码对你有帮助，欢迎点个 Star~
<img width="749" alt="image" src="https://github.com/user-attachments/assets/edcaf836-ac45-49f9-9a64-59fbf599a28d" />

[![Star History Chart](https://api.star-history.com/svg?repos=Texaser/MTN&type=Date)](https://www.star-history.com/#Texaser/MTN&Date)

# 引用

如果你觉得这项工作有帮助，欢迎引用：
```
@article{yi2026progressive,
  title={Progressive Text-to-3D Generation for Automatic 3D Prototyping},
  author={Yi, Han and Zheng, Zhedong and Xu, Xiangyu and Chua, Tat-seng},
  journal={ACM TOMM},
  year={2026}
}
```

## 致谢
本代码库基于以下优秀的开源项目构建：
[Stable DreamFusion](https://github.com/ashawkey/stable-dreamfusion),
[threestudio](https://github.com/threestudio-project/threestudio)

感谢这些作者们的出色工作！
