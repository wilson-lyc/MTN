# MTN 参数说明（中文版）

本文档整理了项目中主要命令行参数，来源于 [main.py](/Users/wilson/Projects/MTN/main.py) 和 [preprocess_image.py](/Users/wilson/Projects/MTN/preprocess_image.py)。

## 1. `main.py` 参数

基本用法示例：

```bash
python main.py --text "a rabbit, animated movie character, high detail 3d model" --workspace trial -O --sd_version 2.1
```

### 1.1 快速开关与运行模式

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--file` | 无 | 从文件中读取更多命令行参数。 |
| `-O` | 关闭 | 等价于启用 `--fp16 --cuda_ray`。 |
| `-O2` | 关闭 | 等价于启用 `--fp16`，并将 `--backbone` 设为 `vanilla`，同时启用 `--progressive_level`。 |
| `--test` | 关闭 | 测试模式。 |
| `--six_views` | 关闭 | 输出六个固定视角的图像。 |
| `--eval_interval` | `10` | 每隔多少个 epoch 在验证集上评估一次。 |
| `--test_interval` | `100` | 每隔多少个 epoch 在测试集上测试一次。 |
| `--workspace` | `workspace` | 工作目录，用于保存日志、检查点和导出结果。 |
| `--seed` | `3407` | 随机种子。 |

### 1.2 文本与图像输入

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--text` | `None` | 文本提示词。 |
| `--negative` | 空字符串 | 负向提示词。 |
| `--image` | `None` | 单张图像输入，用于图像条件生成。 |
| `--image_config` | `None` | 多视图图像配置 CSV。 |
| `--known_view_interval` | `4` | 每隔多少次迭代对默认视角施加一次 RGB 损失，仅在提供 `--image` 时生效。 |

### 1.3 Guidance 与扩散模型

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--IF` | 关闭 | 使用 DeepFloyd-IF 作为 NeRF 阶段的 guidance 模型。 |
| `--guidance` | `['SD']` | guidance 模型列表。 |
| `--guidance_scale` | `100` | 扩散模型 classifier-free guidance scale。 |
| `--sd_version` | `2.1` | Stable Diffusion 版本，可选 `1.5`、`2.0`、`2.1`。 |
| `--hf_key` | `None` | Hugging Face 上 Stable Diffusion 模型的 key。 |
| `--zero123_config` | `./pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml` | zero123 配置文件路径。 |
| `--zero123_ckpt` | `./pretrained/zero123/105000.ckpt` | zero123 权重路径。 |
| `--zero123_grad_scale` | `angle` | zero123 梯度缩放方式。 |
| `--t_range` | `[0.02, 0.98]` | Stable Diffusion 时间步范围。 |
| `--dont_override_stuff` | 关闭 | 关闭图像条件模式下的一些自动参数覆盖。 |

### 1.4 网格导出与 DMTet

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--save_mesh` | 关闭 | 导出带纹理的 OBJ 网格。 |
| `--mcubes_resolution` | `256` | Marching Cubes 提取网格时的分辨率。 |
| `--decimate_target` | `50000` | 网格简化的目标面数。 |
| `--dmtet` | 关闭 | 启用 DMTet 微调。 |
| `--tet_grid_size` | `128` | 四面体网格分辨率。 |
| `--init_with` | 空字符串 | 用于初始化 DMTet 的检查点路径。 |
| `--lock_geo` | 关闭 | 锁定几何，不让 DMTet 学习几何形状。 |
| `--dmtet_reso_scale` | `8` | DMTet 微调时，对训练渲染分辨率 `--h/--w` 的放大倍数。 |

### 1.5 Perp-Neg 相关

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--perpneg` | 关闭 | 启用 Perp-Neg。 |
| `--negative_w` | `-2` | 负向提示词权重。绝对值更大通常更能缓解 Janus 问题，但可能导致表面过平。 |
| `--front_decay_factor` | `2` | 正面提示衰减系数。 |
| `--side_decay_factor` | `10` | 侧面提示衰减系数。 |

### 1.6 训练控制

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--iters` | `6000` | 总训练迭代数。 |
| `--warm_iters` | `600` | 预热迭代数。 |
| `--lr` | `1e-3` | 最大学习率。 |
| `--ckpt` | `latest` | 检查点加载策略，可选 `latest`、`scratch`、`best`、`latest_model`。 |
| `--optim` | `adan` | 优化器，可选 `adan` 或 `adam`。 |
| `--latent_iter_ratio` | `0.2` | 仅使用 albedo 着色的训练前期比例。 |
| `--albedo_iter_ratio` | `0` | 使用 albedo 着色的训练比例。 |
| `--batch_size` | `1` | 每批渲染的图像数量。 |
| `--dataset_size_train` | `100` | 训练集长度，即每个 epoch 的迭代次数。 |
| `--dataset_size_valid` | `8` | 验证时转台视频渲染帧数。 |
| `--dataset_size_test` | `100` | 测试时转台视频渲染帧数。 |
| `--exp_start_iter` | `None` | 实验起始迭代号，用于计算 progressive 相关调度。 |
| `--exp_end_iter` | `None` | 实验结束迭代号，用于计算 progressive 相关调度。 |

### 1.7 光线步进与显存相关

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--cuda_ray` | 关闭 | 使用 CUDA raymarching 替代 PyTorch 实现。 |
| `--taichi_ray` | 关闭 | 使用 Taichi raymarching。 |
| `--max_steps` | `1024` | 每条光线的最大采样步数，仅在 `--cuda_ray` 下有效。 |
| `--num_steps` | `64` | 每条光线的采样步数，仅在未启用 `--cuda_ray` 时有效。 |
| `--upsample_steps` | `32` | 每条光线的额外上采样步数，仅在未启用 `--cuda_ray` 时有效。 |
| `--update_extra_interval` | `16` | 更新额外状态的迭代间隔，仅在 `--cuda_ray` 下有效。 |
| `--max_ray_batch` | `4096` | 推理时每批光线数，用于降低 OOM 风险，仅在未启用 `--cuda_ray` 时有效。 |
| `--fp16` | 关闭 | 使用半精度训练。 |
| `--vram_O` | 关闭 | 低显存优化。 |
| `--grad_clip` | `-1` | 对所有梯度做裁剪，负值表示关闭。 |
| `--grad_clip_rgb` | `-1` | 对 RGB 空间梯度做裁剪，负值表示关闭。 |

### 1.8 模型与体渲染

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--bg_radius` | `1.4` | 大于 0 时启用球形背景模型。 |
| `--density_activation` | `exp` | 密度激活函数，可选 `softplus` 或 `exp`。 |
| `--density_thresh` | `10` | 占据网格的密度阈值。 |
| `--blob_density` | `5` | 初始密度 blob 的中心密度。 |
| `--blob_radius` | `0.2` | 初始密度 blob 的半径。 |
| `--backbone` | `grid` | NeRF 主干网络，可选 `grid_tcnn`、`grid`、`vanilla`、`grid_taichi`。 |
| `--w` | `64` | 训练时 NeRF 渲染宽度。 |
| `--h` | `64` | 训练时 NeRF 渲染高度。 |
| `--known_view_scale` | `1.5` | 已知视角渲染时，对 `--h/--w` 的放大倍数。 |
| `--known_view_noise_scale` | `2e-3` | 加到 `rays_o` 和 `rays_d` 上的随机相机噪声。 |
| `--bound` | `1` | 场景边界盒范围 `(-bound, bound)`。 |
| `--dt_gamma` | `0` | 自适应 ray marching 参数，大于 0 可加速，但通常会牺牲质量。 |
| `--min_near` | `0.01` | 相机最近裁剪距离。 |
| `--min_ambient_ratio` | `0.1` | Lambertian 着色中的最小环境光比例。 |
| `--textureless_ratio` | `0.2` | 无纹理着色占比。 |

### 1.9 相机采样与视角调度

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--radius_range` | `[3.0, 3.5]` | 训练时相机半径采样范围。 |
| `--theta_range` | `[45, 105]` | 训练时极角范围。 |
| `--phi_range` | `[-180, 180]` | 训练时方位角范围。 |
| `--fovy_range` | `[10, 30]` | 训练时视场角范围。 |
| `--default_radius` | `3.2` | 默认视角的相机半径。 |
| `--default_polar` | `90` | 默认视角的极角。 |
| `--default_azimuth` | `0` | 默认视角的方位角。 |
| `--default_fovy` | `20` | 默认视角的视场角。 |
| `--progressive_view` | 开启 | 逐步将视角采样范围从默认视角扩展到完整范围。 |
| `--progressive_view_init_ratio` | `0.2` | progressive_view 的初始范围比例。 |
| `--progressive_level` | 关闭 | 逐步提高 gridencoder 的 `max_level`。 |
| `--angle_overhead` | `30` | 顶部区域角度阈值。 |
| `--angle_front` | `60` | 前后侧区域的角度划分阈值。 |
| `--uniform_sphere_rate` | `0` | 在球面上均匀采样相机位置的概率。 |
| `--jitter_pose` | 关闭 | 对随机采样相机位姿增加扰动。 |
| `--jitter_center` | `0.2` | 相机中心扰动量。 |
| `--jitter_target` | `0.2` | 相机目标点扰动量。 |
| `--jitter_up` | `0.02` | 相机上方向扰动量。 |

### 1.10 损失与正则项

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--lambda_entropy` | `1e-3` | alpha 熵正则权重。 |
| `--lambda_opacity` | `0` | alpha 值正则权重。 |
| `--lambda_orient` | `1e-2` | 朝向正则权重。 |
| `--lambda_tv` | `0` | TV 正则权重。 |
| `--lambda_wd` | `0` | 额外损失权重。 |
| `--lambda_mesh_normal` | `0.5` | 网格法线平滑损失权重。 |
| `--lambda_mesh_laplacian` | `0.5` | 网格 Laplacian 平滑损失权重。 |
| `--lambda_guidance` | `1` | SDS guidance 损失权重。 |
| `--lambda_rgb` | `1000` | RGB 损失权重。 |
| `--lambda_mask` | `500` | mask / alpha 损失权重。 |
| `--lambda_normal` | `0` | 法线图损失权重。 |
| `--lambda_depth` | `10` | 相对深度损失权重。 |
| `--lambda_2d_normal_smooth` | `0` | 2D 法线平滑损失权重。 |
| `--lambda_3d_normal_smooth` | `0` | 3D 法线平滑损失权重。 |
| `--lambda_grid_tv_reg` | `1e-7` | 网格 TV 正则权重。 |
| `--lambda_grid_l2_reg` | `1e-7` | 网格 L2 正则权重。 |

### 1.11 调试与可视化

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--save_guidance` | 关闭 | 保存每次迭代的 NeRF 渲染、加噪、去噪等中间图，调试用，速度很慢且占显存。 |
| `--save_guidance_interval` | `10` | 每隔多少步保存一次 guidance 中间结果。 |
| `--gui` | 关闭 | 启动 GUI。 |
| `--W` | `800` | GUI 宽度。 |
| `--H` | `800` | GUI 高度。 |
| `--radius` | `5` | GUI 默认相机半径。 |
| `--fovy` | `20` | GUI 默认视场角。 |
| `--light_theta` | `60` | GUI 默认光照极角。 |
| `--light_phi` | `0` | GUI 默认光照方位角。 |
| `--max_spp` | `1` | GUI 最大每像素采样数。 |

### 1.12 参数联动说明

1. 启用 `-O` 后，会自动开启 `--fp16` 和 `--cuda_ray`。
2. 启用 `-O2` 后，会自动开启 `--fp16`，并将 `--backbone` 改为 `vanilla`，同时开启 `--progressive_level`。
3. 启用 `--IF` 后，如果 `--guidance` 中存在 `SD`，代码会自动把它替换为 `IF`，并将 `--latent_iter_ratio` 设为 `0`。
4. 只提供 `--image` 或 `--image_config` 而不提供 `--text` 时，代码会切换到 `zero123` guidance，并自动调整部分参数。
5. 同时提供图像和文本时，代码会使用 `['SD', 'clip']` 作为 guidance，并自动调整 `guidance_scale`、`t_range`、`known_view_interval` 等参数。
6. 启用 `--dmtet` 后，训练渲染分辨率会按 `--dmtet_reso_scale` 放大。

## 2. `preprocess_image.py` 参数

基本用法示例：

```bash
python preprocess_image.py data/example.png --size 256 --border_ratio 0.2
```

### 2.1 输入与输出控制

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `path` | 无 | 输入图像路径。这是位置参数，不是 `--` 选项。 |
| `--size` | `256` | 输出图像分辨率。 |
| `--border_ratio` | `0.2` | 目标图像边框占比。 |
| `--recenter` | 开启 | 对主体进行居中重排。 |
| `--dont_recenter` | 关闭 | 关闭居中重排。 |

### 2.2 预处理脚本输出内容

执行后会在输入图像所在目录生成以下文件：

| 文件 | 说明 |
| --- | --- |
| `*_rgba.png` | 抠图后的 RGBA 图像。 |
| `*_depth.png` | 深度图。 |
| `*_normal.png` | 法线图。 |
| `*_caption.txt` | 代码中预留了 caption 输出逻辑，但默认注释掉，不会生成。 |

## 3. 常见使用方式

### 3.1 文本生成 3D

```bash
python main.py --text "a toy motorcycle" --workspace trial -O --sd_version 2.1
```

### 3.2 使用 DeepFloyd-IF

```bash
python main.py --text "a rabbit, animated movie character, high detail 3d model" --workspace trial -O --IF
```

### 3.3 测试并导出网格

```bash
python main.py --workspace trial -O --test --save_mesh
```

### 3.4 图像预处理

```bash
python preprocess_image.py data/example.png --size 256 --border_ratio 0.2
```
