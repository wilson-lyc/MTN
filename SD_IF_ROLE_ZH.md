# SD 与 IF 模型在本项目中的作用

本文档结合项目代码，说明 Stable Diffusion（SD）与 DeepFloyd IF（IF）在 MTN 中的功能定位、调用方式与实现差异，便于写入论文的方法介绍或实现细节部分。

## 1. 总体定位

在这个项目里，`SD` 和 `IF` 都不是最终负责表示三维场景的主模型。真正学习三维几何与外观的是 NeRF 主干网络，而 `SD/IF` 的作用是作为**二维扩散先验（2D diffusion prior）**，在训练阶段对 NeRF 的渲染结果提供文本驱动的监督信号。

更具体地说，训练流程是：

1. NeRF 从随机视角渲染出当前的二维图像。
2. 将该图像送入 `SD` 或 `IF` guidance 模块。
3. guidance 模块依据文本提示词预测噪声，并构造 SDS 风格的梯度。
4. 梯度不用于更新扩散模型本身，而是反向传回 NeRF，使三维表示逐渐生成与文本语义一致的物体。

因此，`SD` 和 `IF` 在本项目中的核心作用可以概括为一句话：**它们充当文本到图像的先验约束，把二维生成模型的语义能力迁移到三维生成过程中。**

## 2. 在代码流程中的位置

### 2.1 guidance 模型的选择

主入口在 [main.py](/Users/wilson/Projects/MTN/main.py)。项目默认使用 `SD`：

- `--guidance` 的默认值是 `['SD']`
- 开启 `--IF` 后，如果 guidance 中存在 `SD`，代码会自动替换为 `IF`
- 同时，`--IF` 会把 `latent_iter_ratio` 设为 `0`

这说明项目把 `IF` 视为 `SD` 的一种替代型文本 guidance，而不是并行的主生成器。

实际加载发生在 `main.py` 的 guidance 初始化部分：

- `StableDiffusion` 定义在 [guidance/sd_utils.py](/Users/wilson/Projects/MTN/guidance/sd_utils.py)
- `IF` 定义在 [guidance/if_utils.py](/Users/wilson/Projects/MTN/guidance/if_utils.py)

### 2.2 在 Trainer 中的调用方式

训练器定义在 [nerf/utils.py](/Users/wilson/Projects/MTN/nerf/utils.py)。

训练开始前，`prepare_embeddings()` 会为 `SD` 或 `IF` 预编码文本嵌入，包括：

- 默认提示词 `text`
- 无条件提示词 `negative`
- `front / side / back view` 三个视角相关提示词

训练时，`train_step()` 先让 NeRF 从当前相机位姿渲染图像，然后根据相机方位角 `azimuth` 在 `front / side / back` 三组文本嵌入之间插值，生成当前视角对应的文本条件。之后：

- 若启用 `SD`，调用 `self.guidance['SD'].train_step(...)`
- 若启用 `IF`，调用 `self.guidance['IF'].train_step(...)`

可见，`SD` 与 `IF` 在系统结构中承担的是同一类职责：**对当前渲染视图施加文本一致性约束**。

## 3. SD 的具体作用

### 3.1 作为默认的文本 guidance 模型

`SD` 是项目中的默认 guidance 模块，也是最基础、最常规的文本到 3D 监督来源。代码支持 `1.5`、`2.0`、`2.1` 三个版本，并通过 `--sd_version` 选择。

在 [guidance/sd_utils.py](/Users/wilson/Projects/MTN/guidance/sd_utils.py) 中，`StableDiffusion` 主要包含四个组成部分：

- `tokenizer`
- `text_encoder`
- `VAE`
- `UNet`

它的作用流程是：

1. 把文本提示词编码为 text embedding。
2. 把 NeRF 渲染图像上采样到 `512x512`。
3. 通过 VAE 编码到 latent 空间。
4. 在随机时间步加噪。
5. 用 UNet 预测噪声，并结合 classifier-free guidance 得到条件噪声估计。
6. 构造 SDS 梯度并回传给 NeRF。

因此，SD 在本项目中相当于一个**工作在 latent space 的语义判别器/先验约束器**。

### 3.2 促进几何与外观共同收敛

SD 不直接输出三维网格或 NeRF 参数，而是通过对每个渲染视图施加“应当更像文本描述图像”的梯度，引导 NeRF 同时优化：

- 几何形状
- 纹理与颜色
- 视角相关的外观合理性

尤其在项目早期训练中，代码支持 `as_latent=True` 的阶段：此时渲染结果会被拼成 4 通道 latent 形式，用于更快地初始化几何。这一策略只对 SD 有效，因为 SD 本身就是基于 latent diffusion 实现的。

## 4. IF 的具体作用

### 4.1 作为 SD 的替代文本 guidance

`IF` 在项目中被标注为 experimental，但它的定位非常清晰：**用 DeepFloyd IF 替代 Stable Diffusion，给 NeRF 提供另一种更强或不同分布的二维扩散先验。**

在 [guidance/if_utils.py](/Users/wilson/Projects/MTN/guidance/if_utils.py) 中，`IF` 使用的是 `DeepFloyd/IF-I-XL-v1.0`。与 SD 不同，它不通过 VAE 进入 latent 空间，而是直接在图像空间附近工作：

1. 将 NeRF 渲染结果缩放到 `64x64`
2. 将图像从 `[0,1]` 映射到 `[-1,1]`
3. 在随机时间步加噪
4. 用 IF 的 UNet 预测噪声
5. 构造 guidance 梯度并反向传播到 NeRF

因此，IF 在这个项目里可以理解为**图像空间版本的扩散 guidance 模块**。

### 4.2 用于替换 SD 的一个重要原因

从实现意图看，作者希望 IF 提供与 SD 不同的生成先验，从而改善文本驱动 3D 生成的效果，例如：

- 提升语义一致性
- 改善某些复杂提示词的成像质量
- 配合 `perpneg` 缓解多视角不一致或 Janus 问题

README 中也专门给出了多条 `--IF` 的训练命令，说明 IF 不是辅助组件，而是一个完整可选的主 guidance 路线。

## 5. SD 与 IF 的共同点

二者在项目中的共同角色主要有三点：

- 都是训练阶段使用的 guidance 模块，而不是三维表示本体。
- 都通过文本嵌入、加噪、噪声预测与 SDS 梯度，把二维扩散模型的知识转移给 NeRF。
- 都支持视角相关文本条件，即根据 `front / side / back` 的嵌入插值来提升多视角一致性。

从论文写法上，可以把二者统一概括为：**文本条件扩散先验模块**。

## 6. SD 与 IF 的关键差异

两者最重要的区别在于监督施加的空间不同：

- `SD`：先将图像编码到 VAE latent，再在 latent 空间做扩散 guidance。
- `IF`：直接对低分辨率图像做扩散 guidance，不经过 VAE latent。

这带来几个实现层面的差异：

- `SD` 支持 `as_latent` 训练阶段，因此项目前期可用 latent warm-up 加速几何初始化。
- `IF` 开启后会强制 `latent_iter_ratio = 0`，说明该路径不兼容基于 latent 的早期训练策略。
- `SD` 的输入渲染会先插值到 `512x512` 再经 VAE 编码；`IF` 则直接使用 `64x64` 图像。
- `IF` 的代码中包含对时间步采样的额外设计，尤其在 `perpneg` 分支下有非线性时间步选择策略，说明作者对 IF 的训练稳定性做了额外调节。

如果写成论文里的对比句，可以概括为：**SD 提供 latent-space diffusion guidance，IF 提供 image-space diffusion guidance。**

## 7. 与视角一致性的关系

本项目并不是简单把同一段文本重复喂给扩散模型，而是根据当前相机方位角动态生成文本条件：

- 正面视角更接近 `front view`
- 侧面视角更接近 `side view`
- 背面视角更接近 `back view`

这一机制在 [nerf/utils.py](/Users/wilson/Projects/MTN/nerf/utils.py) 中同时服务于 `SD` 和 `IF`。它的作用是让扩散 guidance 不仅提供“物体是什么”的语义约束，还提供“从当前视角看应该长什么样”的弱几何约束，从而减轻多视图不一致问题。

## 8. 在整个项目中的边界

需要强调的是，`SD` 与 `IF` 只在**文本驱动的训练监督**中发挥作用。它们并不负责：

- NeRF 的体渲染
- 三维密度场或颜色场建模
- 网格提取与导出

这些任务由 NeRF 主网络、渲染器以及后续的 mesh 导出流程完成。换言之，`SD/IF` 决定的是“训练信号来自哪里”，而 NeRF 决定的是“3D 结果如何被表示出来”。

## 9. 可直接用于论文的总结表述

可以将本项目中 `SD` 与 `IF` 的作用概括为：

> 本项目将 Stable Diffusion 或 DeepFloyd IF 作为二维扩散先验引入 NeRF 优化过程。具体而言，NeRF 从随机视角渲染二维图像，扩散模型根据文本提示词对渲染结果施加 Score Distillation Sampling（SDS）式语义梯度，从而引导三维几何与外观逐步收敛到与文本描述一致的结果。其中，Stable Diffusion 在 latent 空间中提供 guidance，而 DeepFloyd IF 直接在图像空间提供 guidance，二者共同承担文本到三维生成中的先验约束作用。 

## 10. 代码依据

本文档主要依据以下实现文件整理：

- [main.py](/Users/wilson/Projects/MTN/main.py)
- [guidance/sd_utils.py](/Users/wilson/Projects/MTN/guidance/sd_utils.py)
- [guidance/if_utils.py](/Users/wilson/Projects/MTN/guidance/if_utils.py)
- [nerf/utils.py](/Users/wilson/Projects/MTN/nerf/utils.py)
