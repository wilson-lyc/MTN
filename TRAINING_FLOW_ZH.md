# 文本到 3D 的训练流程说明

本文把这个项目里的训练链路对应到代码，回答两个核心问题：

1. `NeRF / DMTet` 在哪里负责 3D 表示与渲染。
2. `Stable Diffusion` 在哪里作为 2D 先验提供优化信号。

一句话先说结论：

- 这个项目里，被优化的主体是 `NeRFNetwork` 以及可选的 `DMTet` 几何参数。
- `Stable Diffusion` 不直接生成 3D，而是对“当前 3D 渲染出的 2D 图像”提供 SDS 梯度。
- 梯度经由渲染链路反传，最终更新 `NeRF / DMTet` 的参数。

## 1. 总体结构

从代码结构看，训练主链路是：

`main.py` -> `Trainer` -> `model.render(...)` -> `guidance['SD'].train_step(...)` -> `loss.backward()` -> `optimizer.step()`

对应关系如下：

- `main.py`：解析参数，构建 `NeRFNetwork`、`guidance`、`optimizer`、`Trainer`。
- `nerf/utils.py` 中的 `Trainer.train_step()`：采样相机，调用 3D 模型渲染图像，组装各种 loss。
- `guidance/sd_utils.py`：把渲染结果送进 Stable Diffusion 的 VAE / UNet，构造 SDS 梯度。
- `optimizer.py`：提供 `Adan` 优化器；实际参数更新也可以走 `Adam`。

## 2. 入口：`main.py` 如何把训练对象接起来

### 2.1 创建 3D 模型

在 `main.py` 中，真正被训练的模型是 `NeRFNetwork(opt)`：

- 普通模式下，这就是 NeRF 风格的隐式 3D 场表示。
- 如果开启 `--dmtet`，底层渲染会切到 DMTet 网格分支。

关键位置：

- 创建模型：`model = NeRFNetwork(opt).to(device)`
- 如果开启 `--dmtet` 且提供 `--init_with`，会先加载已有 NeRF 权重或 mesh，再调用 `model.init_tet()` 初始化四面体网格。

### 2.2 创建 guidance 模型

在训练分支里，`main.py` 会按 `opt.guidance` 构造一个 `nn.ModuleDict()`：

- `guidance['SD'] = StableDiffusion(...)`
- 也可能同时挂上 `IF`、`zero123`、`clip`

如果你当前讨论的是文本驱动 3D，重点是：

- `SD` 是 guidance 模型，不是最终 3D 表示。
- 它只在训练时参与，不是最后导出的 3D 资产本身。

### 2.3 创建优化器

`main.py` 根据参数选择：

- `Adan(model.get_params(...))`
- 或 `torch.optim.Adam(model.get_params(...))`

这里 `model.get_params(...)` 很关键，因为它决定了谁会被更新。

以 `nerf/network_grid.py` 为例：

- 总会优化编码器和 `sigma_net`
- 若启用背景网络，也会优化 `bg_net`
- 若开启 `dmtet` 且没有 `lock_geo`，还会把 `self.sdf` 和 `self.deform` 加进参数组

这说明：

- NeRF 阶段主要优化隐式场参数。
- DMTet 阶段除了材质查询网络，也会优化显式几何参数 `sdf` / `deform`。

## 3. `Trainer` 如何准备“老师”和“学生”

`Trainer` 在初始化时做两件关键的事。

### 3.1 冻结 guidance 参数

在 `Trainer.__init__()` 中：

- 对 `self.guidance[key].parameters()` 统一设置 `requires_grad = False`

这一步非常重要，表示：

- Stable Diffusion 作为老师参与前向推断，提供梯度方向。
- 但 SD 自己不训练，不会被优化器更新。

也就是说，学生是 3D 模型，老师是冻结的扩散模型。

### 3.2 预先编码文本嵌入

`Trainer.prepare_embeddings()` 会为文本 prompt 计算文本特征：

- `default`：主文本 prompt
- `uncond`：空/负面 prompt
- `front` / `side` / `back`：带视角描述的 prompt

对于 `SD`，对应调用是：

- `self.guidance['SD'].get_text_embeds([self.opt.text])`
- `self.guidance['SD'].get_text_embeds([self.opt.negative])`
- `self.guidance['SD'].get_text_embeds([f"{self.opt.text}, front view"])` 等

这说明训练不是把“同一句文本”机械地喂给每个视角，而是根据相机方位，在 front/side/back 文本嵌入之间插值，尽量减少 Janus 问题。

## 4. 训练一步里到底发生了什么

核心函数是 `nerf/utils.py` 里的 `Trainer.train_step()`。

可以把一次训练迭代理解成下面 6 步。

### 4.1 采样相机与渲染设置

`data` 里已经包含当前 batch 的：

- `rays_o`, `rays_d`
- `mvp`
- `H`, `W`
- 以及 `azimuth`、`polar`、`radius` 等视角信息

然后代码会根据训练阶段选择渲染模式：

- 早期可能 `as_latent = True`，`shading = 'normal'`
- 后期一般 `as_latent = False`，在 `lambertian` / `textureless` / `albedo` 等模式中切换
- 如果是图像条件分支，还会定期切到固定视角做 RGBD 监督

### 4.2 调用 3D 模型渲染 2D 图像

这一句是核心：

```python
outputs = self.model.render(...)
```

这里的 `self.model` 就是 `NeRFNetwork`，而它继承自 `NeRFRenderer`。

在 `nerf/renderer.py` 中：

- 如果 `self.dmtet` 为真，就走 `run_dmtet(...)`
- 否则走 NeRF 体渲染分支

所以这里可以更准确地说：

- NeRF/DMTet 负责生成当前视角下的渲染结果。
- 后续所有 guidance 都是基于这个 2D 渲染结果施加的。

### 4.3 从渲染结果整理出 `pred_rgb`

`Trainer.train_step()` 里会把渲染输出整理成：

- `pred_rgb`: `[B, 3, H, W]`，常规 RGB 图像
- 或在 `as_latent=True` 时，拼成 `[B, 4, H, W]`

这一步的注释已经写得很直白：

- 在 latent warmup 阶段，会“借用 normal 和 mask 作为 latent code”，做更快的几何初始化。

也就是说，送给 SD 的不一定永远是最终 RGB；训练早期可能送的是一种近似 latent 表示。

### 4.4 根据视角构造文本条件

如果启用了 `SD` guidance，`Trainer.train_step()` 会取当前视角的 `azimuth`，在这些文本嵌入之间插值：

- `front`
- `side`
- `back`
- 再拼上 `uncond`

最后得到 `text_z`，传给 `guidance['SD'].train_step(...)`。

因此，SD 在这里评估的不是“抽象的 3D 质量”，而是：

- “当前这个视角渲染出来的 2D 图像，是否符合该视角下的文本语义”。

### 4.5 Stable Diffusion 产生 SDS 梯度

真正的 SD guidance 发生在 `guidance/sd_utils.py` 的 `StableDiffusion.train_step()`。

它的流程是：

1. 如果 `as_latent=False`，先把渲染图上采样到 `512x512`
2. 用 VAE encoder 把图像编码到 latent 空间
3. 随机采样时间步 `t`
4. 给 latent 加噪
5. 用冻结的 UNet 预测噪声残差
6. 做 classifier-free guidance：
   `noise_pred_uncond + scale * (noise_pred_pos - noise_pred_uncond)`
7. 根据 `w(t) * (noise_pred - noise)` 构造梯度
8. 用 `SpecifyGradient.apply(latents, grad)` 把这个梯度“挂”回计算图

这里要强调两点：

- SD 不是简单输出一个分数。
- 它通过去噪网络的预测结果构造一个梯度方向，这就是常说的 SDS 风格优化。

所以更准确的表述是：

- SD 作为冻结的 2D 扩散先验，为当前渲染结果提供优化方向。
- 梯度通过 VAE 编码前的图像、再通过渲染过程，最终回传到 3D 参数。

### 4.6 反向传播并更新 3D 参数

训练循环里，真正执行更新的是：

```python
self.scaler.scale(loss).backward()
self.post_train_step()
self.scaler.step(self.optimizer)
self.scaler.update()
```

这几句出现在 `Trainer.train_gui()` 和常规训练循环中。

含义是：

- `loss.backward()`：把 guidance 和其他正则项的梯度反传回 3D 模型
- `post_train_step()`：做梯度裁剪、grid 正则等后处理
- `optimizer.step()`：更新 NeRF 或 DMTet 参数

因此，最终被更新的是：

- NeRF 编码器/MLP
- 可能的背景网络
- 若启用 DMTet，则还包括 `sdf` 与 `deform`

而不是 Stable Diffusion 本身。

## 5. DMTet 在这条链里扮演什么角色

如果不开 `--dmtet`：

- `render()` 走的是 NeRF 体渲染路径
- 3D 表示是隐式体场

如果开了 `--dmtet`：

- `render()` 会切到 `run_dmtet()`
- `run_dmtet()` 会用四面体网格、`sdf`、`deform` 提取显式表面，再做可微栅格化

对应代码逻辑是：

- 初始化时加载 `tets/<grid>_tets.npz`
- 参数化几何：`self.sdf`、`self.deform`
- `run_dmtet()` 中通过 `self.dmtet(self.verts + deform, sdf, self.indices)` 生成 mesh
- 然后做 rasterization，得到颜色、alpha、normal 等输出

这说明 DMTet 并没有改变“SD 当老师”的大逻辑，它只是把学生从“隐式体渲染表示”部分地换成了“显式网格几何 + 神经纹理查询”。

也就是说：

- NeRF 阶段和 DMTet 阶段，都可以接受 SD guidance。
- 区别主要在于 3D 表示和渲染方式不同。

## 6. 这个项目里的 loss 不是只有 SD

虽然你关心的是“SD 作为老师”，但代码里实际优化目标是多个项的和。

### 6.1 Novel-view 主要是 guidance loss

在没有图像监督时，主要是：

- `SD` guidance loss
- 可选 `IF` / `zero123` / `clip`

### 6.2 还有一批正则项

在 `Trainer.train_step()` 里还能看到：

- `lambda_opacity`
- `lambda_entropy`
- `lambda_orient`
- `lambda_2d_normal_smooth`
- `lambda_3d_normal_smooth`
- `lambda_grid_tv_reg`
- `lambda_grid_l2_reg`

如果是 DMTet 分支，还会加：

- `lambda_mesh_normal * outputs['normal_loss']`
- `lambda_mesh_laplacian * outputs['lap_loss']`

所以更完整的理解应该是：

- SD 提供主要的语义驱动信号。
- 其他 loss 负责让几何、透明度、法线、网格光滑性更稳定。

## 7. 把你的原始理解改成更准确的一句话

你原来的理解：

> NeRF/DMTet 用于 3D 生成，SD 作为老师评估生成质量来优化 NeRF/DMTet 的参数。

可以改成下面这个版本，更贴近代码：

> 在这个项目中，NeRF 或 DMTet 是被优化的 3D 表示；训练时先从当前 3D 表示渲染出 2D 图像，再把该图像送入冻结的 Stable Diffusion，利用其去噪网络构造 SDS 梯度；这个梯度通过可微渲染链路反传，最终更新 NeRF/DMTet 的参数。

如果再口语一点，可以说：

> NeRF/DMTet 负责“做出一个 3D 并渲染给老师看”，SD 负责“根据文本告诉它这个视角看起来该往哪个方向改”，然后优化器据此更新 3D 参数。

## 8. 最简调用链总结

下面这条链可以当成你读代码时的索引：

1. `main.py`
   创建 `model`、`guidance['SD']`、`optimizer`、`Trainer`
2. `Trainer.prepare_embeddings()`
   预先算好 `text` / `uncond` / `front-side-back` 文本嵌入
3. `Trainer.train_step(data)`
   根据当前相机视角，调用 `self.model.render(...)` 得到 `pred_rgb`
4. `guidance['SD'].train_step(text_z, pred_rgb, ...)`
   将渲染图编码到 latent，随机加噪，用 UNet 预测噪声并构造 SDS 梯度
5. `loss.backward()`
   梯度通过渲染过程反传到 3D 表示
6. `optimizer.step()`
   更新 NeRF 参数，或更新 DMTet 的 `sdf/deform` 等参数

## 9. 对应文件索引

建议按下面顺序读源码：

- `main.py`：训练入口和组件装配
- `nerf/utils.py`：`Trainer`、训练一步、loss 组装
- `guidance/sd_utils.py`：Stable Diffusion guidance / SDS 细节
- `nerf/renderer.py`：NeRF 与 DMTet 两种渲染路径
- `nerf/network_grid.py`：NeRF 网络参数、`get_params()` 定义
- `optimizer.py`：`Adan` 优化器实现

