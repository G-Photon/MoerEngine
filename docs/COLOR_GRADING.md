# LUT 调色

Raster 的 Color Grading 在 Tonemapping 的同一次全屏绘制中执行：先计算曝光与色调映射，再在线性颜色上采样 LUT，保留现有输出路径的显示编码。

在编辑器的 `Raster Settings > Color Grading` 中使用 `Open LUT...` 选择 PNG。`Enable LUT` 控制开关，`Strength` 在原始色调映射结果与调色结果之间混合。`Clear LUT` 释放当前选择；文件读取或尺寸检查失败时，面板显示原因，渲染沿用未调色的结果。

## 图像约定

使用横向展开的三维 LUT，尺寸为 `N*N × N`，其中 `N >= 2`。例如 16 阶 LUT 为 `256 × 16`，32 阶为 `1024 × 32`。

- 每个 `N × N` 方块表示一个蓝色切片，蓝色从左向右递增。
- 切片内红色从左向右递增，绿色从上向下递增。
- RGB 值表示线性颜色。LUT 使用 UNORM 纹理读取，不进行 sRGB 解码；Alpha 不参与调色。
- 输入颜色限制在 `[0, 1]` 后查询 LUT。切片内双线性过滤，蓝色切片之间线性插值。

不支持 `.cube` 文件或方形网格布局。曝光诊断可视化保持原始数值，不参与调色。

下面的 Python 示例使用 Pillow 生成 16 阶恒等 LUT，可用于检查自己的调色流程：

```python
from PIL import Image

size = 16
image = Image.new("RGB", (size * size, size))
for green in range(size):
    for blue in range(size):
        for red in range(size):
            image.putpixel(
                (blue * size + red, green),
                tuple(round(c * 255 / (size - 1)) for c in (red, green, blue)),
            )
image.save("identity-lut.png")
```

## 资源边界

配置保存路径、开关和强度，`TonemappingPass` 持有已提交上传的纹理，帧准备参数保留本帧使用的引用。上传图以 Sampled 状态导出 LUT，帧图按这个边界导入，由 `Tonemapping.Apply` 显式声明读取；没有新增全屏 Pass 或中间颜色附件。

路径改变时在 CPU 上读取和解码文件，录制上传图后提交到图形队列。`CopyFrom` 在录制时复制像素，因此解码内存可随函数返回释放。Executor 的 FIFO 和同一图形队列的有序提交，加上上传图的 Sampled 导出屏障，保证后续帧读取到上传结果；命令持有的纹理引用保护在途资源。`FlushGPU` 请求提交，并不等待 GPU 完成，这里不新增 CPU 等待或独立的加载完成状态。文件读取、解码和录制仍会占用切换文件这一帧的 CPU 时间。

失败的路径也会记住，避免每帧重复解码；修改同一个文件后，可先清除再重新打开。

## 验证

`TestRHIParallelRecordVulkan --postprocess-regression --parallel` 包含实际 GPU 回读，检查恒等 LUT、通道变换、颜色边界、切片插值、混合强度和无 LUT 回退。

设置环境变量 `MOER_SHADER_SMOKE_LUT` 为有效 LUT 的绝对路径后运行 `TestShaderEffectsSmoke`，还会在场景渲染中检查加载、禁用、强度调整、缺失文件、清除与重新加载。两项测试均需要 Vulkan GPU 与运行时着色器编译环境，场景测试还需要根配置中指定的场景资源。

可额外设置 `MOER_SHADER_SMOKE_LUT_REJECTED` 为损坏或尺寸不匹配的图像路径，用同一流程检查加载失败后的回退与恢复。
