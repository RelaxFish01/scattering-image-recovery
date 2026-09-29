# Scattering Image Recovery

本仓库给出《散射图像恢复算法研究》作业的完整题解、可复现实验脚本、模型文件与结果记录。仓库面向课程组员及后续研究工作，教程、环境试跑记录和与本题无关的材料不纳入版本库。

## 摘要

给定 112×112 的散斑强度观测 `Y`，目标是恢复 28×28 的原始图像 `X`。在全局岭回归和残差 CNN 对照实验的基础上，本题解利用训练对中的强度关系标定复传输矩阵，并在图像域中求解振幅最小二乘问题。最终模型使用 3,439 个通过独立验证筛选的探测通道，其中 2,927 个用于反演，另 512 个仅用于事后测量一致性核验。

`test.mat` 仅包含四幅 test 的散斑观测，不包含对应的 `X` 真值。因此，仓库对四幅 test 只报告独立测量一致性，不报告未经提供的真值 PSNR/SSIM。

## 结果概览

固定划分为 40,000 / 5,000 / 5,000 的训练、验证和保留集。最终物理反演在 256 幅独立保留图像上得到：

| 方法 | 保留集数量 | 平均 PSNR | 平均 SSIM |
| --- | ---: | ---: | ---: |
| 全局岭回归 | 5,000 | 32.36 dB | 0.6913 |
| 岭回归 + 强度收缩 | 5,000 | 34.74 dB | 0.9826 |
| 岭回归 + 残差 CNN | 5,000 | 37.02 dB | 0.9937 |
| 复传输矩阵物理反演 | 256 | 139.87 dB | 1.0000 |

四幅 test 的最终输出依次呈现为 6、5、3、8。由于没有 test 真值，这一表述仅指恢复图像的形态判读，不是带标签的识别结论。512 个未参与初始化、归一化和反演的测量通道上的相对 MSE 分别为 `1.35e-12`、`7.62e-13`、`6.26e-13` 和 `7.60e-13`。

## 目录

```text
data/raw/                 题目提供的 test.mat；训练原始文件不入库
data/processed/           由 prepare_data.py 生成的本地缓存目录
src/                      数据准备、基线、物理标定与反演脚本
models/                   交付模型和初始化模型
artifacts/                标定、划分、审计与环境记录
results/                  test 输出和独立保留集核验结果
docs/                     题目定义、方法和验证说明
```

## 快速复现四幅 test

在仓库根目录执行：

```powershell
python src/physical_recovery/recover.py `
  --input data/raw/test.mat `
  --output results/test/recovered
```

该命令直接读取 `test.mat`，使用仓库中的 `models/physical/inversion_model.npz` 和 `models/ridge12544/model.npz`，输出浮点 NPY、MAT、PNG 预览及诊断 JSON。已提交的 `results/test/X_recovered.npy` 与 `X_recovered.mat` 是同一套最终交付结果。

## 从原始训练数据重做实验

训练文件 `Train.mat` 约 4.8 GB，不适合直接存入普通 GitHub 仓库。请将它放入 `data/raw/Train.mat`，再执行：

```powershell
python src/prepare_data.py
python src/baselines/ridge_baseline.py --out-dir models/ridge12544 --features 12544
python src/physical_recovery/calibrate.py --detectors 4096 --batch 256 --train 10000
python src/physical_recovery/invert.py --threshold 1e-11 --holdout 256 --steps 500
```

`prepare_data.py` 不修改原始 MAT 文件；它只生成 `data/processed/X.npy`、`Y.npy`、`test_Y.npy` 和数据报告。重新标定需要可用 CUDA 的 PyTorch；仅执行已保存模型的 test 推理不需要训练数据缓存。

## 复现边界与数据说明

- 原始数据中 `X` 有 52,000 个样本，`Y` 有 50,000 个样本；未有索引文档证明多出的 2,000 个 `X` 与 test 对应，因此它们未用于训练或真值替代。
- test 没有公开原始图像。候选检索审计未发现可据此认定为 test 真值的训练图像；候选图像不参与最终输出。
- 保留集指标只说明当前数据及其测量关系上的可逆性，不等同于对新散射介质、噪声观测或域外图像的泛化保证。
- PNG 仅用于显示；数值结果以 NPY/MAT 文件为准。MAT 使用 `28×28×4` 的 `X_recovered`，NPY 使用 `4×28×28`。

详细方法见 [`docs/method.md`](docs/method.md)，验证边界见 [`docs/validation.md`](docs/validation.md)。

## 引用与许可

引用信息见 [`CITATION.cff`](CITATION.cff)。代码采用 MIT 许可；题目数据的使用与再分发应遵循数据提供方的课程要求。
