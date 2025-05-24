# ML-HW

我们主要发现了ETT，ECL 数据在长时间尺度上存在分布漂移，虽然这一部分漂移大致符合以年为周期的规律，但是训练数据通常只有一年，因此在进行长时间预测时，模型可能无法很好地捕捉到这种周期性变化。并且测试集集中在一年中的四个月份，这个部分的数据分布和训练集的分布差异较大，导致模型在这一部分的预测效果较差。根据对测试集的可视化可以看到，模型在预测时会存在较大的系统偏差。 <fig: ETT ECL数据分布>


related work crossformer:(CROSSFORMER: TRANSFORMER UTILIZING CROSSDIMENSION DEPENDENCY FOR MULTIVARIATE TIME  SERIES FORECASTING) 通过可视化，我们发现 crossformer 模型，在进行长时间预测时，模型的预测结果会出现较大的系统偏差，尤其是在数据分布发生漂移的情况下。我们希望针对这一问题进行改进。

相关工作： RevIN（REVERSIBLE INSTANCE NORMALIZATION FOR  ACCURATE TIME-SERIES FORECASTING AGAINST  DISTRIBUTION SHIFT） a generallyapplicable normalization-and-denormalization method with learnable affine transformation. 以此极大提升了模型在存在数据漂移的情况下的预测精度。

我们在 Crossformer 中引入了 RevIN，通过实验验证了 RevIN 对与 Crossformer 在处理数据分布漂移时的有效性。

进一步，我们提出了 weekly pattern 的特征提取访问，用统计方法提取数据的周期性特征，让输入减去周期性特征，让 crossformer 模型学习其残差。

通过对 ETT, ECL 数据的特征分析，我们发现其数据具有明显的周期特征：较短的每日周期和较长的每周周。

<fig: ETT ECL 周平均趋势> 分析方法：以一周为长度的滑动窗口计算周平均值
<fig: ETT ECL 每周周期特征图> 每周对应时刻的平均值，为了减少噪声，我们做出了一些处理：

- 考虑到数据存在漂移（不同周整体的平均值不同，方差可能较大），我们对每周的平均值进行了（mean - std）标准化处理
- 考虑到数据存在噪声，得到的 pattern 可能会包含一些高频信息，使用高斯滤波器进行平滑处理

<某个维度数据 pattern 处理示意，1.直接 week 平均 2. +每周归一化 3. +每周归一化+高斯滤波>
最终得到对应的 pattern 图像如上图所示。

在训练数据上对所有维度建立 pattern，在训练前对输入数据减去对应的 pattern，得到残差数据作为模型的输入。参考 RevIN 的经验，为了让模型免于系统偏差的影响，对于 (in_data, out_data) 的数据，我们求出 in_std, in_mean，将pattern 做出对应变换： pattern * in_std + in_mean，再将 pattern 对应周与 in_data，out_data 对齐扩展得到 pattern_in, pattern_out，将 (in_data-pattern_in, out_data-pattern_out) 作为模型的输入输出训练。

实验结果：在 ETT, ECL 数据集上，我们跑了 Crossformer, Crossformer + RevIN, Crossformer + weekly pattern 三个模型，结果如下：


## ETTh1


### 720

- baseline
  mse: 0.521  mae: 0.523
- revin
  mse: 0.475  mae: 0.482
- ours
  mse: 0.460  mae: 0.481
  
### 336

- baseline
  mse: 0.440  mae: 0.461
- revin
  mse: 0.429  mae: 0.446
- ours
  mse: 0.426  mae: 0.445

### 168

- baseline
  mse: 0.409  mae: 0.443
- revin
  mse: 0.397  mae: 0.422
- ours
  mse: 0.402  mae: 0.427

## ECL

### 336

- baseline
  mse: 0.323  mae: 0.369
- revin
  mse: 0.264  mae: 0.331
- ours
  mse: 0.243  mae: 0.320

### 720

- baseline
  mse: 0.404  mae: 0.423
- revin
  mse: 0.319, mae: 0.376
- ours
  mse: 0.278, mae: 0.350

### 960

- baseline
  mse: 0.433 mae: 0.438
- revin
  mse: 0.343 mae: 0.397
- ours
  mse: 0.304 mae: 0.365

