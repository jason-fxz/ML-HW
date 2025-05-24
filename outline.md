
---

## 论文标题（建议）

**Crossformer 在数据漂移下的鲁棒性提升：结合 RevIN 与周周期建模的实证研究**

---

## 摘要（Abstract）

时间序列预测模型在实际应用中常常面临数据分布随时间发生漂移的问题，这一现象在 ETT 和 ECL 等数据集中尤为明显。尽管 Crossformer 模型在多变量预测中表现优异，但在存在分布偏移的测试数据上，其预测结果常出现系统性偏差。本文在 Crossformer 框架中引入 Reversible Instance Normalization（RevIN），以提升模型对分布漂移的适应性，并进一步提出基于统计周期提取的 weekly pattern 消除机制，通过显式提取并减除周期性结构，缓解模型对长期趋势的误判。实验证明，该方法有效提升了 Crossformer 在漂移数据上的预测精度。

---

## 1. 引言（Introduction）

* 介绍时序数据预测中的分布漂移问题，强调在 ETT/ECL 数据集中，训练集与测试集的分布差异显著。
* 指出 Crossformer 在原始设置中存在系统性预测偏差。
* 提出本文两个主要改进方向：

  1. 在 Crossformer 中引入 RevIN；
  2. 基于 weekly pattern 的周期结构提取与残差建模方法。
* 简述实验结果：在多个指标上取得一致改进。

---

## 2. 数据分析与问题动机（Motivation）

* **ETT/ECL 分布漂移现象**

  * 训练集通常仅覆盖一年数据，预测跨季度；
  * 测试集集中在特定四个月，分布不同。
  * \<fig: ETT ECL数据分布> 展示训练/测试集在不同时间段的统计差异；
  * 模型预测中出现系统性偏差，可视化对比 \<fig: Crossformer预测偏差示意图>。
---

## 3. 相关工作（Related Work）

### 3.1 Crossformer

* Two-Stage Attention 结构
* patch 化时间序列建模
* 局限：缺乏对周期结构和漂移数据的鲁棒建模

### 3.2 RevIN

* 提出背景：处理分布漂移问题；
* 结构：可逆归一化 + 可学习仿射变换；
* 优势：不会影响模型结构但可显著提升稳健性。

---

## 4. 方法（Methodology）

### 4.1 RevIN 在 Crossformer 中的整合

* 对输入数据的每个变量通道进行归一化
* 保留 mean/std，用于预测后反归一化
* 不改变原有 Crossformer 架构，仅增强其稳健性

### 4.2 周周期特征提取与残差建模

* 周周期提取方法：

  * 滑动一周窗口求平均；
  * 对应时刻求平均值（同星期同时间点）；
  * mean-std 归一化；
  * 高斯平滑；
* 模型输入调整：

  * 构建 pattern\_in, pattern\_out；
  * 输入变换为：`in_data - pattern_in`；
  * 输出监督目标变为：`out_data - pattern_out`；
  * 保留每次训练样本自己的 `in_mean, in_std` 用于 pattern 缩放。
* 对齐与扩展策略确保时间点对应准确。

---

## 5. 实验设计与结果分析（Experiments）

### 5.1 实验设置

* 数据集：ETTm1、ECL
* 比较模型：

  * 原始 Crossformer；
  * Crossformer + RevIN；
  * Crossformer + weekly pattern；
* 评估指标：MSE, MAE

### 5.2 实验结果

* \<fig: MAE RMSE 表格比较图>
* \<fig: 各模型预测曲线可视化>
* RevIN 有效减小漂移引起的误差；
* weekly pattern 显著提高模型对周期性结构建模的能力；
* 两者结合下误差最小、偏差最小。

---

## 6. 结论与未来工作（Conclusion & Future Work）

* 提出 Crossformer 在应对分布漂移与周期建模上的两个改进方向；
* 未来可探索：
  * 动态学习周期结构（如使用周期注意力或频域建模）；
  * 多层级周期性建模（如每日、每周、每年）；

