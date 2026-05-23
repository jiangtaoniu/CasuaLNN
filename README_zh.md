# CasuaLNN

[English](README.md)

CasuaLNN 是一个针对时间序列前沿问题架构的代码仓库，全面聚焦于长期预测（Long-term forecasting）、短期预测（Short-term forecasting）、异常检测（Anomaly Detection）以及数据插补（Imputation）。本仓库复现并在多个领域的真实世界业务场景中（交通、天气、电力、能源等）验证了最先进的预估模型。

## 概览

该项目基于统一而健壮的执行框架在多个通用的学术界基准测试数据集上进行了实验（包括 ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic, Weather, PEMS03, PEMS08）。

### 核心特性
- **高度统一的执行入口 (`run.py`)**: 摒弃了早期针对每个数据单独开设一个 `.py` 文件的形式。如今所有数据集任务依靠唯一的动态参数驱动脚本启动，大幅降低了实验代码的冗余，也便于迭代与消融实验。
- **批量化的实验执行脚本**: 提供了配置完整的 shell 脚本，可以自动化测试各个数据集在不同预测长度下的表现 (`96`, `192`, `336`, `720`)。
- **先进的预测组件**: 内部解耦并集成了优秀的时空序模块以及深层次嵌入模块。

## 模型架构

<p align="center">
  <img src="figures/model_architecture.png" width="800"/>
</p>
<p align="center">
  <i>CasuaLNN 整体模型架构图，展示了从输入嵌入到因果时序处理模块再到最终预测输出的完整数据流。</i>
</p>

## 快速上手

### 1. 安装环境依赖

建议使用专门的虚拟环境运行：

```bash
pip install -r requirements.txt
```

### 2. 数据集准备

在运行之前，确保你需要测试的数据集分别置于 `./dataset/ETT`, `./dataset/electricity`, `./dataset/traffic`, `./dataset/weather`, or `./dataset/PEMS` 对应的目录架构之下。

### 3. 一键运行实验

如果你想要重现我们配置好的所有实验组合，可直接执行根目录提供的批量测试脚本：

```bash
bash scripts/run.sh
```

此外，当你需要单独测试某个超链接参数时，可以通过传递不同的可选参数来调用 `run.py` 以进行定制化预测：

```bash
python -u run.py \
    --data ETTh1 \
    --root_path ./dataset/ETT/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_96 \
    --pred_len 96 \
    --itr 1 \
    --decomp_method dft_decomp
```

## 抽象参数设定
在 `run.py` 内主要包括如下几类参数可供您微调测试：
- `--model_id`: 针对当前独立实验下发的一个自定义标记
- `--pred_len`: 预期要向未来预测的时间步长
- `--seq_len`: 作为编码器输入的序列长度
- `--label_len`: 解码器开始计算任务前的真实观测标记长度
- `--d_model` / `--d_ff`: 各个网络中间层的前馈隐藏维度
- `--decomp_method`: 不同数值序列的分解方式的切换

## 证书
MIT License.
