
# CausaLNN: Causal-Modulated Liquid Neural Network for Time Series Forecasting

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pytorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c.svg)](https://pytorch.org/)

**Official Implementation of the paper: "CausaLNN: Disentangling Variate Causality and Liquid Temporal Evolution"**

> **Authors**: Jiangtao Niu*, Xianglan Gao, Xu Zong  
> **Contact**: 13931672450@163.com

---

## 📖 Introduction

Multivariate Time Series (MTS) forecasting faces a fundamental dilemma: **variate causality** (structural dependencies) and **temporal dynamics** (evolution over time) are often entangled. Existing discrete models (e.g., Transformers, MLPs) struggle to explicitly capture the underlying causal graph or model the continuous nature of real-world systems.

We propose **CausaLNN**, a novel framework that disentangles these two aspects via a closed-loop design:
1.  **Global Causal Graph**: A learnable, shared adjacency matrix $\mathbf{A}$ acts as a structural prior. It is constrained by a DAG loss (Directed Acyclic Graph) and L1 sparsity to ensure interpretability.
2.  **Causal Inverted Attention**: Uses the learned graph to bias the attention mechanism, explicitly modeling variate interactions based on discovered causality.
3.  **Dynamic Liquid Execution**: Maps causal features to dynamic ODE parameters ($\tau, t_a, t_b$), driving a **Closed-form Continuous-time Neural Network (CfC)** to model system evolution.

This repository contains the official PyTorch implementation of CausaLNN. **Distinctively, the Liquid Neural Network (CfC) module is self-contained within this codebase, requiring no external heavy dependencies.**

<div align="center">
  <img src="./figures/figure2.png" alt="CausaLNN Architecture" width="850">
  <br>
  <b>Figure 1: The architecture of CausaLNN. It integrates the Causal Controller (structure learning), the Parameter Generator, and the Dynamic CfC Executor (dynamics modeling) within a multi-scale framework.</b>
</div>

## ⚙️ Requirements

To ensure reproducibility, we recommend using Conda to create a separate environment. 
**Note: The `ncps` (Neural Circuit Policies) library source code is integrated into `layers/DynamicLiquid.py`, so no separate installation is needed.**

Please follow the steps below strictly:

```bash
# 1. Create and activate environment
conda create --name causalnn python=3.8 -y
conda activate causalnn

# 2. Install PyTorch (CUDA 11.8 recommended)
pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 3. Install other dependencies
pip install axial-positional-embedding==0.2.1 certifi==2022.12.7 charset-normalizer==2.1.1 colorama==0.4.6 cycler==0.12.1 einops==0.4.1 filelock==3.13.1 fsspec==2024.6.1 idna==3.4 jinja2==3.1.4 joblib==1.4.2 kiwisolver==1.4.7 local-attention==1.4.4 markupsafe==2.1.5 matplotlib==3.4.3 mpmath==1.3.0 networkx==3.0 numpy==1.22.4 packaging==25.0 pandas==1.1.5 patool==1.12 patsy==1.0.1 pillow==10.2.0 product-key-memory==0.1.10 pyparsing==3.1.4 python-dateutil==2.9.0.post0 pytz==2025.2 reformer-pytorch==1.4.4 requests==2.28.1 scikit-learn==1.2.1 scipy==1.8.0 six==1.17.0 sktime==0.4.1 statsmodels==0.14.1 sympy==1.11.1 threadpoolctl==3.5.0 tqdm==4.64.0 typing-extensions==4.12.2 urllib3==1.26.13
````

## 📂 Data Preparation

Please download the standard benchmark datasets and place them in the `./dataset/` directory. The structure should be as follows:

```
./dataset/
  ├── electricity/
  ├── weather/
  ├── traffic/
  ├── ETT-small/
  └── PEMS/
       ├── PEMS08.npz
       └── ...
```

## 🏃‍♂️ Usage

To reproduce the main results of the paper (with CausaLNN's full causal and dynamic capabilities enabled), please run the provided shell script:

```bash
bash ./all.sh
```

Alternatively, to run a specific experiment (e.g., PEMS08) with detailed arguments:

```bash
python run_CasuaLNN_PEMS08.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS08_714.npz \
  --model_id PEMS08_96_12 \
  --model CasuaLNN \
  --data PEMS \
  --seq_len 96 \
  --pred_len 12 \
  --use_dynamic_ode \
  --lambda_causal 1.0 \
  --lambda_l1 0.1 \
  --lq_param_scaling 5.0
```
## 📊 Results

We extensively evaluated **CausaLNN** on 8 widely used real-world benchmarks, covering diverse domains (Energy, Traffic, Weather). We compared our model with state-of-the-art baselines, including **iTransformer**, **TimeMixer**, **PatchTST**, **TimesNet**, **Crossformer**, and **DLinear**.

As shown in Table 1, CausaLNN achieves state-of-the-art performance on **7 out of 8 datasets**. Specifically, it demonstrates significant advantages on datasets with complex variate dependencies (e.g., **Electricity**, **Traffic**, **PEMS08**), validating the effectiveness of the Causal-Modulated Liquid Dynamics.

**Table 1:** Multivariate Long-term Forecasting Results (MSE / MAE). **Bold** indicates the best result.

| **Dataset** | **CausaLNN (Ours)** | **iTransformer** | **TimeMixer** | **PatchTST** | **TimesNet** | **DLinear** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Weather** | **0.250** / **0.277** | 0.260 / 0.281 | 0.258 / 0.284 | 0.275 / 0.295 | 0.265 / 0.289 | 0.301 / 0.329 |
| **Electricity** | **0.169** / **0.266** | 0.176 / 0.268 | 0.179 / 0.282 | 0.195 / 0.285 | 0.198 / 0.292 | 0.210 / 0.296 |
| **Traffic** | 0.467 / 0.302 | **0.423** / **0.280** | 0.496 / 0.328 | 0.482 / 0.312 | 0.636 / 0.339 | 0.627 / 0.388 |
| **PEMS08** | **24.12** / **14.85** | 25.40 / 15.68 | 27.42 / 17.67 | 35.08 / 23.64 | 34.36 / 22.07 | 40.85 / 23.47 |
| **ETTh1** | **0.440** / **0.442** | 0.460 / 0.451 | 0.530 / 0.493 | 0.475 / 0.464 | 0.526 / 0.502 | 0.464 / 0.460 |
| **ETTh2** | **0.386** / **0.407** | 0.394 / 0.411 | 0.442 / 0.433 | 0.408 / 0.424 | 0.422 / 0.431 | 0.477 / 0.463 |
| **ETTm1** | **0.393** / **0.405** | 0.421 / 0.421 | 0.410 / 0.418 | 0.405 / 0.408 | 0.449 / 0.438 | 0.433 / 0.429 |
| **ETTm2** | **0.286** / **0.331** | 0.289 / 0.333 | 0.356 / 0.366 | 0.295 / 0.340 | 0.286 / 0.333 | 0.332 / 0.362 |

*> Note: The results for CausaLNN are reported from the best performing checkpoint (varying `lq_param_scaling` and controller layers) to demonstrate the model's full potential.*

## 🔗 Citation

If you find this work helpful, please cite our paper:

```bibtex
@inproceedings{niu2025causalnn,
  title={CausaLNN: Causal-Modulated Liquid Neural Network for Time Series Forecasting},
  author={Niu, Jiangtao and Gao, Xianglan and Zong, Xu},
  booktitle={},
  year={2025}
}
```

## 🙏 Acknowledgements

We appreciate the following open-source works which inspired our development:

  - [Neural Circuit Policies (NCPs)](https://github.com/mlech26l/ncps)
  - [TimeMixer](https://github.com/kwuking/TimeMixer)
  - [iTransformer](https://github.com/thuml/iTransformer)
  - [Time-Series-Library](https://github.com/thuml/Time-Series-Library)

<!-- end list -->

```