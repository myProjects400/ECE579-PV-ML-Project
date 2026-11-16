# Solar Photovoltaic Power Prediction: A Multi-Paradigm Machine Learning Study

Course project for **ECE 579: Machine Learning for Engineering Applications** (Worcester Polytechnic Institute, Prof. Ziming Zhang).
Author: **Albert Shaju**

This project uses a five-city solar irradiance and PV-output dataset (Davis, Huron, Santa Barbara, La Jolla, Amherst) to study five learning paradigms:

| Problem | Paradigm | Notebook |
|---|---|---|
| P1 | Supervised classification (sky condition, generation regime) | [`P1_Classification.ipynb`](Code_base_v2/P1_Classification.ipynb) |
| P2 | Supervised regression of Output Power (same-city, cross-city, sequential K=12, 3-yr vs 6-yr) | [`P2_Regression.ipynb`](Code_base_v2/P2_Regression.ipynb) |
| P3 | Dimensionality reduction (PCA, kernel PCA, autoencoder; t-SNE/UMAP) for classification and regression | [`P3_Dimensionality_Reduction.ipynb`](Code_base_v2/P3_Dimensionality_Reduction.ipynb) |
| P4 | Semi-supervised learning at 10 / 30 / 50 % labels | [`P4_Semi_Supervised_Learning.ipynb`](Code_base_v2/P4_Semi_Supervised_Learning.ipynb) |
| P5 | Transfer learning, Davis (source) to Amherst (target) | [`P5_Transfer_Learning.ipynb`](Code_base_v2/P5_Transfer_Learning.ipynb) |

## Deliverables

- **Report:** [`Report/Report.pdf`](Report/Report.pdf) (LaTeX source in [`LaTeX/main.tex`](LaTeX/main.tex))
- **Slides:** [`Slides/Slides.pdf`](Slides/Slides.pdf) (LaTeX source in [`LaTeX/ppt.tex`](LaTeX/ppt.tex))
- **Video presentation:** [Google Drive](https://drive.google.com/file/d/1yEQ6FFIMTAmYXIgzWIxX5eqDjJoTPSaY/view?usp=sharing)
- **Final metrics:** [`results.csv`](results.csv) / [`results.json`](results.json) (headline results); full per-seed metrics in [`Code_base_v2/Results/`](Code_base_v2/Results)
- **Figures:** [`Code_base_v2/Figures/`](Code_base_v2/Figures)

## Headline results

Held-out test set, mean over three seeds (42, 123, 2026).

| Problem | Setting | Best result |
|---|---|---|
| P1 | Sky condition, Davis / Amherst | 99.15 % / 98.82 % balanced accuracy (XGBoost) |
| P1 | Generation regime, Davis / Amherst | 96.35 % (XGBoost) / 75.90 % (Random Forest) |
| P2 | Same-city, Davis / Amherst | 11.85 kW RMSE, 4.72 % nRMSE (XGBoost/MLP) / 19.96 kW, 15.69 % (MLP) |
| P2 | Sequential K=12, Davis / Amherst | 13.30 kW / 15.96 kW RMSE (Random Forest) |
| P2 | Zero-shot Davis to Huron / Santa Barbara / La Jolla | 114.61 / 102.96 / 77.27 kW RMSE (RBF SVR), > 100 % nRMSE |
| P3 | Closest reduced vs raw features | Amherst generation: KPCA-10 75.56 % vs raw 76.06 %; raw features win all 8 cases |
| P4 | 10 % labels, best SSL vs best supervised | +0.08 to +0.16 pp balanced accuracy (3 of 4 tasks); largest matched gain +5.59 pp (MLP) |
| P5 | Davis to Amherst, k = 10 target labels | 23.22 kW RMSE, joint calibrated pooling (Ridge), vs 75.82 kW zero-shot and 26.19 kW target-only |

**Most interesting finding:** Davis-trained regressors reach about 5 % nRMSE at home but exceed 100 % nRMSE in every other city, because Davis produces 3.3 to 3.5 times more power. Ten labelled Amherst observations are enough to calibrate and pool the source data, cutting RMSE by 69 % versus zero-shot. The weather-to-power *shape* transfers between cities; the absolute *scale* does not.

## Repository layout

```
ECE579-PV-ML-Project/
├── Code_base_v2/
│   ├── 00_Data_Preparation.ipynb      # run first: processed data + chronological 80/20 splits
│   ├── P1_Classification.ipynb
│   ├── P2_Regression.ipynb
│   ├── P3_Dimensionality_Reduction.ipynb
│   ├── P4_Semi_Supervised_Learning.ipynb
│   ├── P5_Transfer_Learning.ipynb
│   ├── Report_making.ipynb            # Colab helper that bundles Results/ and Figures/ for the report
│   ├── Data/Raw/Further Consolidated Data, HnL.xlsx
│   ├── Results/P1 … P5/               # saved metrics (CSV), P4 label masks, P5 few-shot samples
│   └── Figures/P1 … P5/               # figures used in the report and slides
├── Report/Report.pdf
├── Slides/Slides.pdf
├── LaTeX/                             # Overleaf source for the report (main.tex) and slides (ppt.tex)
├── results.csv / results.json         # headline final metrics
└── requirements.txt
```

## How to run

The experiments were run on **Google Colab** (Python 3.13.16, Tesla T4 GPU). They also run in local Jupyter; a GPU is optional.

### Option A: Google Colab (as used for the reported runs)

1. Copy the `Code_base_v2` folder to Google Drive at `MyDrive/Colab Notebooks/ML_ECE_571_Project/Code_base_v2`.
2. Open each notebook in Colab. The notebooks mount Drive and find the project folder automatically.

### Option B: Local Jupyter

```bash
git clone https://github.com/myProjects400/ECE579-PV-ML-Project.git
cd ECE579-PV-ML-Project
pip install -r requirements.txt
cd Code_base_v2
jupyter notebook
```

Start Jupyter from inside `Code_base_v2` so the notebooks find the project folder. You can also set the environment variable `ECE579_PROJECT_ROOT` to the full path of `Code_base_v2`.

### Run order

1. **`00_Data_Preparation.ipynb`** reads `Data/Raw/Further Consolidated Data, HnL.xlsx`. It writes one processed CSV per sheet to `Data/Processed/` and the exact chronological 80/20 train/test split of every sheet to `Data/Processed/Splits/<sheet>_chronological_80_20.csv`.
2. **`P1` to `P5`**, in any order. Each notebook writes its metrics to `Results/Pn/` and its figures to `Figures/Pn/`.

## Reproducibility

- **Seeds:** 42, 123 and 2026 for every stochastic step (model initialization, tree randomness, P4 label masking, P5 target-sample selection). Every metric is reported as mean ± SD over the three seeds.
- **Splits:** chronological 80/20 per sheet after sorting by timestamp, with no random shuffling. The same split indices are reused by every method in a comparison. The split files are regenerated deterministically by `00_Data_Preparation.ipynb`. The P4 labelled/unlabelled masks are in `Results/P4/Label_Splits/` and the P5 few-shot target samples are in `Results/P5/Few_Shot_Splits/`.
- **Leakage control:** scalers, encoders, tercile thresholds, reducers and calibration maps are fitted on training data only (or on the k target labels in P5). GHI and Clearsky GHI are excluded from the sky-condition features. The effect of including Output Power as a feature is tested separately.
- **Package versions:** see [`requirements.txt`](requirements.txt). The recorded versions are scikit-learn 1.6.1, XGBoost 3.4.1 and PyTorch 2.11.0 (CUDA 13.0).

## Data

`Further Consolidated Data, HnL.xlsx` is the course-provided workbook. It has 9 sheets: 5 cities × 3 years (2014–2016; Amherst 2018–2020) and 4 cities × 6 years (2011–2016). Each sheet holds 30-minute measurements between 10:00 and about 14:30 local solar time, with 21 irradiance and weather features plus the target, Output Power (kW).

## AI assistance disclosure

As permitted by the course policy, ChatGPT was used to help understand library APIs, debug code and improve the wording of the report. All experiments were run by the author, and all reported numbers come from those runs.
