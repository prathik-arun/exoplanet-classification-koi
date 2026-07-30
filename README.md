# exoplanet-classification-koi
# 🪐 Exoplanet Classification — Kepler Objects of Interest

**India High School Exoplanet Data Challenge · Celesta**

A machine learning pipeline that classifies NASA Kepler transit signals as real confirmed exoplanets or false positives, using physical and stellar features from the Kepler mission's cumulative KOI catalog.

---

## 🎯 Overview

NASA's Kepler Space Telescope monitored ~200,000 stars for years, looking for the tiny brightness dips caused by planets passing in front of them. Not every dip is a planet — some are eclipsing binary stars, background noise, or instrument artifacts. This project builds a classifier that separates the real planets from the false alarms, tackling the same vetting problem NASA scientists face.

**Task:** Predict `koi_disposition` (CONFIRMED vs. FALSE POSITIVE) from transit geometry and stellar properties.

## 📊 Dataset

- **Source:** [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) — Cumulative KOI Table (DOI: [10.26133/NEA4](https://doi.org/10.26133/NEA4))
- **Size:** 9,564 rows × 140 columns
- **Target distribution:**

| Class | Count | Share |
|---|---|---|
| FALSE POSITIVE | 4,839 | 50.6% |
| CONFIRMED | 2,747 | 28.7% |
| CANDIDATE (unresolved) | 1,978 | 20.7% |

## 🧹 Approach

1. **Explore** — examined class balance, missingness, and feature distributions.
2. **Clean & de-leak** — removed 59 columns that either leak the label directly or add no signal:
   - Pipeline-derived leakage: `koi_pdisposition`, `koi_fpflag_*`, `koi_vet_stat`, `koi_disp_prov`, `koi_comment`, `koi_vet_date`
   - IDs: `rowid`, `kepid`, `kepoi_name`, `kepler_name`
   - Measurement-uncertainty columns (`_err1` / `_err2`)
   
   Missing values were imputed with the **median** of each feature (robust to outliers). This left 66 physical and signal-quality features.
3. **Engineer** — added 4 domain-aware derived features grounded in transit physics (see [Feature Engineering](#-feature-engineering) below), bringing the total to 70 features.
4. **Model** — trained a binary classifier (CONFIRMED vs. FALSE POSITIVE), setting aside unresolved CANDIDATEs for final scoring.
5. **Evaluate** — 5-fold stratified cross-validation, held-out test set, full metric suite.
6. **Interpret** — permutation importance + a deliberate leakage demonstration.

## 🤖 Models

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.972 | 0.969 | 0.953 | 0.961 | 0.994 |
| **XGBoost** (main model) | **0.977** | **0.969** | **0.967** | **0.968** | **0.996** |

XGBoost was selected as the primary model — it scored highest across every metric and cross-validation confirmed the result was stable, not a fluke.

**Confusion matrix (XGBoost, test set of 1,518):**

| | Predicted: False Pos | Predicted: Confirmed |
|---|---|---|
| **Actual: False Pos** | 951 | 17 |
| **Actual: Confirmed** | 18 | 532 |

Only 35 misclassifications out of 1,518 — a 2.3% error rate.

## 🔍 Key finding: what the model learned

**`koi_prad` (planet radius) is by far the most important feature** — over 3× more influential than any other. This has a clean physical explanation: many false positives are eclipsing binary stars, which block far more light than a real planet could, producing an implausibly large "radius." The next most important features (`koi_dikco_msky`, `koi_max_mult_ev`, `koi_model_snr`) are signal-quality and centroid checks — essentially the model verifying the dimming is real and comes from the right star.

## 🕵️ The leakage demonstration

To confirm the leakage removal actually mattered, a second model was trained **with** the false-positive flag columns left in:

- Honest model (flags removed): **97.7%** accuracy
- "Leaky" model (flags included): **99.1%** accuracy

The leaky model looks better, but it isn't learning anything real — it's just reading NASA's own pipeline verdict back to itself. This mirrors the challenge's own decision to remove `koi_score` for the same reason.

## 🌌 Scoring the unresolved candidates

As a final step, the trained model was applied to the ~1,978 CANDIDATE rows that NASA hasn't yet confirmed or ruled out, ranking them by predicted probability of being a real planet.

## 🧪 Feature Engineering

Beyond the raw columns, four domain-aware features were engineered from transit physics:

- **`transit_snr_ratio`** = depth / duration — separates deep-short transits from shallow-long grazing events
- **`period_duration_ratio`** = period / duration — real transits have a physically consistent timing relationship; binaries often don't
- **`insol_habitable_proxy`** — flags whether a candidate's insolation falls in an Earth-like range (0.25–2)
- **`prad_teq_interaction`** = radius × equilibrium temperature — flags cases where both are unusually large together, a signature of binary-star false positives

Adding these gave a marginal CV F1 improvement (0.965 → ~0.966), which is itself informative: it confirms `koi_prad` alone already captures most of the separable signal, rather than the engineering being wasted effort.

## 🛠️ Tech stack

- Python, pandas, NumPy
- scikit-learn (Random Forest, preprocessing, metrics, permutation importance)
- XGBoost
- matplotlib, seaborn

## ▶️ Running this project

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

Open `Exoplanet_Challenge_Starter.ipynb` and run all cells top to bottom. The notebook automatically downloads the dataset, so no manual file setup is required.

## 📁 Repository contents

- `Exoplanet_Challenge_Starter.ipynb` — full analysis notebook (EDA → cleaning → modeling → evaluation → interpretation)
- `README.md` — this file

## ✍️ Author

Built by **Prathik** for the Celesta India High School Exoplanet Data Challenge.
