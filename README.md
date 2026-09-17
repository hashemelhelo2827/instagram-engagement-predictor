<div align="center">

<img src="assets/icon.svg" alt="Instagram Engagement Predictor icon" width="72" align="center" />

# Instagram Engagement Predictor

*Predict engagement before you publish — a mixture-of-experts model trained on 64,000+ Instagram posts.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://instagram-engagement-predictor-8nprsw4e8mmarpvaznpzzd.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange?style=flat-square)](https://scikit-learn.org/stable/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?style=flat-square)](https://streamlit.io)

[Live demo](#live-demo) • [Features](#features) • [Quick start](#quick-start) • [How it works](#how-it-works) • [Datasets](#datasets) • [Repository layout](#repository-layout) • [Retraining](#retraining) • [Notes](#notes--troubleshooting) • [Resources](#resources)

</div>

## Description

This app predicts the engagement a planned Instagram post is likely to receive
*before it goes live*. You describe the post — audience, caption, hashtags,
format and publish time — and the app returns an estimated engagement score
along with honest hold-out quality metrics.

The model is a **mixture-of-experts (MoE)** ensemble: posts are grouped into
three behavioral clusters with **KMeans**, and each cluster is assigned its own
specialist regressor (Random Forest or Hist-Gradient Boosting). It was trained
on roughly **64,800 real posts** spanning Grammy-adjacent artist accounts and a
mix of image / carousel / reel posts, so it generalizes across audience scales —
from small creators to accounts with hundreds of millions of followers.

## Live demo

Try it right now, no setup needed:

**→ [instagram-engagement-predictor-8nprsw4e8mmarpvaznpzzd.streamlit.app](https://instagram-engagement-predictor-8nprsw4e8mmarpvaznpzzd.streamlit.app/)**

Fill in the form (followers, hashtags, caption, number of images, post format
and schedule), press predict, and you get an estimated engagement score plus a
model report card showing the cluster used and its quality. A light/dark theme
toggle is built into the toolbar.

## Features

- **Mixture-of-experts prediction** — KMeans routes each post to one of three
  specialist regressors: a Random Forest (cluster 0) or one of two Hist-Gradient
  Boosting models (clusters 1–2).
- **Real planning inputs** — follower count, number of hashtags, caption length,
  image count, post type (image / carousel / reel), posting date & time, and the
  account's historical engagement behavior.
- **Endpoint-agnostic target** — engagement is summed differently per source
  (`likes + comments` from the Grammy dataset, plus `shares + saves` where
  available in the analytics dataset), then scored consistently at inference.
- **Honest model card** — held-out R², RMSE and MAE are displayed right in the
  app, not hidden behind marketing numbers.
- **Editorial UI** — theme-aware light & dark palettes (warm bone / warm dark),
  Outfit + JetBrains Mono typography, compact and responsive form controls.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

> [!IMPORTANT]
> `model_artifacts.joblib` is pickled with `scikit-learn==1.9.1`. Keep the pin in
> `requirements.txt` — upgrading scikit-learn without retraining will break
> loading. If you do upgrade, retrain first (see [Retraining](#retraining)).

## How it works

### Prediction pipeline

1. **Compose the post** — audience size, caption, hashtags, image count, format
   and publish date/time.
2. **Engineering** — build derived features from the raw fields (see the table
   below).
3. **Log transform** — numeric features get `log1p` to tame the long-tailed
   engagement distribution.
4. **Standardize** — features are scaled to zero mean / unit variance.
5. **Route** — KMeans assigns the post to one of three clusters.
6. **Predict** — the cluster's expert regressor produces the engagement estimate.


### Target definition

Engagement is measured as reactions to the post:

```text
Grammy dataset:     Engagement = likes + comments
Analytics dataset:  Engagement = likes + comments + shares + saves
```

The two datasets are harmonized onto a common feature schema before training.

### Feature engineering

| Feature | Logic |
| --- | --- |
| `hour` / `reach_time_bucket` | Posting hour mapped to `peak` (6–8, 11–12, 19–20), `normal`, or `low` (23–5) |
| `hashtag_bucket` | 0, 1–5, 6–15, 16+ → `none` / `low` / `medium` / `high` |
| `caption_length_bucket` | <50, 50–150, 150–300, 300+ → `short` / `medium` / `long` / `very_long` |
| `is_weekend` | 1 when posting on a Saturday or Sunday |
| `is_holiday` | 1 when the posting date is a US public holiday |
| `user_post_count` | Sequential post index per account (posting cadence) |
| `user_median_engagement` | Expanding median of the account's prior engagement (account history) |
| `video` / `carousel` / `post_images` | One-hot flags for reel / carousel / single image |
| `followers`, `number_hashtags`, `length_caption`, `Engagement` | Log-transformed (`log1p`) |

In total the model consumes **24 numeric feature columns** after one-hot encoding.

### Expert models

| Cluster | Model | Key parameters | Training rows |
| :---: | :--- | :--- | :---: |
| 0 | Random Forest | 300 estimators, max depth 8 | 5,984 |
| 1 | Hist-Gradient Boosting | lr 0.08, 300 iters, max depth 8 | 27,450 |
| 2 | Hist-Gradient Boosting | lr 0.08, 300 iters, max depth 8 | 31,376 |

Models were selected per cluster (RF for one, HGBR for the others) to balance
capacity against cluster size.

### Validation

The combined dataset is split **80/20** (`random_state=42`), and held-out
performance is measured on the routed predictions of the cluster experts:

| Metric | Value |
| :--- | :---: |
| R² | 0.865 |
| RMSE | 183,020 |
| MAE | 30,967 |

> [!WARNING]
> Engagement is long-tailed — a few posts go viral. Predictions are most
> reliable for typical posts and least precise for extreme outliers, which
> inflate RMSE far above the median-level MAE.

## Datasets

The model is trained on two complementary public-style datasets (for research /
demonstration):

| File | Rows | Timespan | Highlights |
| --- | --- | --- | --- |
| `data/grammy_posts.csv` | 34,811 | ~2022–23 | Grammy-adjacent artist posts; `gender` M/F/MIXED, `followers` up to 380M, likes up to 12.9M |
| `data/instagram_analytics.csv` | 29,999 | 2024–25 | Business-account analytics; media types image (11,927) / carousel (10,627) / reel (7,445); includes `reach`, `impressions`, `engagement_rate` |

Both files are **only used for training**. The deployed app never loads them at
runtime — everything needed for inference lives in `ml/model_artifacts.joblib`.

## Repository layout

```
├── app.py                    # Streamlit app (entry point)
├── requirements.txt          # Python dependencies
├── .streamlit/config.toml    # App + theme config (light & dark palettes, fonts)
├── assets/                   # style.css + icon.svg (UI styling)
├── ml/
│   ├── train_model.py        # Training + artifact builder
│   └── model_artifacts.joblib# Scalers, KMeans, experts, metadata (runtime-only)
├── data/                     # Training CSVs (not used at runtime)
├── notebooks/                # Original research notebook (Colab-oriented)
├── .devcontainer/            # Dev container configuration (GitHub Codespaces)
└── README.md
```

Environment and dependency pins:

```text
streamlit, pandas, numpy, scikit-learn==1.9.1, holidays, joblib
```

## Retraining

Regenerate the model artifact with the same training pipeline:

```bash
cd ml
python train_model.py
```

This rewrites `ml/model_artifacts.joblib` with a dictionary of:
`scaler`, `kmeans`, `experts`, `feature_columns`, `log_cols`,
`cat_categories`, `catcol`, and `holdout_metrics` (R²/RMSE/MAE).

Then commit, push, and the deployed app picks up the new artifact on the next
Streamlit Cloud redeploy.

## Notes & troubleshooting

- **scikit-learn pin** — keep `scikit-learn==1.9.1` to stay pickle-compatible with
  the artifact; retrain after any upgrade.
- **Streamlit Cloud entry point** — make sure the deployed app's main file path
  is `app.py` (repo root). After a repository restructure, an old entry point
  (e.g. a moved folder) will fail its redeploy until corrected in the dashboard.
- **Interpretation** — predictions are an estimate of total reactions, not a
  guarantee; use them to compare post ideas and time posts strategically.

## Resources

- [Streamlit Community Cloud — Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
- [Streamlit theming](https://docs.streamlit.io/develop/concepts/configuration/theming)
- [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html)
