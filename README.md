<div align="center">

<img src="assets/icon.svg" alt="Instagram Engagement Predictor icon" width="72" align="center" />

# Instagram Engagement Predictor

*Predict engagement before you publish — a mixture-of-experts model trained on 64,000+ Instagram posts.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://instagram-engagement-predictor-8nprsw4e8mmarpvaznpzzd.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange?style=flat-square)](https://scikit-learn.org/stable/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?style=flat-square)](https://streamlit.io)

[Live demo](#live-demo) • [Features](#features) • [Quick start](#quick-start) • [How it works](#how-it-works) • [Repository](#repository-layout) • [Retraining](#retraining) • [Resources](#resources)

</div>

## Live demo

Try it right now, no setup needed:

**→ [instagram-engagement-predictor-8nprsw4e8mmarpvaznpzzd.streamlit.app](https://instagram-engagement-predictor-8nprsw4e8mmarpvaznpzzd.streamlit.app/)**

Describe a post you're planning — followers, hashtags, caption, number of images,
format (image, carousel, reel) and when it goes live — and get an estimated
engagement score instantly. A light/dark theme toggle is built in.

## Features

- **Mixture-of-experts prediction** — KMeans routes each post to one of three
  specialist regressors (Random Forest or Hist-Gradient Boosting).
- **Real planning inputs** — follower count, hashtags, caption length, image
  count, post type, posting date and time, plus account history.
- **Honest model card** — held-out R², RMSE and MAE are shown right in the app.
- **Editorial UI** — theme-aware light & dark palettes, custom fonts and layout.

## Quick start

The project pins `scikit-learn==1.9.1` because `model_artifacts.joblib` is
pickled with that version.

```bash
pip install -r requirements.txt
streamlit run app.py
```

> [!IMPORTANT]
> Keep `scikit-learn` pinned to the exact version used to retrain the model.
> If you bump the version, retrain (see [Retraining](#retraining)) so the pickle
> stays compatible.

## How it works

1. **Compose your post** — audience size, caption, hashtags, image count, format
   and publish date/time.
2. **Route to an expert** — KMeans clusters the post among three historical
   behavior groups.
3. **Get a prediction** — the cluster's tuned regressor estimates engagement
   (likes + comments, plus shares/saves where recorded).

| Cluster | Model | Held-out R² |
| :---: | :--- | :---: |
| 0 | Random Forest | 0.92 |
| 1 | Hist-Gradient Boosting | 0.60 |
| 2 | Hist-Gradient Boosting | 0.89 |
| **Overall** | **Mixture-of-experts** | **0.87** |

> [!WARNING]
> Engagement is long-tailed — a few posts go viral. Predictions are most
> reliable for typical posts and less precise for extreme outliers.

## Repository layout

```
├── app.py                    # Streamlit app (entry point)
├── requirements.txt          # Python dependencies
├── .streamlit/config.toml    # Theme (light + dark palettes, fonts)
├── assets/                   # style.css + icon.svg (UI styling)
├── ml/                       # train_model.py + model_artifacts.joblib
├── data/                     # training CSVs (not used at runtime)
├── notebooks/                # original research notebook (Colab-oriented)
└── README.md
```

The deployed app only needs `ml/model_artifacts.joblib` — it never reads the
CSVs in `data/` at runtime.

## Retraining

Regenerate the artifact from the same pipeline:

```bash
cd ml
python train_model.py
```

This rewrites `ml/model_artifacts.joblib`, then commit and push — the deployed
app picks it up on the next Streamlit Cloud redeploy.

## Resources

- [Streamlit Community Cloud — Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
- [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html)
- [Streamlit theming](https://docs.streamlit.io/develop/concepts/configuration/theming)