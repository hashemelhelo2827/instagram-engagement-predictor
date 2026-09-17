# Instagram Engagement Predictor

Predict how much engagement a planned Instagram post will get, before you hit
publish. The app combines **KMeans clustering** and **mixture-of-experts
regressors** (Random Forest + Hist-Gradient Boosting) trained on 64,000+
historical Instagram posts, then keeps predictions honest with reported
held-out performance.

## How it works

1. **Compose your post** — audience size, hashtags, caption length, number of
   images/slides, format (image, carousel, reel) and when it goes live.
2. **Route to an expert** — KMeans groups the post into one of three
   historical behavior clusters.
3. **Get a prediction** — a Random Forest or Gradient Boosting expert, tuned
   per cluster, estimates likes + comments (+ shares/saves where available).

> **Long-tail data caveat:** engagement is long-tailed (a few viral posts).
> Predictions are most reliable for typical posts and less precise on extreme
> outliers. Reported R² / RMSE / MAE are held-out numbers shown at the bottom
> of the app.

## Repository layout

```
├── app.py                      # Streamlit app (entry point)
├── requirements.txt            # Python dependencies
├── .streamlit/config.toml      # Theme (light + dark palettes, fonts)
├── assets/                     # App styling & icon
│   ├── style.css               # Editorial theme, theme-aware CSS
│   └── icon.svg                # Tab icon
├── ml/                         # Model pipeline
│   ├── train_model.py          # Training script (KMeans + experts)
│   └── model_artifacts.joblib  # Trained scaler, clusters & experts
├── data/                       # Source datasets (not used at runtime)
│   ├── grammy_posts.csv        # Grammy IG posts, semi-colon separated
│   └── instagram_analytics.csv # Account analytics
├── notebooks/                  # Original research notebook (Colab-oriented)
└── README.md
```

The deployed app never reads the CSVs — it only needs
`ml/model_artifacts.joblib`. Data and notebook are kept for reproducibility.

## Run locally (Windows)

`scikit-learn` is pinned to `==1.9.1` because the model artifact is pickled
with that version. Keep the pin in sync with whatever version you retrain
with.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)

1. Push this repository to GitHub.
2. Sign in at https://streamlit.io with your GitHub account.
3. **Create app** → pick this repo and branch, set the main file path to
   **`app.py`**.
4. Click **Deploy**. Streamlit installs `requirements.txt` and gives you a
   public `*.streamlit.app` URL.

Redeploys happen automatically on every push to the deployed branch.

## Retraining on new data

```bash
cd ml
python train_model.py
```

This regenerates `ml/model_artifacts.joblib` with the same pipeline. Commit
the new artifact and push — the app picks it up on the next deploy.
The `notebooks/` copy is the original Colab research; `train_model.py` is the
scripted, reproducible version of the same pipeline.

> If you upgrade or downgrade scikit-learn, retrain so the pickle version
> matches the pin in `requirements.txt`.