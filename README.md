# Instagram Engagement Predictor

A Streamlit app that predicts **Engagement** (likes + comments, plus
shares/saves where available) for a planned Instagram post, using a
mixture-of-experts model: **KMeans (3 clusters)** routes each post to a
**Random Forest** (cluster 0) or **Gradient Boosting** (clusters 1 & 2)
regressor.

Live demo: *[your-app](https://share.streamlit.io)* — deploy it yourself with
the steps below (takes ~2 minutes, free).

## Repository layout

```
├── streamlit_engagement_app/     # the deployed app
│   ├── app.py                    # Streamlit app (form → prediction)
│   ├── model_artifacts.joblib    # trained scaler + KMeans + 3 experts + encoding metadata
│   ├── train_model.py            # script that produces model_artifacts.joblib
│   └── requirements.txt          # Python dependencies
└── research/                     # source data & research notebook (not needed by the app)
    ├── instagram_engagement_model.ipynb   # original Colab notebook
    ├── Copy of Grammy_IG_posts_v2.csv     # Grammy IG posts (semi-colon separated)
    └── Copy of Instagram_Analytics.csv    # Instagram account analytics
```

> The deployed app never touches the CSVs — it only needs
> `model_artifacts.joblib`. The `research/` folder is kept for retraining and
> reproducibility.

## Run locally (Windows)

The project pins `scikit-learn==1.9.1` because `model_artifacts.joblib` is
pickled with that version — keep the pin in sync with the version used to
train.

```bash
cd streamlit_engagement_app
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)

1. Push this repository to GitHub (or just wait — it's already done for you).
2. Sign in at https://streamlit.io with your GitHub account.
3. Click **"Create app"** → pick this repo and branch, set the main file path
   to **`streamlit_engagement_app/app.py`**.
4. Click **Deploy**. Streamlit installs `requirements.txt` and gives you a
   public `*.streamlit.app` URL.

Redeploys happen automatically on every push to the deployed branch.

## Retraining on new data

The notebook in `research/` is Colab-oriented (it mounts Google Drive), while
`train_model.py` reads the CSVs from `../research/`. To retrain:

```bash
cd streamlit_engagement_app
python train_model.py
```

This regenerates `model_artifacts.joblib` with the same pipeline. Commit the
new artifact and push — the app picks it up on the next deploy. If you
upgrade/downgrade scikit-learn, retrain so the pickle version matches the pin
in `requirements.txt`.

## Notes on the model

- The engagement distribution is long-tailed (a handful of viral posts), so
  predictions are most reliable for "typical" posts and less precise on
  extreme outliers — reported held-out R² / RMSE / MAE are shown at the bottom
  of the app.
- "Number of previous posts" and "median engagement of previous posts" are
  account-history features from the original data. For a brand-new account
  leave them at 0; fill in what you know otherwise.