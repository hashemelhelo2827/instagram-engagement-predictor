import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import holidays
from datetime import date, time

st.set_page_config(page_title="Instagram Engagement Predictor", page_icon="📈", layout="centered")

# ----------------------------------------------------------------- load model
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return joblib.load(os.path.join(base_dir, "model_artifacts.joblib"))

artifacts = load_artifacts()
scaler = artifacts["scaler"]
kmeans = artifacts["kmeans"]
experts = artifacts["experts"]
feature_columns = artifacts["feature_columns"]
log_cols = artifacts["log_cols"]
cat_categories = artifacts["cat_categories"]
catcol = artifacts["catcol"]


def bucket(h):
    if h in [6, 7, 8, 11, 12, 19, 20]:
        return "peak"
    elif h in [23, 0, 1, 2, 3, 4, 5]:
        return "low"
    else:
        return "normal"


def build_feature_row(followers, number_hashtags, length_caption, publication_weekday,
                       post_datetime, is_video, is_carousel, post_images,
                       user_post_count, user_median_engagement):
    hour = post_datetime.hour
    reach_time_bucket = bucket(hour)
    us_holidays = holidays.US(years=[post_datetime.year])
    is_holiday = post_datetime.date() in us_holidays
    is_weekend = publication_weekday in ["Saturday", "Sunday"]

    hashtag_bucket = pd.cut([number_hashtags], bins=[-1, 0, 5, 15, 30],
                             labels=['none', 'low', 'medium', 'high'])[0]
    caption_length_bucket = pd.cut([length_caption], bins=[-1, 50, 150, 300, 10000],
                                    labels=['short', 'medium', 'long', 'very_long'])[0]

    raw = {
        "number_hashtags": number_hashtags,
        "user_median_engagement": user_median_engagement,
        "post_images": post_images,
        "user_post_count": user_post_count,
        "length_caption": length_caption,
        "followers": followers,
        "video": "1" if is_video else "0",
        "carousel": "1" if is_carousel else "0",
        "publication_weekday": publication_weekday,
        "caption_length_bucket": str(caption_length_bucket),
        "hashtag_bucket": str(hashtag_bucket),
        "reach_time_bucket": reach_time_bucket,
        "is_weekend": "1" if is_weekend else "0",
        "is_holiday": str(is_holiday),
    }

    row = {}
    for c in log_cols:
        row["log" + c] = np.log1p(raw[c])
    for c in catcol:
        for cat in cat_categories[c][1:]:  # drop_first: skip first (baseline) category
            col_name = f"{c}_{cat}"
            row[col_name] = 1.0 if raw[c] == cat else 0.0

    X = pd.DataFrame([row]).reindex(columns=feature_columns, fill_value=0.0)
    return X


def predict(X):
    X_scaled = scaler.transform(X)
    cluster = kmeans.predict(X_scaled)[0]
    pred = experts[cluster].predict(X)[0]
    return max(pred, 0), cluster


# --------------------------------------------------------------------- UI
st.title("📈 Instagram Engagement Predictor")
st.caption("Estimate likes + comments (+ shares/saves where applicable) for a planned post, "
           "using a mixture-of-experts model (KMeans clustering + Random Forest / Gradient "
           "Boosting) trained on historical Instagram data.")

with st.form("post_form"):
    col1, col2 = st.columns(2)
    with col1:
        followers = st.number_input("Follower count", min_value=0, value=10000, step=100)
        number_hashtags = st.number_input("Number of hashtags", min_value=0, max_value=30, value=5)
        number_images = st.number_input("Number of images (carousel slides / single image)",
                                        min_value=0, max_value=30, value=1, step=1,
                                        help="Images / slides in the post. Use 0 for video-only posts; "
                                             "set 2+ for carousels.")
        caption_text = st.text_area("Caption (used to measure length)", height=100,
                                     placeholder="Type or paste your caption here...")
        length_caption = len(caption_text)
        st.caption(f"Caption length: {length_caption} characters")
    with col2:
        post_date = st.date_input("Post date", value=date.today())
        post_time_of_day = st.time_input("Post time", value=time(12, 0))
        post_type = st.radio("Post type", ["Image", "Carousel", "Video / Reel"])
        publication_weekday = post_date.strftime("%A")
        st.caption(f"Day of week: {publication_weekday}")

    st.markdown("**Account posting history** (leave at 0 if unknown / new account)")
    col3, col4 = st.columns(2)
    with col3:
        user_post_count = st.number_input("Number of previous posts by this account",
                                           min_value=0, value=0, step=1)
    with col4:
        user_median_engagement = st.number_input(
            "Median engagement of this account's previous posts", min_value=0, value=0, step=10)

    submitted = st.form_submit_button("Predict engagement", use_container_width=True)

if submitted:
    from datetime import datetime
    post_datetime = datetime.combine(post_date, post_time_of_day)
    is_video = post_type == "Video / Reel"
    is_carousel = post_type == "Carousel"

    X = build_feature_row(
        followers=followers, number_hashtags=number_hashtags, length_caption=length_caption,
        publication_weekday=publication_weekday, post_datetime=post_datetime,
        is_video=is_video, is_carousel=is_carousel, post_images=number_images,
        user_post_count=user_post_count, user_median_engagement=user_median_engagement,
    )
    pred, cluster = predict(X)

    st.success(f"### Predicted engagement: **{pred:,.0f}**")
    st.caption(f"(routed to expert model {cluster} — "
               f"{experts[cluster].__class__.__name__})")

    with st.expander("See the features the model used"):
        st.dataframe(X.T.rename(columns={0: "value"}))

st.divider()
metrics = artifacts.get("holdout_metrics", {})
if metrics:
    st.caption(
        f"Model held-out performance: R² {metrics.get('r2', 0):.3f} · "
        f"RMSE {metrics.get('rmse', 0):,.0f} · MAE {metrics.get('mae', 0):,.0f}. "
        f"Note: this dataset has a long-tailed engagement distribution (a few viral posts), "
        f"so predictions for typical posts are more reliable than for extreme outliers."
    )
