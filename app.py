import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_percentage_error

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "demand_gbr.pkl")
DATA_PATH = os.path.join(HERE, "sales.csv")

FEATS = ["week", "month", "promo", "lag_1", "lag_4", "roll_4"]

st.set_page_config(page_title="Retail Demand Forecast", page_icon="🛒", layout="wide")


# ---------------------------------------------------------------- loading ---
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_sales():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


model = load_model()
sales = load_sales()


# ------------------------------------------------------------- features ---
def make_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values("date").copy()
    d["week"] = d.date.dt.isocalendar().week.astype(int)
    d["month"] = d.date.dt.month
    d["lag_1"] = d.units.shift(1)
    d["lag_4"] = d.units.shift(4)
    d["roll_4"] = d.units.shift(1).rolling(4).mean()
    return d


@st.cache_data
def compute_mape(_model_ref):
    # held-out evaluation identical to the training notebook (80/20 time split)
    feat = (
        sales.groupby(["store", "category"], group_keys=False)
        .apply(make_features)
        .dropna()
    )
    cut = feat.date.quantile(0.8)
    te = feat[feat.date > cut]
    preds = model.predict(te[FEATS])
    return mean_absolute_percentage_error(te.units, preds)


def forecast(store: str, category: str, weeks: int = 4, promo: int = 0) -> pd.DataFrame:
    hist = (
        sales[(sales.store == store) & (sales.category == category)]
        .sort_values("date")
        .copy()
    )
    out = []
    for _ in range(weeks):
        nxt = hist.date.max() + pd.Timedelta(weeks=1)
        row = {
            "week": int(nxt.isocalendar().week),
            "month": nxt.month,
            "promo": promo,
            "lag_1": hist.units.iloc[-1],
            "lag_4": hist.units.iloc[-4],
            "roll_4": hist.units.iloc[-4:].mean(),
        }
        pred = float(model.predict(pd.DataFrame([row])[FEATS])[0])
        out.append({"date": nxt, "units": round(pred)})
        hist = pd.concat(
            [
                hist,
                pd.DataFrame(
                    [
                        {
                            "date": nxt,
                            "units": pred,
                            "store": store,
                            "category": category,
                            "promo": promo,
                            "region": hist.region.iloc[0],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return pd.DataFrame(out)


# ---------------------------------------------------------------- sidebar ---
st.sidebar.header("Forecast settings")

region = st.sidebar.selectbox("Region", sorted(sales.region.unique()))
stores_in_region = sorted(sales.loc[sales.region == region, "store"].unique())
store = st.sidebar.selectbox("Store", stores_in_region)
category = st.sidebar.selectbox("Category", sorted(sales.category.unique()))
weeks = st.sidebar.slider("Forecast horizon (weeks)", min_value=1, max_value=12, value=4)
promo = st.sidebar.checkbox("Promo running next weeks?", value=False)

st.title("🛒 Retail Store Demand Forecast")
st.caption(f"{region} · {store} · {category}")

# ---------------------------------------------------------------- compute ---
fc = forecast(store, category, weeks=weeks, promo=int(promo))
mape = compute_mape(model)

total_units = int(fc.units.sum())
peak_row = fc.loc[fc.units.idxmax()]
peak_week_label = peak_row.date.strftime("%Y-%m-%d")

col1, col2, col3 = st.columns(3)
col1.metric(f"Units — next {weeks} weeks", f"{total_units:,}")
col2.metric("Model MAPE (held-out)", f"{mape:.1%}")
col3.metric("Peak week", peak_week_label, f"{int(peak_row.units)} units")

# ---------------------------------------------------------------- chart ---
hist = (
    sales[(sales.store == store) & (sales.category == category)]
    .sort_values("date")
    .tail(12)[["date", "units"]]
    .copy()
)
hist["kind"] = "Actual"

fc_chart = fc.rename(columns={"units": "units"}).copy()
fc_chart["kind"] = "Forecast"

chart_df = pd.concat([hist, fc_chart], ignore_index=True)
chart_df["label"] = chart_df["date"].dt.strftime("%Y-%m-%d")

st.subheader("Last 12 weeks + forecast")
pivot = chart_df.pivot(index="label", columns="kind", values="units")
pivot = pivot.reindex(chart_df.drop_duplicates("label")["label"])
st.bar_chart(pivot)

# ---------------------------------------------------------------- tabs ---
tab_table, tab_about = st.tabs(["📄 Raw data", "ℹ️ About"])

with tab_table:
    st.dataframe(chart_df[["date", "kind", "units"]], use_container_width=True)
    csv_bytes = chart_df[["date", "kind", "units"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download table as CSV",
        data=csv_bytes,
        file_name=f"forecast_{store}_{category}.csv",
        mime="text/csv",
    )

with tab_about:
    st.write(
        "Forecast produced by a `GradientBoostingRegressor` trained on weekly "
        "sales per store/category, using last-week (`lag_1`), four-weeks-ago "
        "(`lag_4`) and rolling 4-week average (`roll_4`) demand as features, "
        "plus week, month and promo flags."
    )
    st.write("Model file: `demand_gbr.pkl` · Data file: `sales.csv`")
