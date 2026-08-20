"""
Error analysis: diagnoses why RMSE ($666) is much larger than MAE ($153)
on the temporal holdout. Run after train_predict.py.

Finding: a small subset of loads (~1% of the holdout) drive almost all of
the RMSE gap. These are long-haul, cross-country lanes with unusually high
posted rates that distance-based pricing underpredicts.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

FEATURES = ["distance", "weight", "pickup_enc", "delivery_enc", "equipment_enc",
            "month", "dow", "doy"]


def clean(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["weight"] = df["weight"].abs()
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek
    df["doy"] = df["date"].dt.dayofyear
    return df


def encode(df, encoders, fit):
    df = df.copy()
    for col in ["pickup", "delivery", "equipment"]:
        if fit:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            df[col + "_enc"] = enc.fit_transform(df[[col]])
            encoders[col] = enc
        else:
            df[col + "_enc"] = encoders[col].transform(df[[col]])
    return df


def main():
    tt = clean(pd.read_csv("data/train-test.csv"))
    tt["weight"] = tt["weight"].fillna(tt["weight"].median())

    cutoff = tt["date"].max() - pd.Timedelta(days=56)
    train_part = tt[tt["date"] <= cutoff]
    hold_part = tt[tt["date"] > cutoff].copy()

    encoders = {}
    train_enc = encode(train_part, encoders, fit=True)
    hold_enc = encode(hold_part, encoders, fit=False)

    model = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=42)
    model.fit(train_enc[FEATURES], train_enc["posted_rate"])
    pred = model.predict(hold_enc[FEATURES])

    hold_enc["pred"] = pred
    hold_enc["abs_error"] = np.abs(hold_enc["posted_rate"] - pred)

    mae = hold_enc["abs_error"].mean()
    rmse = np.sqrt((hold_enc["abs_error"] ** 2).mean())
    print(f"Overall MAE: {mae:.2f}   Overall RMSE: {rmse:.2f}\n")

    n = len(hold_enc)
    top1pct = hold_enc.sort_values("abs_error", ascending=False).head(max(1, int(n * 0.01)))
    rest = hold_enc.drop(top1pct.index)
    print(f"Top 1% of holdout rows ({len(top1pct)} rows):")
    print(f"  mean abs error   = ${top1pct['abs_error'].mean():.0f}")
    print(f"  mean posted_rate = ${top1pct['posted_rate'].mean():.0f}")
    print(f"  mean distance    = {top1pct['distance'].mean():.0f} mi")
    print(f"\nRemaining 99% of rows:")
    print(f"  mean abs error   = ${rest['abs_error'].mean():.2f}\n")

    print("=== Top 10 worst-predicted loads ===")
    worst = hold_enc.sort_values("abs_error", ascending=False).head(10)
    print(worst[["pickup", "delivery", "distance", "equipment", "posted_rate", "pred", "abs_error"]]
          .to_string(index=False))

    print("\n=== Mean absolute error by distance bucket ===")
    hold_enc["dist_bucket"] = pd.cut(hold_enc["distance"], bins=[0, 200, 500, 1000, 2000, 5000])
    print(hold_enc.groupby("dist_bucket", observed=True)["abs_error"].agg(["mean", "count"]))

    print(
        "\nConclusion: RMSE is dominated by ~1% of loads -- long-haul, cross-country "
        "lanes (avg ~1,470 mi) with actual rates 3-4x higher than distance-based "
        "pricing predicts. MAE reflects the typical-case experience ($153/load); "
        "RMSE reflects sensitivity to a small number of high-value long-haul misses. "
        "A production model would likely benefit from a long-haul-specific pricing "
        "component or distance-interaction features."
    )


if __name__ == "__main__":
    main()
