"""
Spotter Labs — Freight Rate Prediction Challenge
Ayesha Shahid
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import json

RANDOM_STATE = 42
FEATURES = ["distance", "weight", "pickup_enc", "delivery_enc", "equipment_enc",
            "month", "dow", "doy"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["weight"] = df["weight"].abs()
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek
    df["doy"] = df["date"].dt.dayofyear
    return df


def encode(df: pd.DataFrame, encoders: dict, fit: bool) -> pd.DataFrame:
    df = df.copy()
    for col in ["pickup", "delivery", "equipment"]:
        if fit:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            df[col + "_enc"] = enc.fit_transform(df[[col]])
            encoders[col] = enc
        else:
            df[col + "_enc"] = encoders[col].transform(df[[col]])
    return df


def make_model():
    return xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=RANDOM_STATE,
    )


def main():
    tt = clean(pd.read_csv("data/train-test.csv"))
    val = clean(pd.read_csv("data/validation.csv"))
    dec = clean(pd.read_csv("data/december-chart-inputs.csv"))

    weight_median = tt["weight"].median()
    tt["weight"] = tt["weight"].fillna(weight_median)
    val["weight"] = val["weight"].fillna(weight_median)
    dec["weight"] = dec["weight"].fillna(weight_median)

    cutoff = tt["date"].max() - pd.Timedelta(days=56)
    train_part = tt[tt["date"] <= cutoff]
    hold_part = tt[tt["date"] > cutoff]

    encoders = {}
    train_part_enc = encode(train_part, encoders, fit=True)
    hold_part_enc = encode(hold_part, encoders, fit=False)

    model_cv = make_model()
    model_cv.fit(train_part_enc[FEATURES], train_part_enc["posted_rate"])
    pred_hold = model_cv.predict(hold_part_enc[FEATURES])

    mae = mean_absolute_error(hold_part_enc["posted_rate"], pred_hold)
    rmse = np.sqrt(mean_squared_error(hold_part_enc["posted_rate"], pred_hold))
    mape = float(np.mean(np.abs((hold_part_enc["posted_rate"] - pred_hold)
                                 / hold_part_enc["posted_rate"])) * 100)

    metrics = {
        "holdout_start": str((cutoff + pd.Timedelta(days=1)).date()),
        "holdout_end": str(tt["date"].max().date()),
        "holdout_rows": int(len(hold_part)),
        "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE_pct": round(mape, 2),
    }
    print("Holdout validation metrics:", json.dumps(metrics, indent=2))

    final_encoders = {}
    tt_enc = encode(tt, final_encoders, fit=True)
    final_model = make_model()
    final_model.fit(tt_enc[FEATURES], tt_enc["posted_rate"])

    # ---- validation.csv predictions: write to a NEW file, project root,
    # correctly named -- never touch data/validation.csv itself ----
    val_enc = encode(val, final_encoders, fit=False)
    val_pred = final_model.predict(val_enc[FEATURES])
    val_pred = np.clip(val_pred, 1, None)

    out = pd.DataFrame({"load_id": val["load_id"], "predicted_rate": val_pred})
    out.to_csv("validation_predictions.csv", index=False)
    print(f"\nWrote validation_predictions.csv ({len(out)} rows) -- in project root, NOT inside data/")

    # ---- december: fill predicted_rate into the same file (this one IS
    # meant to be overwritten in place, per the assessment instructions) ----
    dec_enc = encode(dec, final_encoders, fit=False)
    dec_pred = final_model.predict(dec_enc[FEATURES])
    dec_pred = np.clip(dec_pred, 1, None)

    dec_out = dec[["pickup", "delivery", "distance", "equipment", "weight", "date"]].copy()
    dec_out["date"] = dec_out["date"].dt.strftime("%Y-%m-%d")
    dec_out["predicted_rate"] = dec_pred
    dec_out.to_csv("data/december-chart-inputs.csv", index=False)
    print(f"Wrote predicted_rate into data/december-chart-inputs.csv ({len(dec_out)} rows)")

    importance = pd.Series(final_model.feature_importances_, index=FEATURES) \
        .sort_values(ascending=False).round(4).to_dict()
    with open("metrics.json", "w") as f:
        json.dump({"holdout_metrics": metrics, "feature_importance": importance}, f, indent=2)
    print("\nFeature importance:", json.dumps(importance, indent=2))


if __name__ == "__main__":
    main()
