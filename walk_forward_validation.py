"""
Walk-forward (expanding-window) validation: checks whether the model
generalizes consistently across multiple future periods, not just the
single 8-week holdout used in train_predict.py.

Three folds, each expanding the training window and validating on the
next full calendar month:
  Fold 1: train Jan-Jul, validate Aug
  Fold 2: train Jan-Aug, validate Sep
  Fold 3: train Jan-Sep, validate Oct

Note: this gives a MORE conservative (higher, more realistic) error
estimate than the single 8-week holdout in train_predict.py. That's an
intentional and honest finding, not a bug -- a single holdout window can
understate real-world variance; averaging across folds gives a more
defensible number for stakeholders.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

FEATURES = ["distance", "weight", "pickup_enc", "delivery_enc", "equipment_enc",
            "month", "dow", "doy"]

FOLDS = [
    ("2025-01-01", "2025-07-31", "2025-08-01", "2025-08-31"),
    ("2025-01-01", "2025-08-31", "2025-09-01", "2025-09-30"),
    ("2025-01-01", "2025-09-30", "2025-10-01", "2025-10-31"),
]


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

    results = []
    for tr_start, tr_end, val_start, val_end in FOLDS:
        train_part = tt[(tt.date >= tr_start) & (tt.date <= tr_end)]
        val_part = tt[(tt.date >= val_start) & (tt.date <= val_end)]

        encoders = {}
        tr_enc = encode(train_part, encoders, fit=True)
        va_enc = encode(val_part, encoders, fit=False)

        model = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                  random_state=42)
        model.fit(tr_enc[FEATURES], tr_enc["posted_rate"])
        pred = model.predict(va_enc[FEATURES])

        mae = mean_absolute_error(va_enc["posted_rate"], pred)
        rmse = np.sqrt(mean_squared_error(va_enc["posted_rate"], pred))
        mape = np.mean(np.abs((va_enc["posted_rate"] - pred) / va_enc["posted_rate"])) * 100

        results.append({"val_period": f"{val_start} to {val_end}",
                         "train_rows": len(train_part), "val_rows": len(val_part),
                         "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE_pct": round(mape, 2)})
        print(f"Val {val_start} to {val_end}: train_rows={len(train_part)} "
              f"val_rows={len(val_part)} MAE={mae:.2f} RMSE={rmse:.2f} MAPE={mape:.2f}%")

    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False))
    print(f"\nMean MAE: {df_results['MAE'].mean():.2f}  (std: {df_results['MAE'].std():.2f})")
    print(f"Mean MAPE: {df_results['MAPE_pct'].mean():.2f}%  (std: {df_results['MAPE_pct'].std():.2f}%)")


if __name__ == "__main__":
    main()
