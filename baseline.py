import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
#FOR RESULT PERSISTENCE..
RANDOM_STATE = 42

NUMERIC = ["distance", "weight", "month", "dow", "doy"]
CATEGORICAL = ["pickup", "delivery", "equipment"]
FEATURES = NUMERIC + CATEGORICAL


def clean(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["weight"] = df["weight"].abs()
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek
    df["doy"] = df["date"].dt.dayofyear
    return df


def preprocess():
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CATEGORICAL)
    ])


def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 1, None)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = np.mean(np.abs((y_test - pred) / y_test)) * 100

    print(f"\n{name}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE_pct": round(mape, 2)
    }


def main():

    tt = clean(pd.read_csv("data/train-test.csv"))

    # Same 56-day temporal split as XGBoost
    cutoff = tt["date"].max() - pd.Timedelta(days=56)

    train = tt[tt["date"] <= cutoff]
    holdout = tt[tt["date"] > cutoff]

    X_train = train[FEATURES]
    y_train = train["posted_rate"]

    X_test = holdout[FEATURES]
    y_test = holdout["posted_rate"]

    results = {}

    # Linear Regression
    linear = Pipeline([
        ("prep", preprocess()),
        ("model", LinearRegression())
    ])

    results["LinearRegression"] = evaluate(
        "Linear Regression",
        linear,
        X_train, y_train,
        X_test, y_test
    )

    # Random Forest
    rf = Pipeline([
        ("prep", preprocess()),
        ("model", RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    results["RandomForest"] = evaluate(
        "Random Forest",
        rf,
        X_train, y_train,
        X_test, y_test
    )

    print("\nFinal comparison:")
    print(pd.DataFrame(results).T)


if __name__ == "__main__":
    main()