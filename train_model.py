# train_model.py
"""
Retrain pipeline for Seasonal Flu Vaccine prediction.
Produces:
 - seasonal_flu_pipeline.pkl   (Pipeline with preprocessing + RandomForest)
 - full_feature_list.pkl       (list of original features used as pipeline input)
 - feature_list.pkl            (top-8 original features by importance)
 - defaults.pkl                (default values for each original feature)
"""
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# --- Paths ---
TRAIN_FEATURES = "Data/training_set_features.csv"
TRAIN_LABELS = "Data/training_set_labels.csv"

OUT_PIPELINE = "seasonal_flu_pipeline.pkl"
OUT_FULL_FEATURES = "full_feature_list.pkl"
OUT_TOP8 = "feature_list.pkl"
OUT_DEFAULTS = "defaults.pkl"

# --- Categorical columns ---
CATEGORICAL_COLS = [
    "age_group",
    "education",
    "race",
    "sex",
    "income_poverty",
    "marital_status",
    "rent_or_own",
    "employment_status",
    "hhs_geo_region",
    "census_msa",
    "employment_industry",
    "employment_occupation",
]

def load_and_clean():
    X = pd.read_csv(TRAIN_FEATURES)
    y_df = pd.read_csv(TRAIN_LABELS)

    if "seasonal_vaccine" in y_df.columns:
        y = y_df[["respondent_id", "seasonal_vaccine"]]
    elif "seasonal" in y_df.columns:
        y = y_df[["respondent_id", "seasonal"]]
    else:
        raise ValueError("Could not find seasonal target in labels CSV")

    if "respondent_id" in X.columns and "respondent_id" in y.columns:
        merged = X.merge(y, on="respondent_id", how="inner")
    else:
        raise ValueError("respondent_id must be in both features and labels")

    h1n1_cols = [
        "h1n1_concern", "h1n1_knowledge", "doctor_recc_h1n1",
        "opinion_h1n1_vacc_effective", "opinion_h1n1_risk",
        "opinion_h1n1_sick_from_vacc", "h1n1_vaccine"
    ]
    for c in h1n1_cols:
        if c in merged.columns and c != "seasonal_vaccine":
            merged.drop(columns=c, inplace=True)

    return merged


def build_and_train(merged):
    y = merged["seasonal_vaccine"]
    X = merged.drop(columns=["seasonal_vaccine", "respondent_id"])

    numeric_cols = [c for c in X.columns if c not in CATEGORICAL_COLS]
    categorical_cols = [c for c in CATEGORICAL_COLS if c in X.columns]

    print("Categorical columns used:", categorical_cols)
    print("Numeric columns used:", numeric_cols)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_cols),
            ("num", numeric_pipeline, numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=0
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    print(f"Validation accuracy: {acc:.4f}, AUC: {auc:.4f}")

    pipeline.fit(X, y)

    full_feature_list = list(X.columns)

    defaults = {}
    for col in full_feature_list:
        if col in categorical_cols:
            mode_series = X[col].mode(dropna=True)
            defaults[col] = mode_series.iloc[0] if not mode_series.empty else "Unknown"
        else:
            defaults[col] = float(X[col].median(skipna=True)) if not X[col].dropna().empty else 0.0

    preproc = pipeline.named_steps["preprocessor"]

    try:
        transformed_feature_names = preproc.get_feature_names_out()
    except:
        cat_names = []
        if categorical_cols:
            cat_names = list(
                preproc.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(categorical_cols)
            )
        transformed_feature_names = np.array(cat_names + numeric_cols)

    clf = pipeline.named_steps["clf"]
    importances = clf.feature_importances_

    # -----------------------------------------------------------------------
    # FIXED BLOCK: Correct mapping of transformed → original feature names
    # -----------------------------------------------------------------------
    orig_map = []
    for name in transformed_feature_names:
        s = str(name)

        if "__" in s:
            s = s.split("__", 1)[1]

        matched = None
        for col in full_feature_list:
            if s.startswith(col + "_"):
                matched = col
                break

        if matched is None and s in full_feature_list:
            matched = s

        orig_map.append(matched)

    df_imp = pd.DataFrame({
        "transformed": transformed_feature_names,
        "importance": importances,
        "original": orig_map
    })

    df_imp = df_imp.dropna(subset=["original"])

    orig_importance = (
        df_imp.groupby("original")["importance"]
        .sum()
        .sort_values(ascending=False)
    )

    top8 = list(orig_importance.iloc[:8].index)
    # -----------------------------------------------------------------------

    joblib.dump(pipeline, OUT_PIPELINE)
    joblib.dump(full_feature_list, OUT_FULL_FEATURES)
    joblib.dump(top8, OUT_TOP8)
    joblib.dump(defaults, OUT_DEFAULTS)

    print(f"Saved pipeline -> {OUT_PIPELINE}")
    print(f"Saved full feature list -> {OUT_FULL_FEATURES}")
    print(f"Saved top-8 features -> {OUT_TOP8}")
    print(f"Saved defaults -> {OUT_DEFAULTS}")

    return pipeline, full_feature_list, top8, defaults


if __name__ == "__main__":
    merged = load_and_clean()
    pipeline, full_feature_list, top8, defaults = build_and_train(merged)
