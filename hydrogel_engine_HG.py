from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

from joblib import dump, load

from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


@dataclass
class HGConfig:
    header_row: int = 1

    # BEFORE (monomer)
    A_id_col: str = "ID"
    A_name_col: str = "Monomer"
    A_smiles_col: str = "Monomer.1"
    A_MW_col: str = "MW"
    A_HSP_dD_col: str = "HSP_dD"
    A_HSP_dP_col: str = "HSP_dP"
    A_HSP_dH_col: str = "HSP_dH"

    # BEFORE (gCP)
    B_id_col: str = "ID"
    B_name_col: str = "gCP"
    B_smiles_col: str = "gCP.1"
    B_MW_col: str = "MW"
    B_PDI_col: str = "PDI"
    B_HSP_dD_col: str = "HSP_dD"
    B_HSP_dP_col: str = "HSP_dP"
    B_HSP_dH_col: str = "HSP_dH"

    # BEFORE (solvent)
    C_id_col: str = "ID"
    C_name_col: str = "Solvent"
    C_smiles_col: str = "Solvent.1"
    C_bp_col: str = "Boiling Point"
    C_dielectric_col: str = "Dielectric"
    C_HSP_dD_col: str = "HSP_dD"
    C_HSP_dP_col: str = "HSP_dP"
    C_HSP_dH_col: str = "HSP_dH"

    # AFTER (afterHG)
    run_id_col: str = "hydrogel#"
    after_A_id_col: str = "Monomer"
    after_B_id_col: str = "gCP"
    after_C_id_col: str = "Solvent"

    ratio_col: str = "P_A_to_B_wtfrac"
    crosslink_col: str = "P_crosslink_time_min"
    solids_col: str = "P_solids_wt/c"

    label_col: str = "1/0"
    type_col: str = "p- or n-"

    ratio_decimals: int = 3
    solids_decimals: int = 2

    cv_splits: int = 5
    random_state: int = 42

    reg_targets: List[str] = None
    weights: Dict[str, float] = None


def default_hg_config() -> HGConfig:
    cfg = HGConfig()
    cfg.reg_targets = [
        "Mobility_cm2V-1s-1",
        "Transconductance_mS",
        "Vth_V",
        "Electical_conductivity_Scm-1",
        "On-off_ratio",
        "Volumetric_capacitance_C*",
        "Ionic_conductivity",
        "uC*_Figure-of-merit",
        "Youngs_modulus",
        "Fracture_strain",
        "Toughness",
        "Cyclability",
    ]
    cfg.weights = {
        "Mobility_cm2V-1s-1": 0.20,
        "Transconductance_mS": 0.20,
        "Vth_V": -0.05,
        "Electical_conductivity_Scm-1": 0.10,
        "On-off_ratio": 0.05,
        "Volumetric_capacitance_C*": 0.15,
        "Ionic_conductivity": 0.05,
        "uC*_Figure-of-merit": 0.20,
        "Youngs_modulus": -0.10,
        "Fracture_strain": 0.10,
        "Toughness": 0.05,
        "Cyclability": 0.05,
    }
    return cfg


def _clean_columns(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        c2 = str(c).strip()
        c2 = re.sub(r"\s+", " ", c2)
        out.append(c2)
    return out


def read_excel(path: str, sheet_name: Optional[str] = None, header_row: int = 1) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row, engine="openpyxl")
    df.columns = _clean_columns(df.columns.tolist())
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def parse_A_table(dfA: pd.DataFrame, cfg: HGConfig) -> pd.DataFrame:
    df = dfA.copy()
    df.columns = _clean_columns(df.columns.tolist())
    A = pd.DataFrame({
        "A_id": _to_num(df[cfg.A_id_col]),
        "A_name": df.get(cfg.A_name_col, np.nan),
        "A_smiles": df.get(cfg.A_smiles_col, np.nan),
        "A_MW": _to_num(df.get(cfg.A_MW_col, np.nan)),
        "A_HSP_dD": _to_num(df.get(cfg.A_HSP_dD_col, np.nan)),
        "A_HSP_dP": _to_num(df.get(cfg.A_HSP_dP_col, np.nan)),
        "A_HSP_dH": _to_num(df.get(cfg.A_HSP_dH_col, np.nan)),
    }).dropna(subset=["A_id"]).drop_duplicates(subset=["A_id"]).copy()
    A["A_id"] = A["A_id"].astype(int)
    return A


def parse_B_table(dfB: pd.DataFrame, cfg: HGConfig) -> pd.DataFrame:
    df = dfB.copy()
    df.columns = _clean_columns(df.columns.tolist())
    B = pd.DataFrame({
        "B_id": _to_num(df[cfg.B_id_col]),
        "B_name": df.get(cfg.B_name_col, np.nan),
        "B_smiles": df.get(cfg.B_smiles_col, np.nan),
        "B_MW": _to_num(df.get(cfg.B_MW_col, np.nan)),
        "B_PDI": _to_num(df.get(cfg.B_PDI_col, np.nan)),
        "B_HSP_dD": _to_num(df.get(cfg.B_HSP_dD_col, np.nan)),
        "B_HSP_dP": _to_num(df.get(cfg.B_HSP_dP_col, np.nan)),
        "B_HSP_dH": _to_num(df.get(cfg.B_HSP_dH_col, np.nan)),
    }).dropna(subset=["B_id"]).drop_duplicates(subset=["B_id"]).copy()
    B["B_id"] = B["B_id"].astype(int)
    return B


def parse_C_table(dfC: pd.DataFrame, cfg: HGConfig) -> pd.DataFrame:
    df = dfC.copy()
    df.columns = _clean_columns(df.columns.tolist())
    C = pd.DataFrame({
        "C_id": _to_num(df[cfg.C_id_col]),
        "C_name": df.get(cfg.C_name_col, np.nan),
        "C_smiles": df.get(cfg.C_smiles_col, np.nan),
        "C_boiling_point": _to_num(df.get(cfg.C_bp_col, np.nan)),
        "C_dielectric": _to_num(df.get(cfg.C_dielectric_col, np.nan)),
        "C_HSP_dD": _to_num(df.get(cfg.C_HSP_dD_col, np.nan)),
        "C_HSP_dP": _to_num(df.get(cfg.C_HSP_dP_col, np.nan)),
        "C_HSP_dH": _to_num(df.get(cfg.C_HSP_dH_col, np.nan)),
    }).dropna(subset=["C_id"]).drop_duplicates(subset=["C_id"]).copy()
    C["C_id"] = C["C_id"].astype(int)
    return C


def _hsp_dist(dD1, dP1, dH1, dD2, dP2, dH2) -> np.ndarray:
    return np.sqrt((dD1 - dD2) ** 2 + (dP1 - dP2) ** 2 + (dH1 - dH2) ** 2)


def make_formulation_id(df_after: pd.DataFrame, cfg: HGConfig) -> pd.DataFrame:
    df = df_after.copy()

    ratio = _to_num(df["P_A_to_B_wtfrac"])
    solids = _to_num(df["P_solids_wt/c"])

    if ratio.isna().any():
        raise ValueError("P_A_to_B_wtfrac contains non-numeric values.")
    if solids.isna().any():
        raise ValueError("P_solids_wt/c contains non-numeric values.")
    if ((solids < 0) | (solids > 100)).any():
        raise ValueError("P_solids_wt/c must be in 0..100 (wt%).")

    ratio_r = ratio.round(cfg.ratio_decimals)
    solids_r = solids.round(cfg.solids_decimals)

    a_key = ratio_r.map(lambda x: f"{x:.{cfg.ratio_decimals}f}")
    s_key = solids_r.map(lambda x: f"{x:.{cfg.solids_decimals}f}")

    key_str = (
        df[["A_id", "B_id", "C_id"]].astype(str).agg("|".join, axis=1)
        + "|" + a_key + "|" + s_key
    )
    df["formulation_id"] = key_str.map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()[:12])
    df["P_A_to_B_wtfrac_r3"] = ratio_r
    df["P_solids_r2"] = solids_r
    return df


def build_after_enriched(A: pd.DataFrame, B: pd.DataFrame, C: pd.DataFrame,
                         df_after_raw: pd.DataFrame, cfg: HGConfig) -> pd.DataFrame:
    """
    Enriches afterHG with computed HSP_dist_* and formulation_id.
    Overwrites existing columns if present.
    """
    df = df_after_raw.copy()
    df.columns = _clean_columns(df.columns.tolist())

    df = df.rename(columns={
        cfg.run_id_col: "hydrogel_id",
        cfg.after_A_id_col: "A_id",
        cfg.after_B_id_col: "B_id",
        cfg.after_C_id_col: "C_id",
        cfg.ratio_col: "P_A_to_B_wtfrac",
        cfg.crosslink_col: "P_crosslink_time_min",
        cfg.solids_col: "P_solids_wt/c",
        cfg.label_col: "formable",
        cfg.type_col: "type_pn",
    })

    for k in ["A_id", "B_id", "C_id"]:
        df[k] = _to_num(df[k])
    df = df.dropna(subset=["A_id", "B_id", "C_id"]).copy()
    df["A_id"] = df["A_id"].astype(int)
    df["B_id"] = df["B_id"].astype(int)
    df["C_id"] = df["C_id"].astype(int)

    df = make_formulation_id(df, cfg)

    # join HSP for distance computation
    h = (
        df[["A_id", "B_id", "C_id"]]
        .merge(A[["A_id", "A_HSP_dD", "A_HSP_dP", "A_HSP_dH"]], on="A_id", how="left")
        .merge(B[["B_id", "B_HSP_dD", "B_HSP_dP", "B_HSP_dH"]], on="B_id", how="left")
        .merge(C[["C_id", "C_HSP_dD", "C_HSP_dP", "C_HSP_dH"]], on="C_id", how="left")
    )
    for col in ["A_HSP_dD","A_HSP_dP","A_HSP_dH","B_HSP_dD","B_HSP_dP","B_HSP_dH","C_HSP_dD","C_HSP_dP","C_HSP_dH"]:
        h[col] = _to_num(h[col])

    hsp_ac = _hsp_dist(h["A_HSP_dD"], h["A_HSP_dP"], h["A_HSP_dH"], h["C_HSP_dD"], h["C_HSP_dP"], h["C_HSP_dH"])
    hsp_ab = _hsp_dist(h["A_HSP_dD"], h["A_HSP_dP"], h["A_HSP_dH"], h["B_HSP_dD"], h["B_HSP_dP"], h["B_HSP_dH"])
    hsp_bc = _hsp_dist(h["B_HSP_dD"], h["B_HSP_dP"], h["B_HSP_dH"], h["C_HSP_dD"], h["C_HSP_dP"], h["C_HSP_dH"])

    df["HSP_dist_A_C"] = hsp_ac
    df["HSP_dist_A_B"] = hsp_ab
    df["HSP_dist_B_C"] = hsp_bc

    df["formable"] = _to_num(df.get("formable", np.nan))
    return df


def build_ml_table(A: pd.DataFrame, B: pd.DataFrame, C: pd.DataFrame,
                   df_after_raw: pd.DataFrame, cfg: HGConfig) -> pd.DataFrame:
    df_after = build_after_enriched(A, B, C, df_after_raw, cfg)
    df = df_after.merge(A, on="A_id", how="left").merge(B, on="B_id", how="left").merge(C, on="C_id", how="left")
    return df


def _build_preprocess(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    X = df[feature_cols].copy()
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        sparse_threshold=0.3
    )
    return pre, num_cols, cat_cols


def _candidate_classifiers(pre, num_cols, cat_cols, rs: int):
    models = {}
    models["RF"] = Pipeline([("preprocess", pre),
                             ("model", RandomForestClassifier(n_estimators=500, random_state=rs, class_weight="balanced"))])

    if HAS_XGB:
        models["XGB"] = Pipeline([("preprocess", pre),
                                  ("model", XGBClassifier(
                                      n_estimators=700, max_depth=6, learning_rate=0.05,
                                      subsample=0.9, colsample_bytree=0.9,
                                      eval_metric="logloss", tree_method="hist", random_state=rs
                                  ))])

    if HAS_LGBM:
        models["LGBM"] = Pipeline([("preprocess", pre),
                                   ("model", LGBMClassifier(
                                       n_estimators=1000, learning_rate=0.05,
                                       subsample=0.9, colsample_bytree=0.9,
                                       random_state=rs
                                   ))])

    num_poly = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                         ("scaler", StandardScaler(with_mean=False))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre_poly = ColumnTransformer(
        transformers=[("num", num_poly, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        sparse_threshold=0.3
    )
    models["LogRegPoly2"] = Pipeline([("preprocess", pre_poly),
                                     ("model", LogisticRegression(max_iter=5000))])
    return models


def _candidate_regressors(pre, rs: int):
    models = {}
    models["RF_MO"] = Pipeline([("preprocess", pre),
                               ("model", MultiOutputRegressor(RandomForestRegressor(n_estimators=700, random_state=rs, n_jobs=-1)))])

    if HAS_XGB:
        models["XGB_MO"] = Pipeline([("preprocess", pre),
                                     ("model", MultiOutputRegressor(XGBRegressor(
                                         n_estimators=1000, max_depth=8, learning_rate=0.05,
                                         subsample=0.9, colsample_bytree=0.9,
                                         tree_method="hist", random_state=rs
                                     )))])

    if HAS_LGBM:
        models["LGBM_MO"] = Pipeline([("preprocess", pre),
                                      ("model", MultiOutputRegressor(LGBMRegressor(
                                          n_estimators=1400, learning_rate=0.05, random_state=rs
                                      )))])
    return models


def _mo_r2_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean([r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]))


def _mo_rmse_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmses = [float(np.sqrt(np.mean((y_true[:, i] - y_pred[:, i]) ** 2))) for i in range(y_true.shape[1])]
    return float(np.mean(rmses))


def train_pipeline(A_path: str, B_path: str, C_path: str, after_train_path: str,
                   outdir: str, cfg: Optional[HGConfig] = None,
                   export_enriched_after: bool = True,
                   logger=None) -> Dict[str, Any]:
    """
    Train pipeline:
    1. Parse A/B/C
    2. Enrich afterHG with formulation_id + HSP distances
    3. If export_enriched_after=True, overwrite original afterHG file with enriched version
    4. Build ML table and train models (classifier + regressor via CV)
    5. Save models and reports
    """
    os.makedirs(outdir, exist_ok=True)
    cfg = cfg or default_hg_config()

    def log(m: str):
        if logger:
            logger(m)
        else:
            print(m)

    log("Loading inputs...")
    dfA = read_excel(A_path, header_row=cfg.header_row)
    dfB = read_excel(B_path, header_row=cfg.header_row)
    dfC = read_excel(C_path, header_row=cfg.header_row)
    df_after_raw = read_excel(after_train_path, header_row=cfg.header_row)

    A = parse_A_table(dfA, cfg)
    B = parse_B_table(dfB, cfg)
    C = parse_C_table(dfC, cfg)

    log("Building enriched afterHG + ML table...")
    df_after_enriched = build_after_enriched(A, B, C, df_after_raw, cfg)

    if export_enriched_after:
        log(f"Overwriting afterHG file with enriched version: {after_train_path}")
        df_after_enriched.to_excel(after_train_path, index=False, sheet_name="Sheet1", engine="openpyxl")

    df_ml = df_after_enriched.merge(A, on="A_id", how="left").merge(B, on="B_id", how="left").merge(C, on="C_id", how="left")

    target_cols = [t for t in cfg.reg_targets if t in df_ml.columns]
    drop_cols = set(["formable"]) | set(target_cols) | {"hydrogel_id", "formulation_id"}
    feature_cols = [c for c in df_ml.columns if c not in drop_cols]
    feature_cols = [c for c in feature_cols if not c.endswith("_smiles")]

    log("Problem 1: CV select classifier (metric=ROC-AUC)...")
    df_clf = df_ml.dropna(subset=["formable"]).copy()
    Xc = df_clf[feature_cols]
    yc = df_clf["formable"].astype(int).values

    pre, num_cols, cat_cols = _build_preprocess(df_clf, feature_cols)
    clf_models = _candidate_classifiers(pre, num_cols, cat_cols, cfg.random_state)
    cv_clf = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state)
    scoring = {"auc": "roc_auc", "f1": "f1", "acc": "accuracy"}

    clf_rows = []
    for name, model in clf_models.items():
        log(f"  CV {name}...")
        out = cross_validate(model, Xc, yc, scoring=scoring, cv=cv_clf, n_jobs=-1)
        clf_rows.append({
            "model": name,
            "cv_auc_mean": float(np.mean(out["test_auc"])),
            "cv_f1_mean": float(np.mean(out["test_f1"])),
            "cv_acc_mean": float(np.mean(out["test_acc"])),
        })
    clf_report = pd.DataFrame(clf_rows).sort_values("cv_auc_mean", ascending=False)
    best_clf_name = str(clf_report.iloc[0]["model"])
    best_clf = clf_models[best_clf_name].fit(Xc, yc)

    clf_report.to_csv(os.path.join(outdir, "cv_classification_results.csv"), index=False)
    dump(best_clf, os.path.join(outdir, f"best_classifier_{best_clf_name}.joblib"))

    log("Problem 2: CV select regressor (metric=mean R2)...")
    df_reg = df_ml.dropna(subset=target_cols).copy()
    Xr = df_reg[feature_cols]
    Yr = df_reg[target_cols].astype(float).values

    pre_r, _, _ = _build_preprocess(df_reg, feature_cols)
    reg_models = _candidate_regressors(pre_r, cfg.random_state)
    cv_reg = KFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state)

    reg_rows = []
    for name, model in reg_models.items():
        log(f"  CV {name}...")
        r2s, rmses = [], []
        for tr, te in cv_reg.split(Xr):
            model.fit(Xr.iloc[tr], Yr[tr])
            pred = model.predict(Xr.iloc[te])
            r2s.append(_mo_r2_mean(Yr[te], pred))
            rmses.append(_mo_rmse_mean(Yr[te], pred))
        reg_rows.append({"model": name, "cv_r2_mean": float(np.mean(r2s)), "cv_rmse_mean": float(np.mean(rmses))})

    reg_report = pd.DataFrame(reg_rows).sort_values("cv_r2_mean", ascending=False)
    best_reg_name = str(reg_report.iloc[0]["model"])
    best_reg = reg_models[best_reg_name].fit(Xr, Yr)

    reg_report.to_csv(os.path.join(outdir, "cv_regression_results.csv"), index=False)
    dump(best_reg, os.path.join(outdir, f"best_regressor_{best_reg_name}.joblib"))

    df_ml.to_csv(os.path.join(outdir, "ml_table_merged.csv"), index=False)

    log(f"Best classifier: {best_clf_name}")
    log(f"Best regressor : {best_reg_name}")

    return {
        "outdir": outdir,
        "best_classifier": best_clf_name,
        "best_regressor": best_reg_name,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "n_rows_ml": int(len(df_ml)),
        "n_rows_clf": int(len(df_clf)),
        "n_rows_reg": int(len(df_reg)),
    }


def predict_pipeline(A_path: str, B_path: str, C_path: str, after_candidate_path: str,
                     classifier_joblib: str, regressor_joblib: str,
                     out_csv: str, cfg: Optional[HGConfig] = None,
                     export_enriched_after: bool = True,
                     logger=None) -> Dict[str, Any]:
    """
    Predict pipeline:
    1. Parse A/B/C
    2. Enrich candidate afterHG with formulation_id + HSP distances
    3. If export_enriched_after=True, overwrite original candidate afterHG file
    4. Load trained models + predict
    5. Rank by weighted score
    6. Save predictions to CSV
    """
    cfg = cfg or default_hg_config()

    def log(m: str):
        if logger:
            logger(m)
        else:
            print(m)

    log("Loading inputs + models...")
    dfA = read_excel(A_path, header_row=cfg.header_row)
    dfB = read_excel(B_path, header_row=cfg.header_row)
    dfC = read_excel(C_path, header_row=cfg.header_row)
    df_after_raw = read_excel(after_candidate_path, header_row=cfg.header_row)

    A = parse_A_table(dfA, cfg)
    B = parse_B_table(dfB, cfg)
    C = parse_C_table(dfC, cfg)

    df_after_enriched = build_after_enriched(A, B, C, df_after_raw, cfg)

    if export_enriched_after:
        log(f"Overwriting candidate afterHG file with enriched version: {after_candidate_path}")
        df_after_enriched.to_excel(after_candidate_path, index=False, sheet_name="Sheet1", engine="openpyxl")

    df_ml = df_after_enriched.merge(A, on="A_id", how="left").merge(B, on="B_id", how="left").merge(C, on="C_id", how="left")

    clf = load(classifier_joblib)
    reg = load(regressor_joblib)

    target_cols = [t for t in cfg.reg_targets if t in df_ml.columns]
    drop_cols = set(["formable"]) | set(target_cols) | {"hydrogel_id", "formulation_id"}
    feature_cols = [c for c in df_ml.columns if c not in drop_cols]
    feature_cols = [c for c in feature_cols if not c.endswith("_smiles")]

    log("Predicting + ranking...")
    pred = reg.predict(df_ml[feature_cols])

    out = df_ml.copy()
    for i, t in enumerate(target_cols):
        out[f"pred_{t}"] = pred[:, i]

    if hasattr(clf, "predict_proba"):
        out["pred_formable_prob"] = clf.predict_proba(df_ml[feature_cols])[:, 1]
    else:
        out["pred_formable"] = clf.predict(df_ml[feature_cols])

    score = np.zeros(len(out), dtype=float)
    for t in target_cols:
        score += float(cfg.weights.get(t, 0.0)) * out[f"pred_{t}"].values
    out["score"] = score

    out.sort_values("score", ascending=False).to_csv(out_csv, index=False)
    log(f"Saved: {out_csv}")
    return {"out_csv": out_csv, "n_rows": int(len(out))}
