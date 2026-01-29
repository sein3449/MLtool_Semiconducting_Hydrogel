<<Hydrogel ML System (HG Schema)>>

This repository provides a small, reproducible machine-learning workflow for semiconducting hydrogel datasets using two Python modules: hydrogel_engine_hg.py (data processing + training/prediction pipeline) and hydrogel_gui_hg.py (PySide6 GUI for interactive use). The system is designed around an “HG schema” where the materials library is split into three Excel files (Monomer / gCP / Solvent) and the experimental log is stored in an afterHG Excel file. The pipelines compute consistent, stable identifiers (formulation_id) and automatically calculate HSP distance features (HSP_dist_A_C, HSP_dist_A_B, HSP_dist_B_C) from the HSP parameters in the materials tables, then use these features for (1) formability classification and (2) multi-target property regression. A separate Predict flow loads previously trained models and ranks new candidate formulations using a weighted score.

<What’s included?>

- hydrogel_engine_hg.py
	Core logic to read your Excel templates (2-row header), validate/clean columns, merge materials properties into the afterHG table, generate formulation_id, compute HSP distances, run cross-validated model selection for classification/regression, save trained models, and generate predictions + ranking for candidate sets.

- hydrogel_gui_hg.py
	A two-tab GUI (“Train” and “Predict”) built with PySide6. It lets you select input Excel files, output folders, model files (.joblib), and run the pipeline without using the command line. Long operations run in a background thread to keep the UI responsive.

<Input file format>
This repo assumes you maintain four Excel files:

	A table (Monomer): Input_Schema_beforeHG_monomer.xlsx
	Must include columns such as ID, Monomer (name), MW, HSP_dD, HSP_dP, HSP_dH. If there is a duplicated header for SMILES, Excel may appear as 	Monomer.1 after reading; the engine is written to handle this typical “duplicated header name” behavior from pandas.

	B table (gCP): Input_Schema_beforeHG_gCPs.xlsx
	Must include ID, gCP (name), MW, PDI, and HSP_dD/dP/dH (plus optional SMILES).

	C table (Solvent): Input_Schema_beforeHG_solvents.xlsx
	Must include ID, Solvent (name), Boiling Point, Dielectric, and HSP_dD/dP/dH (plus optional SMILES).

	afterHG table (Experiments / Candidates): Input_Schema_afterHG.xlsx
	Must include integer IDs for Monomer, gCP, Solvent (these are keys into A/B/C tables), plus processing columns like P_A_to_B_wtfrac, 	P_crosslink_time_min, P_solids_wt/c, and optional measured properties (electrical/ionic/mechanical). The column 1/0 is used as the formability label (binary).

The engine reads Excel using the second row as the header (header=1) because the provided templates typically have a descriptive header row on top and the real column names on the next row.

<Key behaviors>

	Stable formulation IDs
formulation_id is deterministically generated from (A_id, B_id, C_id, round(P_A_to_B_wtfrac, 3), round(P_solids_wt/c, 2)) and hashed into a short string. This gives you a fixed formulation key even when hydrogel sample IDs change.

	Automatic HSP distance calculation
HSP distances are computed from A/B/C HSP parameters and written into afterHG as HSP_dist_A_C, HSP_dist_A_B, HSP_dist_B_C. This keeps the dataset consistent and prevents manual calculation errors.

	Overwriting afterHG (“enriched”)
When enabled in the GUI, the pipeline overwrites the selected afterHG Excel file with an enriched version containing computed columns (notably formulation_id and HSP distances). If you prefer a safer workflow, you can disable overwriting in the GUI via checkbox (or modify defaults).

	Model training + persistence
The Train step performs cross-validation across multiple candidate models (RandomForest always; optionally XGBoost/LightGBM if installed), selects the best performing model for classification and regression, and saves them as best_classifier_*.joblib and best_regressor_*.joblib.

	Candidate prediction + ranking
Predict loads the saved models, predicts all requested target properties, estimates formability probability (if available), and produces a score as a weighted sum of predicted properties for ranking.

<Installation>

Create a clean environment (recommended) and install dependencies:

- bash
pip install pandas numpy scikit-learn openpyxl joblib pyside6
Optional (enables additional candidate models automatically if installed):

- bash
pip install xgboost lightgbm

<Running the GUI>
Place hydrogel_engine_hg.py and hydrogel_gui_hg.py in the same folder, then run:

bash
python hydrogel_gui_hg.py
If you encounter import errors, ensure you are running the command from the same directory containing both files.

<Output files>

Training produces:

	cv_classification_results.csv
	cv_regression_results.csv
	ml_table_merged.csv
	best_classifier_*.joblib
	best_regressor_*.joblib

Prediction produces:

	A user-specified predictions_candidate.csv containing pred_* columns, pred_formable_prob (if available), and score.

	If “overwrite afterHG with enriched version” is enabled, the selected afterHG file itself will be updated to include computed columns.

Notes / recommended workflow

To avoid accidental data loss, consider keeping a version-controlled or timestamped copy of Input_Schema_afterHG.xlsx before enabling overwrite. If you later decide to freeze formulation_id permanently in Excel and treat it as immutable, the current approach is compatible: the engine can continue to compute it and you can compare/validate it against stored values.

<License>
	- MIT
	- 2026 Copyright Sein Chung POSTECH All rights reserved.