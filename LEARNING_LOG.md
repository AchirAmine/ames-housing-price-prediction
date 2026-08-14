# Learning Log — Ames Housing Price Prediction

## S1 — Project Setup (2026-08-11)

**What I did:** Created the production src layout, configured a dedicated Python 3.11 virtual environment with all required dependencies, set up .gitignore to isolate datasets and model artifacts, connected the remote GitHub repository, and documented the project foundation.

**What I learned:** Isolating dependencies in a dedicated virtual environment prevents package version conflicts. A clean src package layout separates reusable production code from notebooks and deployment interfaces, while .gitignore safeguards repository performance and data hygiene.

## S2 — Exploratory Data Analysis & Target Transformation (2026-08-14)

**What I did:** Analyzed the 1,460 residential transaction records across 81 attributes, resolved target distribution skewness (1.88) via log1p normalization, distinguished semantic absence from true missingness in amenities (PoolQC, MiscFeature, Alley), and identified Dean De Cock's documented physical outliers (GrLivArea > 4000) in an executed notebook.

**What I learned:** Target distribution skewness dropped from 1.88 to 0.12 using log1p, stabilizing linear error variance. Imputing a dedicated 'None' category for absent amenities preserves critical negative structural signal, whereas discarding or blindly mode-imputing would introduce severe domain distortion.
