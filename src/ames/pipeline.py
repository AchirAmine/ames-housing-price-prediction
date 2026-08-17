from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

ABSENT_CATEGORICAL_FEATURES = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "MasVnrType"
]

def get_feature_groups(X):
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    absent_cat = [c for c in ABSENT_CATEGORICAL_FEATURES if c in cat_cols]
    general_cat = [c for c in cat_cols if c not in absent_cat]
    return num_cols, absent_cat, general_cat

def build_preprocessor(X):
    num_cols, absent_cat, general_cat = get_feature_groups(X)

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    absent_cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    general_cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("absent_cat", absent_cat_transformer, absent_cat),
            ("general_cat", general_cat_transformer, general_cat)
        ],
        remainder="drop"
    )

    return preprocessor

def create_full_pipeline(model, preprocessor):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
