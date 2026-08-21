from src.ames.tune import get_train_test_split, evaluate_final_holdout
from src.ames.pipeline import build_preprocessor, create_full_pipeline
from sklearn.linear_model import Ridge

def test_train_test_split():
    X_train, X_test, y_train, y_test = get_train_test_split(test_size=0.2, random_seed=42)
    total_len = len(X_train) + len(X_test)
    assert abs(len(X_test) / total_len - 0.2) < 0.02
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

def test_evaluate_final_holdout():
    X_train, X_test, y_train, y_test = get_train_test_split(test_size=0.2, random_seed=42)
    preprocessor = build_preprocessor(X_train)
    pipeline = create_full_pipeline(Ridge(), preprocessor)
    res = evaluate_final_holdout(pipeline, X_train.head(100), y_train.head(100), X_test.head(30), y_test.head(30))
    assert "log_rmse" in res
    assert "dollar_mae" in res
    assert res["log_rmse"] > 0
    assert res["dollar_mae"] > 0
