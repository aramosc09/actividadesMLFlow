import argparse
import json
import tempfile
from math import sqrt
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "wine+quality"
TEST_SIZE = 0.2
RANDOM_STATE = 42
EXPERIMENT_NAME = "TC5061_Wine_Quality_Experiments"


def load_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    red = pd.read_csv(data_dir / "winequality-red.csv", sep=";")
    white = pd.read_csv(data_dir / "winequality-white.csv", sep=";")
    red["wine_type"] = "red"
    white["wine_type"] = "white"
    return pd.concat([red, white], ignore_index=True)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    numeric_columns = cleaned.select_dtypes(include="number").columns.tolist()

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")

    if cleaned["quality"].mod(1).ne(0).any():
        raise ValueError("La variable quality contiene valores no enteros.")
    if cleaned.isna().any().any():
        raise ValueError("Se encontraron valores faltantes en el dataset.")

    cleaned["quality"] = cleaned["quality"].astype("int64")
    cleaned["wine_type"] = cleaned["wine_type"].astype("category")
    return cleaned.drop_duplicates(keep="first").reset_index(drop=True)


def build_model(
    numeric_features: list[str],
    polynomial_degree: int,
    fit_intercept: bool,
    positive: bool,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                PolynomialFeatures(degree=polynomial_degree, include_bias=False),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
                ["wine_type"],
            ),
        ],
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                LinearRegression(fit_intercept=fit_intercept, positive=positive),
            ),
        ]
    )


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    predictions = model.predict(X_test)
    absolute_errors = (y_test.to_numpy() - predictions).__abs__()
    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "within_one_point_pct": float((absolute_errors <= 1).mean() * 100),
    }


def run_training(
    data_dir: Path = DATA_DIR,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    polynomial_degree: int = 1,
    fit_intercept: bool = True,
    positive: bool = False,
    experiment_name: str = EXPERIMENT_NAME,
) -> tuple[Pipeline, dict[str, float]]:
    data = clean_data(load_data(data_dir))
    X = data.drop(columns="quality")
    y = data["quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    params = {
        "model_type": "LinearRegression",
        "polynomial_degree": polynomial_degree,
        "fit_intercept": fit_intercept,
        "positive": positive,
        "test_size": test_size,
        "random_state": random_state,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "numeric_features": ",".join(numeric_features),
        "categorical_features": "wine_type",
    }
    run_name = (
        f"linear_deg{polynomial_degree}_intercept{fit_intercept}_"
        f"positive{positive}_seed{random_state}"
    )

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        model = build_model(
            numeric_features=numeric_features,
            polynomial_degree=polynomial_degree,
            fit_intercept=fit_intercept,
            positive=positive,
        )
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        for name, value in params.items():
            mlflow.log_param(name, value)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        with tempfile.TemporaryDirectory() as artifact_dir:
            metrics_path = Path(artifact_dir) / "metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")

        model_example = X_train.head(5).copy()
        model_example["wine_type"] = model_example["wine_type"].astype(str)
        model_signature = infer_signature(
            model_example,
            model.predict(model_example),
        )
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="baseline_model",
            signature=model_signature,
            input_example=model_example,
        )

    print(f"Registros limpios: {len(data):,}")
    print(f"Entrenamiento: {len(X_train):,} filas")
    print(f"Prueba: {len(X_test):,} filas")
    print(f"Grado polinómico: {polynomial_degree}")
    print(f"Fit intercept: {fit_intercept}")
    print(f"Coeficientes positivos: {positive}")
    print(f"Random state: {random_state}")
    print(f"Experimento: {experiment_name}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Run ID: {run.info.run_id}")
    print(f"Modelo registrado en: {model_info.model_uri}")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")
    return model, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el baseline de calidad del vino.")
    parser.add_argument("--degree", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--fit-intercept",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--positive",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    return parser.parse_args()


def main() -> None:
    """Parsea las opciones CLI y ejecuta el entrenamiento.

    --degree controla el grado polinómico (1)
    --fit-intercept activa el intercepto (True)
    --positive restringe los coeficientes a positivos (False)
    --random-state fija la semilla (42)
    --test-size define la proporción de prueba (0.2)
    --experiment-name define el experimento MLflow
    Las opciones booleanas aceptan --no-*.
    """
    args = parse_args()
    run_training(
        test_size=args.test_size,
        random_state=args.random_state,
        polynomial_degree=args.degree,
        fit_intercept=args.fit_intercept,
        positive=args.positive,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()