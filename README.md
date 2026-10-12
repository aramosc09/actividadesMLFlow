# Experimentos de calidad del vino

Script de regresión lineal polinómica para predecir `quality`. Registra parámetros, métricas, el archivo `metrics.json` y el modelo en MLflow. Los CSV deben estar en `wine+quality/`. (https://archive.ics.uci.edu/dataset/186/wine+quality)

## Requisitos

Python 3.13 y Conda o `venv`. Desde la raíz del proyecto, instala las dependencias fijadas en `requirements.txt` con una de estas opciones.

### Conda

```bash
conda create -n wine-mlflow python=3.13 pip -y
conda activate wine-mlflow
python -m pip install -r requirements.txt
```

### venv

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Entrenar

Desde la raíz del proyecto, usa la misma URI para registrar y consultar las corridas:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python train.py
```

El script usa valores por defecto y registra en `TC5061_Wine_Quality_Experiments`. Para cambiar parámetros:

```bash
python train.py --degree 2 --fit-intercept --no-positive --random-state 42 --experiment-name TC5061_Wine_Quality_Experiments
```

Opciones disponibles: `--degree` (1, 2 o 3), `--fit-intercept`/`--no-fit-intercept`, `--positive`/`--no-positive`, `--random-state`, `--test-size` y `--experiment-name`. La semilla predeterminada es 42.

## Consultar corridas

En la misma raíz del proyecto y con la variable `MLFLOW_TRACKING_URI` configurada, inicia la interfaz:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Abre `http://127.0.0.1:5000` y selecciona el experimento deseado. Si el puerto 5000 está ocupado, cambia `--port` a `5001`. La base SQLite y los artefactos se generan localmente; no se requiere un servidor remoto.