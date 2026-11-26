import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TARGET = "GASTO_SUMINISTROS"
SEED = 42

# Columnas que el modelo final no usa (desgloses redundantes del hogar y binarias sin señal)
SIN_USO = [
    "NMIEM1", "NMIEM2", "NMIEM3", "NMIEM5", "NMIEM6", "NMIEM7", "NMIEM8", "NMIEM11", "NMIEM12", "NMIEM13",
    "NUMACTI", "NUMINACTI", "NUMESTU", "NUMNOESTU", "NHIJOSD",
    "CALEF", "OCUSP", "SITURESSP", "PENSIOINCAP", "OTROIN",
]
# Tratamiento de nulos: el resto de categóricas usa la moda
NO_APLICA = ["FUENCALE", "FUENAGUA", "JORNADASP", "SECTOR", "CONTRATO", "TIPOCONT", "TRABAJO"]
NO_CONSTA = ["OCUPA", "OCUPARED", "ACTESTB", "ACTESTBRED", "SITPROF", "SITSOCI", "SITSOCIRE"]
NUM_CON_NULOS = ["VALOREST", "NHABIT"]

# Hiperparámetros elegidos con GridSearchCV (CV 5 folds sobre train)
PARAMS_HGB = dict(
    learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0,
    max_iter=300, early_stopping=True, validation_fraction=0.1, random_state=SEED,
)


def construir_pipeline(X: pd.DataFrame, variables: list[str]) -> Pipeline:
    num = [c for c in variables if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in variables if c not in num]
    no_aplica = [c for c in cat if c in NO_APLICA]
    no_consta = [c for c in cat if c in NO_CONSTA]
    moda = [c for c in cat if c not in no_aplica + no_consta]
    num_nulos = [c for c in num if c in NUM_CON_NULOS]
    num_ok = [c for c in num if c not in num_nulos]

    def cat_pipe(imputer):
        return Pipeline([("imputer", imputer), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    bloques = [
        ("cat_no_aplica", cat_pipe(SimpleImputer(strategy="constant", fill_value="No_aplica")), no_aplica),
        ("cat_no_consta", cat_pipe(SimpleImputer(strategy="constant", fill_value="No_consta")), no_consta),
        ("cat_moda", cat_pipe(SimpleImputer(strategy="most_frequent")), moda),
        ("num_nulos", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                                ("scaler", StandardScaler())]), num_nulos),
        ("num_limpias", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                  ("scaler", StandardScaler())]), num_ok),
    ]
    prep = ColumnTransformer([b for b in bloques if b[2]], remainder="drop", sparse_threshold=0)
    return Pipeline([("prep", prep), ("model", HistGradientBoostingRegressor(**PARAMS_HGB))])


def main() -> None:
    df = pd.read_csv(ROOT / "data/processed/datos_limpios.csv")
    train, test = train_test_split(df, test_size=0.2, random_state=SEED)
    for carpeta, parte in (("train", train), ("test", test)):
        (ROOT / "data" / carpeta).mkdir(parents=True, exist_ok=True)
        parte.to_csv(ROOT / "data" / carpeta / f"{carpeta}.csv", index=False)

    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    variables = [c for c in X_train.columns if c not in SIN_USO]
    modelo = construir_pipeline(X_train, variables).fit(X_train[variables], y_train)

    ruta = ROOT / "models" / "final_model.pkl"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"train {len(train)} | test {len(test)} | {len(variables)} variables -> {ruta.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
