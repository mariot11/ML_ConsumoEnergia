import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
TARGET = "GASTO_SUMINISTROS"
SALIDA = ROOT / "docs" / "resultados"


def metricas(y, pred) -> dict:
    return {"r2": r2_score(y, pred), "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": mean_absolute_error(y, pred)}


def main() -> None:
    train = pd.read_csv(ROOT / "data/train/train.csv")
    test = pd.read_csv(ROOT / "data/test/test.csv")
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]
    with open(ROOT / "models/final_model.pkl", "rb") as f:
        modelo = pickle.load(f)
    pred = modelo.predict(X_test)

    tabla = pd.DataFrame([
        {"modelo": "baseline (media de train)", **metricas(y_test, np.full(len(y_test), train[TARGET].mean()))},
        {"modelo": "modelo final", **metricas(y_test, pred)},
    ])

    # Error por cuartiles del gasto real
    d = pd.DataFrame({"real": y_test.to_numpy(), "pred": pred})
    d["tramo"] = pd.qcut(d["real"], 4, labels=["Q1 (más bajo)", "Q2", "Q3", "Q4 (más alto)"])
    d["error"] = d["pred"] - d["real"]
    g = d.groupby("tramo", observed=True)
    tramos = pd.DataFrame({"gasto_real_medio": g["real"].mean(), "gasto_predicho_medio": g["pred"].mean(),
                           "sesgo_medio": g["error"].mean(), "mae": g["error"].apply(lambda e: e.abs().mean())})

    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA / "metricas_test.csv", index=False)
    tramos.to_csv(SALIDA / "error_por_tramo_test.csv")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    lim = float(max(d["real"].max(), d["pred"].max()))
    hb = ax[0].hexbin(d["real"], d["pred"], gridsize=35, bins="log", cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax[0], label="Hogares (escala log)")
    ax[0].plot([0, lim], [0, lim], "r--", lw=1, label="Predicción perfecta")
    ax[0].set(xlabel="Gasto real (€/año)", ylabel="Gasto predicho (€/año)", title="Real vs predicho (test)")
    ax[0].legend(loc="upper left")
    ax[1].hist(d["error"], bins=80, color="#4c72b0")
    ax[1].axvline(0, color="r", ls="--", lw=1)
    ax[1].set(xlabel="Error = predicho − real (€)", ylabel="Hogares", title="Distribución del error (test)")
    fig.tight_layout()
    fig.savefig(SALIDA / "real_vs_predicho_test.png", dpi=150)

    pd.options.display.float_format = "{:,.3f}".format
    print(f"Test: {len(test)} hogares\n\n{tabla.to_string(index=False)}\n\n{tramos.to_string()}")


if __name__ == "__main__":
    main()
