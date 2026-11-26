from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = "GASTO_SUMINISTROS"
SUMINISTROS = ["04411", "04511", "04521", "04523", "04531", "04541", "04543", "04545", "04548"]
ELIMINAR = ["NUMERO", "ANOENC", "TAMANO", "IMPUTGAS", "CLAVE", "CLATEO", "FACTOR",
            "IMPEXACPSP", "INTERINPSP", "INTERIN", "NMIEMSD", "NMIEMHU", "NMIEMIN"]


def solo_numero(serie: pd.Series) -> pd.Series:
    """'300 metros o más' -> 300 ; '085' -> 85."""
    return pd.to_numeric(serie.astype(str).str.strip().str.extract(r"(\d+)")[0], errors="coerce")


def main() -> None:
    gastos = pd.read_csv(ROOT / "data/raw/EPFgastos_2025_decodificado.tab", sep="\t", dtype={"CODIGO": str})
    hogar = pd.read_csv(ROOT / "data/raw/EPFhogar_2025_decodificado.tab", sep="\t")

    # Columnas con más de un 50 % de nulos fuera
    gastos = gastos.dropna(axis=1, thresh=len(gastos) / 2)
    hogar = hogar.dropna(axis=1, thresh=len(hogar) / 2)

    # Target: gasto anual del hogar en suministros, sin el efecto de la elevación a población
    g = gastos[gastos["CODIGO"].isin(SUMINISTROS)].copy()
    g[TARGET] = g["GASTMON"] / g["FACTOR"]
    target = g.groupby("NUMERO", as_index=False)[TARGET].sum()

    df = target.merge(hogar, on="NUMERO", how="left")

    # Limpieza
    for col in ["SUPERF", "EDADSP", "NHABIT"]:
        df[col] = solo_numero(df[col])
    df["VALOREST"] = df["VALOREST"].replace(-9, np.nan)
    gast = [c for c in df.columns if c.startswith("GAST") and c != TARGET]
    norden = [c for c in df.columns if c.startswith("NORDEN")]
    df = df.drop(columns=gast + norden + ELIMINAR)

    salida = ROOT / "data/processed/datos_limpios.csv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida, index=False)
    print(f"{salida.relative_to(ROOT)}: {df.shape[0]} hogares x {df.shape[1]} columnas")


if __name__ == "__main__":
    main()
