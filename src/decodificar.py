from pathlib import Path
import io
import json
import zipfile
import pandas as pd

RAW_DIR = Path("data/raw")


def _normalizar_codigo(val, longitud=None):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if longitud and len(s) < longitud:
        if s.startswith("-"):
            s = "-" + s[1:].zfill(max(0, longitud - 1))
        else:
            s = s.zfill(longitud)
    return s


def decodificar_tab(df: pd.DataFrame, dr_path: str | Path, reemplazar: bool = True) -> pd.DataFrame:
    """
    Decodifica las columnas con `codelist_ref` usando el JSON de diseño de registro.
    Si `reemplazar=True`, sustituye el código por su etiqueta string;
    si `reemplazar=False`, crea una columna adicional `<COL>_desc`.
    """
    with open(dr_path, "r", encoding="utf-8") as f:
        dr = json.load(f)

    out = df.copy()
    layout_map = {item["name"]: item for item in dr.get("layout", [])}
    codelists = dr.get("codelists", {})

    for col in out.columns:
        spec = layout_map.get(col, {})
        ref = spec.get("codelist_ref")
        if not ref or ref not in codelists:
            continue

        mapping = codelists[ref]
        claves_sin_signo = [k for k in mapping.keys() if not k.startswith("-")]
        longitudes = {len(k) for k in claves_sin_signo}
        longitud_objetivo = next(iter(longitudes)) if len(longitudes) == 1 else spec.get("length")

        serie_norm = out[col].apply(lambda v: _normalizar_codigo(v, longitud_objetivo))
        etiquetas = serie_norm.map(mapping)

        faltantes = etiquetas.isna() & serie_norm.notna()
        if faltantes.any():
            mapa_sin_ceros = {k.lstrip("0") or "0": v for k, v in mapping.items()}
            etiquetas.loc[faltantes] = serie_norm.loc[faltantes].apply(
                lambda x: mapa_sin_ceros.get(x.lstrip("0") or "0", x)
            )

        resultado = etiquetas.fillna(serie_norm)
        if reemplazar:
            out[col] = resultado
        else:
            out[f"{col}_desc"] = resultado

    return out


def _leer_tab_desde_zip(zip_path: Path) -> pd.DataFrame:
    """Lee el primer fichero de datos (.tab/.tsv/.csv/.txt) dentro de un ZIP."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        candidatos = [
            name for name in zf.namelist()
            if not name.endswith("/") and name.lower().endswith((".tab", ".tsv", ".csv", ".txt"))
        ]
        if not candidatos:
            raise FileNotFoundError(f"No se encontró ningún .tab/.csv dentro de {zip_path}")
        with zf.open(candidatos[0]) as f:
            return pd.read_csv(io.BytesIO(f.read()), sep="\t", dtype=str, low_memory=False)


def procesar_y_guardar(raw_dir: Path = RAW_DIR) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Procesar EPFhogar_2025 desde data/raw/ y guardar en data/raw/
    hogar_tab = raw_dir / "EPFhogar_2025.tab"
    hogar_json = raw_dir / "dr_EPFhogar_2025.json"
    if hogar_tab.exists() and hogar_json.exists():
        df_hogar = pd.read_csv(hogar_tab, sep="\t", dtype=str, low_memory=False)
        df_hogar_dec = decodificar_tab(df_hogar, hogar_json, reemplazar=True)
        out_hogar = raw_dir / "EPFhogar_2025_decodificado.tab"
        df_hogar_dec.to_csv(out_hogar, sep="\t", index=False)
        print(f"Guardado: {out_hogar} ({len(df_hogar_dec)} filas)")

    # 2. Procesar EPFgastos_2025 (desde .tab o .zip en data/raw/) y guardar en data/raw/
    gastos_tab = raw_dir / "EPFgastos_2025.tab"
    gastos_zip = raw_dir / "EPFgastos_2025.zip"
    gastos_json = raw_dir / "dr_EPFgastos_2025.json"
    if gastos_json.exists() and (gastos_tab.exists() or gastos_zip.exists()):
        if gastos_tab.exists():
            df_gastos = pd.read_csv(gastos_tab, sep="\t", dtype=str, low_memory=False)
        else:
            df_gastos = _leer_tab_desde_zip(gastos_zip)

        df_gastos_dec = decodificar_tab(df_gastos, gastos_json, reemplazar=False)
        out_gastos = raw_dir / "EPFgastos_2025_decodificado.tab"
        df_gastos_dec.to_csv(out_gastos, sep="\t", index=False)
        print(f"Guardado: {out_gastos} ({len(df_gastos_dec)} filas)")


if __name__ == "__main__":
    procesar_y_guardar()