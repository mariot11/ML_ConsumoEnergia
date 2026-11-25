import io
import json
import zipfile
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
SAMPLE_DIR = Path("data/sample")
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

N_ROWS = 200

# 1. Muestra de EPFhogar_2025.tab
hogar_path = RAW_DIR / "EPFhogar_2025.tab"
if hogar_path.exists():
    df_hogar = pd.read_csv(hogar_path, sep="\t", nrows=N_ROWS, low_memory=False)
    out_hogar = SAMPLE_DIR / "EPFhogar_2025_sample.tab"
    df_hogar.to_csv(out_hogar, sep="\t", index=False)
    print(f"Guardado {out_hogar} ({len(df_hogar)} filas)")

# 2. Muestra desde EPFgastos_2025.zip (primer archivo de datos interno)
zip_path = RAW_DIR / "EPFgastos_2025.zip"
if zip_path.exists():
    with zipfile.ZipFile(zip_path, "r") as zf:
        data_files = [
            name for name in zf.namelist()
            if not name.endswith("/") and name.lower().endswith((".tab", ".tsv", ".csv", ".txt"))
        ]
        if data_files:
            inner_name = data_files[0]
            with zf.open(inner_name) as f:
                df_gastos = pd.read_csv(io.BytesIO(f.read()), sep="\t", nrows=N_ROWS, low_memory=False)
            out_gastos = SAMPLE_DIR / "EPFgastos_2025_sample.tab"
            df_gastos.to_csv(out_gastos, sep="\t", index=False)
            print(f"Guardado {out_gastos} ({len(df_gastos)} filas, desde {inner_name})")

# 3. Copia o recorte de archivos JSON de diseño de registro si existen
for json_path in sorted(RAW_DIR.glob("*.json")):
    with json_path.open("r", encoding="utf-8") as jf:
        data_json = json.load(jf)
    out_json = SAMPLE_DIR / json_path.name
    if isinstance(data_json, list):
        payload = data_json[:50]
    elif isinstance(data_json, dict):
        payload = dict(list(data_json.items())[:50])
    else:
        payload = data_json
    with out_json.open("w", encoding="utf-8") as of:
        json.dump(payload, of, ensure_ascii=False, indent=2)
    print(f"Guardado {out_json}")