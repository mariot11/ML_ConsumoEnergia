# Predicción del gasto anual en suministros del hogar (EPF 2025, INE)

Proyecto final del módulo de Machine Learning del bootcamp de Data Science.

**Objetivo:** predecir cuánto gasta al año un hogar español en suministros de su vivienda principal
(agua, electricidad, gas y otros combustibles) a partir de las características del hogar: vivienda,
composición, ubicación, tipo de calefacción, ingresos, etc.

**Utilidad de negocio:** estimar el gasto típico de un hogar sin conocer su consumo real, por ejemplo para
dimensionar ofertas comerciales, segmentar clientes o detectar zonas y perfiles con mayor carga energética.

## Resultados

Métrica principal: R². Se informan también RMSE y MAE en euros, que son más fáciles de interpretar.

**Validación cruzada (5 folds, solo sobre train, 66 variables):**

| Modelo | R² | RMSE (€) | MAE (€) |
|---|---|---|---|
| **HistGradientBoosting (final)** | **0.267** | 670.2 | 478.4 |
| MLP | 0.256 | 675.4 | 481.4 |
| Ridge | 0.247 | 679.6 | 486.8 |
| Random Forest | 0.244 | 680.6 | 488.9 |
| ElasticNet | 0.241 | 682.2 | 487.9 |

**Test (3.909 hogares, tocado una sola vez):**

| | R² | RMSE (€) | MAE (€) |
|---|---|---|---|
| Baseline (predecir siempre la media de train) | 0.000 | 837.1 | 583.6 |
| **Modelo final** | **0.275** | **712.7** | **485.1** |

El modelo reduce el error absoluto medio un 16.9 % frente al baseline.

**Cómo leer estos números**
- R² ≈ 0.27 significa que las variables disponibles explican algo más de una cuarta parte de la variabilidad
  del gasto. Se probaron varios modelos, transformaciones del target (log), distintos conjuntos de variables,
  otras imputaciones y tratamientos de outliers, y todos convergen al mismo valor: el límite parece estar
  en la información de la encuesta (no incluye consumo en kWh, tarifas ni clima detallado) y no en el modelo.
- El error típico es de unos 485 € sobre un gasto medio de ~1.390 €.
- El modelo tiende a la media: sobrestima a los hogares de gasto bajo y subestima a los de gasto alto
  (en el cuartil más alto, el gasto medio real es ≈ 2.470 € y el predicho ≈ 1.710 €).

Variables que más aportan: fuente de calefacción (`FUENCALE`), fuente de agua caliente (`FUENAGUA`),
tipo de edificio (`TIPOEDIF`), ingresos del hogar (`IMPEXAC`), superficie (`SUPERF`) y comunidad
autónoma (`CCAA`). Ver `docs/figuras/importancia_hgb.png`.

## Estructura del repositorio

```
.
├── data/
│   ├── raw/          # ficheros originales del INE (.tab) y diccionarios
│   ├── processed/    # datos tras el procesamiento (datos_limpios.csv)
│   ├── train/        # train.csv (80 %)
│   └── test/         # test.csv (20 %)
├── notebooks/        # exploración, limpieza/EDA y entrenamiento/evaluación
├── src/
│   ├── data_processing.py   # data/raw -> data/processed
│   ├── training.py          # entrena y guarda los modelos
│   └── evaluation.py        # evalúa en test y genera tablas y gráficos
├── models/
│   ├── trained_model_n.pkl  # un modelo entrenado por algoritmo
│   ├── final_model.pkl      # el mejor modelo (por CV)
│   ├── cv_results.csv       # comparativa de modelos en CV
│   └── model_info.json      # modelo final y variables que usa
├── docs/
│   ├── figuras/, resultados/
│   ├── presentacion/
│   └── memoria.md
└── README.md
```

## Datos

Encuesta de Presupuestos Familiares (EPF) 2025 del INE, microdatos decodificados:
- `EPFgastos_2025_decodificado.tab`: gasto por hogar y tipo de gasto (~1,96 M de registros).
- `EPFhogar_2025_decodificado.tab`: características de 19.703 hogares (136 variables).

**Target (`GASTO_SUMINISTROS`):** suma por hogar de los gastos de los códigos COICOP 04411, 04511, 04521,
04523, 04531, 04541, 04543, 04545 y 04548, dividiendo cada gasto entre el factor de elevación para quitar el efecto de la
ponderación a población. Quedan 19.544 hogares (159 no registran gasto en suministros). Media ≈ 1.393 €,
mediana ≈ 1.218 €, máximo ≈ 11.040 €, con asimetría positiva.

**Limpieza principal:**
- Eliminadas las columnas de gasto (`GAST*`), que filtrarían información del target, y las de
  identificación, muestreo y poca información.
- `SUPERF`, `EDADSP` y `NHABIT` pasan de texto ("300 metros o más") a número; `VALOREST = -9` pasa a nulo.
- Imputación y codificación se hacen dentro del Pipeline, ajustadas solo con train.

Los ficheros de `data/raw` no se incluyen si superan el límite de GitHub; se descargan de la web del INE.

## Metodología

1. **Separación:** 80 % train / 20 % test (`random_state=42`). El análisis exploratorio y todas las decisiones
   se toman solo con train; el test se usa una vez al final.
2. **Preprocesado (`ColumnTransformer`):** categóricas con "No aplica"/"No consta" como categoría propia o
   moda, más `OneHotEncoder(handle_unknown="ignore")`; numéricas con mediana e indicador de nulo.
3. **Modelos (5, todos en `Pipeline`):** Ridge, ElasticNet, Random Forest, MLP y HistGradientBoosting.
4. **Ajuste de hiperparámetros:** `GridSearchCV` con KFold (5, shuffle, `random_state=42`) en los modelos
   que mejor funcionaban (HistGradientBoosting y MLP). `training.py` guarda los parámetros ya ajustados y puede
   repetir la búsqueda con `--grid-search`.
5. **Selección del modelo final:** mayor R² medio en CV sobre train.
6. **Interpretación:** caída de R² al quitar cada variable (CV) e importancia por permutación en datos
   no usados para entrenar.

## Cómo ejecutarlo

Requisitos: Python 3.10+, `pandas`, `numpy`, `scikit-learn`, `matplotlib`.

```bash
pip install pandas numpy scikit-learn matplotlib

python src/data_processing.py   # data/raw  -> data/processed/datos_limpios.csv
python src/training.py          # genera train/test, models/*.pkl y cv_results.csv
python src/evaluation.py        # métricas en test -> docs/resultados/
```

Opciones útiles:
- `python src/training.py --grid-search` repite la búsqueda de hiperparámetros (tarda más).
- `python src/training.py --reducido` usa un subconjunto de 24 variables, con rendimiento prácticamente igual.
- `python src/evaluation.py --todos` evalúa también los otros cuatro modelos (solo informativo: no usar
  el test para elegir modelo).

**Nota sobre tamaño:** el pickle del Random Forest es muy grande (~285 MB) y GitHub rechaza ficheros de más de 100 MB.
Añádelo al `.gitignore` o reduce el modelo (`min_samples_leaf` mayor).

## Limitaciones y próximos pasos

- El techo de información de los datos limita el R² a ≈ 0.27; los errores son mayores en los extremos.
- La encuesta es de un año y no incluye consumo real, tarifas ni clima detallado.
- Posibles mejoras: ampliar con otros años de la EPF, modelar por separado electricidad, gas, agua y
  combustibles, y ofrecer rangos de gasto (regresión cuantílica) en lugar de un único valor.

## Autoría

Mario Tormo García · Bootcamp de Data Science · Noviembre 2026_