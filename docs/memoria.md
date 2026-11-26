# Memoria del proyecto: Predicción del gasto anual en suministros del hogar

**Datos:** Encuesta de Presupuestos Familiares (EPF) 2025 (INE) · **Tipo de problema:** Regresión supervisada

**Modelo final:** `HistGradientBoostingRegressor` · **Resultado en test:** $R^2 = 0.275$ · $\text{MAE} = 485.1\text{ €}$ (baseline: $583.6\text{ €}$)

## 1. Planteamiento y resumen del proyecto

El objetivo de este proyecto es estimar cuánto gasta al año un hogar español en suministros de su vivienda principal (agua, electricidad, gas y otros combustibles) a partir de sus características socioeconómicas y de vivienda, sin conocer su consumo real.

A lo largo del trabajo partí de los dos ficheros principales del INE (gastos y hogares), construí la variable objetivo, limpié y exploré más de cien variables y comparé cinco algoritmos distintos mediante validación cruzada. El mejor modelo, un **`HistGradientBoostingRegressor`** ajustado con `GridSearchCV`, alcanza **$R^2 = 0.275$**, **$\text{RMSE} = 712.7\text{ €}$** y **$\text{MAE} = 485.1\text{ €}$** en el conjunto de test (evaluado una sola vez), lo que supone reducir el error absoluto medio en un **16.9 %** frente a predecir siempre la media.

La principal conclusión del proyecto es que **existe un techo de predictibilidad en torno a $R^2 \approx 0.27$** con la información disponible en la encuesta. Probé a cambiar de algoritmo, reducir o ampliar variables, imputar de formas más complejas, transformar el target o eliminar *outliers*, y todas las variaciones se movieron dentro del ruido propio de la validación cruzada ($\approx 0.01$). Por tanto, el modelo funciona bien para estimar el gasto típico según el perfil del hogar, pero no permite predecir con exactitud el gasto individual de una vivienda concreta.

## 2. Organización del repositorio y preparación inicial

El primer obstáculo técnico fue que uno de los ficheros originales superaba el límite de 100 MB de GitHub y no me dejaba subirlo. Decidí comprimirlo en local, añadir `data/raw` y `data/processed` al `.gitignore` para mantenerlos solo en mi equipo y, como el enunciado pedía los datos originales o una muestra recortada, subí una versión reducida en `data/sample`. También preparé un entorno virtual (`.venv`) con su `requirements.txt` para asegurar la reproducibilidad.

Para trabajar con los datos de partida (`EPFgastos_2025` y `EPFhogar_2025`), me encontré con que las variables categóricas venían codificadas mediante diccionarios JSON (`dr_EPFgastos_2025.json` y `dr_EPFhogar_2025.json`). Con ayuda de IA las decodifiqué para hacerlas legibles y poder tomar decisiones fundamentadas sobre qué columnas conservar o descartar (generando los ficheros `_decodificado.tab`):

| Fichero | Contenido | Dimensiones |
|---|---|---|
| `EPFgastos_2025_decodificado.tab` | Un registro por hogar y tipo de gasto (374 códigos) | 1 956 764 filas × 15 columnas |
| `EPFhogar_2025_decodificado.tab` | Un registro por hogar (vivienda, composición, ingresos, ubicación...) | 19 703 filas × 136 columnas |

### 2.1 Construcción del target y el hallazgo del factor de elevación

Ambos ficheros comparten la columna `NUMERO` que identifica a cada hogar: en el de hogares aparece una única vez, mientras que en el de gastos aparece tantas veces como partidas tenga registradas. Para crear el objetivo (`GASTO_SUMINISTROS`), filtré los 9 códigos ECOICOP correspondientes a suministros de la vivienda principal y los agrupé por `NUMERO`:

| Código ECOICOP | Descripción oficial (EPF 2025) |
|---|---|
| `04411` | Suministro de agua (vivienda principal, garajes y trasteros ligados a la vivienda principal) |
| `04511` | Electricidad (vivienda principal, garajes y trasteros ligados a la vivienda principal) |
| `04521` | Gas ciudad y natural (vivienda principal) |
| `04523` | Gas licuado (vivienda principal) |
| `04531` | Combustibles líquidos (vivienda principal) |
| `04541` | Carbón y turba (vivienda principal) |
| `04543` | Combustibles sólidos a base de madera (leña, pellets…) (vivienda principal) |
| `04545` | Carbón vegetal (vivienda principal) |
| `04548` | Otros combustibles sólidos (vivienda principal) |

Al revisar los datos más adelante —justo cuando preparaba el contraste U de Mann-Whitney en el análisis exploratorio— me di cuenta de un error importante: había sumado directamente los gastos sin tener en cuenta que la EPF los multiplica por `FACTOR` (el peso muestral de cada hogar en la población). Dividí cada partida entre `FACTOR` (`GASTMON / FACTOR`) y volví a ejecutar todo el flujo desde ese punto. Con esta corrección, las cifras pasaron a ser totalmente coherentes (mediana en torno a $1\,218\text{ €/año}$):

| Estadístico | Valor |
|---|---|
| Hogares válidos | 19 544 |
| Media | 1 393 € |
| Mediana | 1 218 € |
| Desviación típica | 795 € |
| Percentiles 25 / 75 | 873 € / 1 721 € |
| Máximo | 11 040 € |
| Hogares con gasto 0 | 15 |
| Asimetría | $\approx 1.8$ (cola derecha) |

### 2.2 Hogares que desaparecen al agrupar

Al unir el target con el fichero de hogares comprobé que faltaban **159 hogares** respecto a los 19 703 iniciales (quedando 19 544). Hice varias comprobaciones para asegurarme de que no se estaban perdiendo por culpa de la limpieza de nulos o del `groupby`, y verifiqué que esos 159 hogares sí están en el fichero original, pero no tienen ningún registro en los códigos de suministros (solo declaran gastos en partidas como alquiler real `04110`, alquiler imputado `04210` o mantenimiento `04311`...). Al no tener gasto observable en suministros, quedan fuera del problema. En cambio, los 15 hogares con gasto exactamente igual a $0\text{ €}$ sí se conservaron.

## 3. Limpieza de datos

Antes de dividir en *train* y *test*, apliqué únicamente las reglas de limpieza que no dependen de estadísticas de los datos para evitar cualquier fuga de información (*data leakage*):

* **Eliminación de columnas vacías o sin señal:** descarté variables con más de un 50 % de nulos y variables casi constantes como `NMIEMSD`, `NMIEMHU` y `NMIEMIN` ($> 99.8\text{ \%}$ de ceros), así como `IMPEXACPSP`, `INTERINPSP` e `INTERIN` (plagadas de valores negativos o "No consta").
* **Prevención de fuga de información:** eliminé todas las columnas `GAST*` (`GASTOT`, `GASTOTGR01-13`, `GASTMON`, `GASTNOM1-4`...), ya que contienen o derivan del gasto que quiero predecir, además de identificadores y variables de muestreo (`NUMERO`, `ANOENC`, `TAMANO`, `IMPUTGAS`, `CLAVE`, `CLATEO`, `FACTOR`, `NORDEN*`).
* **Conversión de formatos y códigos especiales:** transformé a número `SUPERF`, `EDADSP` y `NHABIT` (que venían como texto tipo *"300 metros o más"* o *"Persona de 85 o más años"*) y reemplacé el código `-9` de `VALOREST` ("no aplica", presente en el 18.5 % de los hogares) por `NaN`.
* **Nulos con significado propio:** aunque `FUENCALE` (fuente de calefacción) tenía un 30.9 % de nulos, al cruzarla con `CALEF` comprobé que todos correspondían a hogares sin calefacción (`CALEF = No`). Lo mismo sucedía con `JORNADASP` y `OCUSP` en personas no ocupadas. Por ello decidí no borrarlas y tratar sus nulos como una categoría explícita (`No_aplica`).

Tras este proceso obtuve `datos_limpios.csv` con **19 544 hogares y 87 columnas** (86 variables explicativas + target).

## 4. Análisis exploratorio (EDA sobre *train*)

Dividí los datos en **80 % *train* (15 635 hogares)** y **20 % *test* (3 909 hogares)** con `random_state = 42`, reservando el *test* únicamente para la evaluación final. Todo el análisis exploratorio se realizó exclusivamente sobre *train*:

1. **Variables numéricas:** revisé sus histogramas (observando fuertes asimetrías en `VALOREST`, `IMPEXAC` y `SUPERF`) y calculé correlaciones de Pearson y Spearman. Esto permitió detectar grupos altamente colineales ($|r|$ entre 0.93 y 0.99), como las distintas variables de tamaño y composición del hogar (`NMIEM1`, `UC2`, `PF2TEO`, `NMIEMB`...), descartando las redundantes.
2. **Variables binarias:** las convertí a formato `0/1` y vi que muchas estaban extremadamente desbalanceadas. Antes de eliminarlas, apliqué el test **U de Mann-Whitney** para comprobar si aun así diferenciaban significativamente el gasto.
3. **Variables categóricas:** analicé media, mediana y desviación típica por categoría, apoyándome en ANOVA, Kruskal-Wallis y especialmente en **eta cuadrado ($\eta^2$)** para ordenar su impacto real, ya que con más de 15 000 registros casi cualquier *p-valor* resulta significativo.

El EDA mostró que las variables con mayor capacidad explicativa son la comunidad autónoma (`CCAA`), las fuentes de energía de calefacción y agua caliente (`FUENCALE`, `FUENAGUA`), el tipo de edificio y vivienda, la superficie (`SUPERF`) y la tipología del hogar (`TIPHOGAR1`, $\eta^2 = 0.074$), mientras que variables demográficas como la nacionalidad (`NACIONASP`, $\eta^2 = 0.00002$) apenas influyen.

## 5. Preprocesado y validación

Para evitar fugas de datos durante los experimentos, encapsulé todas las transformaciones dentro de un `Pipeline` con `ColumnTransformer`, evaluado mediante validación cruzada de 5 particiones (`KFold(n_splits=5, shuffle=True, random_state=42)`):

* **Categóricas con nulo informativo:** imputación con `No_aplica` (por ejemplo `FUENCALE`, `FUENAGUA`, `JORNADASP`, `CONTRATO`...) o `No_consta` (`OCUPA`, `ACTESTB`, `SITPROF`...) seguida de `OneHotEncoder`.
* **Resto de categóricas:** imputación por moda y `OneHotEncoder` (`handle_unknown="ignore"`).
* **Numéricas:** imputación por mediana y escalado estándar (`StandardScaler`), añadiendo además un indicador binario de valor faltante en `VALOREST` y `NHABIT` (ya que `VALOREST` falta sistemáticamente en hogares en alquiler).

En esta validación cruzada, el modelo **baseline** (predecir siempre la media) obtiene $R^2 \approx 0$ y $\text{MAE} \approx 573\text{ €}$.

## 6. Experimentos y decisiones de modelado

Comencé entrenando una regresión **Ridge** como punto de partida, obteniendo $R^2 \approx 0.25$ en validación cruzada. Dado que la desviación típica entre particiones rondaba $0.012$, tomé como criterio que **diferencias inferiores a $\sim 0.01$ entran dentro del ruido muestral**. A partir de ahí probé distintas vías para mejorar el rendimiento o simplificar el modelo:

### 6.1 Simplificación de columnas e imputación iterativa

* **Eliminación de redundancias y versiones agrupadas:** al quitar variables codificadas por duplicado y sustituir ciertas categóricas detalladas por sus versiones agrupadas (`ESTUDREDSP`, `OCUPARED`, `ACTESTBRED`...), pasé de 488 columnas codificadas a **182** manteniendo intacto el rendimiento ($R^2 = 0.2531$, $\text{MAE} = 485.1\text{ €}$). En cambio, agrupar `CCAA` o `TIPHOGAR1` sí restaba señal, por lo que las conservé completas.
* **Imputación multivariante (`IterativeImputer`):** probé a imputar `VALOREST` de forma iterativa, pero el cambio en $R^2$ fue de apenas $0.0002$ y carecía de sentido imputar un valor de propiedad a hogares en alquiler, por lo que descarté esta opción.

### 6.2 Transformaciones logarítmicas

Probé aplicar `log1p` tanto a variables explicativas asimétricas (`VALOREST`, `IMPEXAC`, `SUPERF`) como al propio target:

* En las variables explicativas el efecto fue neutro con Ridge e innecesario con árboles.
* Aplicar `log1p` al target redujo el $\text{MAE}$ unos $8\text{–}11\text{ €}$ (al acercar la predicción a la mediana), pero hizo caer el $R^2$ de $0.2668$ a $0.2190$ en `HistGradientBoosting` (y a $0.2568$ incluso tras corregir el sesgo exponencial), por lo que mantuve el target en euros originales.

### 6.3 Selección de variables

Analicé cuánto caía el $R^2$ en CV al retirar variables una a una y combiné ese criterio con los tamaños de efecto del EDA:

* Con Ridge bastaban **22 variables** para mantener el $R^2$ de partida ($0.2520$).
* Al trasladar esa reducción a `HistGradientBoosting`, vi que los árboles aprovechaban ligeramente mejor el conjunto amplio ($R^2 = 0.2693$ con 66 variables frente a $0.2686$ con la selección reducida de 24 variables del EDA). Finalmente mantuve el conjunto depurado de **66 variables** (descartando las 20 columnas claramente redundantes), dejando la lista compacta como alternativa equivalente.

### 6.4 Comparación de algoritmos

Evalué cinco modelos distintos bajo exactamente el mismo preprocesado y esquema de validación cruzada:

| Modelo | $R^2$ (CV) | $\pm$ | $\text{RMSE}$ (€) | $\text{MAE}$ (€) |
|---|---|---|---|---|
| **`HistGradientBoostingRegressor`** | **0.267** | 0.009 | **670.2** | **478.4** |
| `MLPRegressor` (red neuronal) | 0.256 | 0.010 | 675.4 | 481.4 |
| `Ridge` | 0.247 | 0.012 | 679.6 | 486.8 |
| `RandomForestRegressor` | 0.244 | 0.007 | 680.6 | 488.9 |
| `ElasticNet` | 0.241 | 0.008 | 682.2 | 487.9 |

Los modelos lineales se quedan cortos ante relaciones no lineales, mientras que el `RandomForest` clásico sobreajusta ($R^2 \approx 0.66$ en *train* frente a $0.24$ en validación). Los dos mejores candidatos fueron **`HistGradientBoostingRegressor`** y **`MLPRegressor`**. Tras ajustarlos con `GridSearchCV`, el modelo ganador fue `HistGradientBoostingRegressor` (`learning_rate=0.05`, `max_leaf_nodes=15`, `min_samples_leaf=100`, `l2_regularization=1.0`, `early_stopping=True` con 10 % de validación y 300 iteraciones máximas).

### 6.5 Tratamiento de *outliers* en el target

También comprobé si recortar o eliminar los hogares con mayor gasto mejoraba el modelo:

* Eliminar el 1 % o el 5 % más alto solo en *train* empeoraba el $R^2$ en validación ($0.2584$ y $0.2109$), porque el modelo dejaba de aprender los consumos altos reales y los subestimaba aún más.
* Eliminarlos tanto en *train* como en validación reducía artificialmente el $\text{RMSE}$ y el $\text{MAE}$, pero bajaba el $R^2$ y falseaba la evaluación al dejar fuera los casos reales más difíciles, por lo que descarté cualquier recorte de *outliers*.

## 7. Evaluación final en *test* e interpretación

Una vez cerrado todo el flujo, entrené el `HistGradientBoostingRegressor` definitivo con todo el conjunto de *train* y lo evalué una sola vez sobre los 3 909 hogares de *test*:

| Modelo | $R^2$ | $\text{RMSE}$ (€) | $\text{MAE}$ (€) |
|---|---|---|---|
| Baseline (media de *train*) | 0.000 | 837.1 | 583.6 |
| **Modelo final (`HistGradientBoosting`)** | **0.275** | **712.7** | **485.1** |

El resultado en *test* ($R^2 = 0.275$) confirma la estabilidad respecto a la validación cruzada ($0.267$), logrando una reducción del **16.9 % en el MAE**. El ligero aumento del $\text{RMSE}$ en *test* se debe a que esa partición contiene hogares con gastos especialmente altos (incluidos dos casos por encima de $11\,000\text{ €}$).

### 7.1 Comportamiento por tramos de gasto y variables más influyentes

Al analizar los errores por cuartiles de gasto real, se observa claramente el efecto de **regresión a la media**:

| Tramo de gasto real | Hogares | Gasto real medio | Gasto predicho medio | Sesgo | $\text{MAE}$ |
|---|---|---|---|---|---|
| Q1 (más bajo) | 978 | 619 € | 1 152 € | +533 € | 534 € |
| Q2 | 977 | 1 031 € | 1 297 € | +266 € | 307 € |
| Q3 | 977 | 1 430 € | 1 439 € | +9 € | 259 € |
| Q4 (más alto) | 977 | 2 470 € | 1 710 € | -760 € | 840 € |

El modelo acierta con mucha precisión en los hogares de consumo medio (Q3, sesgo de apenas $+9\text{ €}$), pero tiende a sobrestimar los consumos muy bajos y a subestimar los muy elevados.

En cuanto a la importancia de las variables (medida tanto por caída de $R^2$ al eliminar cada variable en CV como por permutación), el gasto energético viene gobernado por factores físicos, económicos y climáticos coherentes con el dominio:

1. **`FUENCALE`** (fuente de energía de la calefacción) y **`FUENAGUA`** (agua caliente).
2. **`TIPOEDIF`** (tipo de edificio) y **`SUPERF`** (superficie de la vivienda).
3. **`IMPEXAC`** (nivel de ingresos del hogar) y **`CCAA`** (comunidad autónoma / zona climática).

## 8. Conclusiones y líneas futuras

* **Validez del modelo:** se ha construido un flujo reproducible (`src/data_processing.py`, `src/training.py`, `src/evaluation.py`) que reduce casi un 17 % el error medio respecto al baseline sin fuga de datos.
* **Limitación estructural de los datos:** cerca del 73 % de la varianza del gasto queda sin explicar porque la EPF no recoge variables determinantes como los kWh reales consumidos, la tarifa contratada, el aislamiento térmico de la vivienda ni los hábitos diarios de uso.
* **Posibles mejoras futuras:** incorporar datos de otros años de la EPF para ampliar muestra, añadir variables climáticas externas por provincia, modelar por separado cada suministro (electricidad, gas, agua) antes de sumarlos, o emplear regresión cuantílica para ofrecer intervalos de gasto en lugar de una estimación puntual.

---

## 9. Diccionario de variables mencionadas

A continuación se detalla el significado oficial (según los diseños de registro `dr_EPFgastos_2025.json` y `dr_EPFhogar_2025.json` del INE) de todas las columnas citadas en esta memoria:

### 9.1 Identificadores, muestreo y variables de gasto

| Variable | Fichero | Descripción |
|---|---|---|
| `ANOENC` | Ambos | Año de la encuesta ($\ge 2024$). |
| `NUMERO` | Ambos | Número secuencial que identifica a cada hogar en el fichero. |
| `CLAVE` | Hogar | Clave de colaboración efectiva del hogar (`Primera` / `Segunda`). |
| `CLATEO` | Hogar | Clave de colaboración teórica del hogar. |
| `FACTOR` | Ambos | Factor de elevación espacial / poblacional del hogar. |
| `CODIGO` | Gastos | Código ECOICOP de 5 dígitos del bien o servicio consumido. |
| `GASTMON` | Ambos | Importe total del gasto monetario anual, elevado temporal y poblacionalmente. |
| `GASTOT` | Hogar | Importe total del gasto anual del hogar (monetario y no monetario), elevado. |
| `GASTOTGR01`–`13` | Hogar | Gasto total anual del hogar desglosado por cada uno de los 13 grupos ECOICOP. |
| `GASTNOM1`–`5` | Ambos | Partidas de gasto no monetario (autoconsumo, autosuministro, salario en especie y alquiler imputado). |
| `IMPUTGAS` | Hogar | Porcentaje de imputación del gasto total del hogar. |

### 9.2 Territorio y características de la vivienda

| Variable | Descripción |
|---|---|
| `CCAA` | Comunidad autónoma de residencia del hogar. |
| `NUTS1` | Región / gran agrupación territorial NUTS-1 (Noroeste, Noreste, Madrid, Central, Este, Sur, Canarias). |
| `TIPOEDIF` | Tipo de edificio de la vivienda principal (unifamiliar independiente, adosada, edificio de $<10$ o $\ge 10$ viviendas, otros). |
| `SUPERF` | Superficie útil de la vivienda principal en metros cuadrados. |
| `NHABIT` | Número de habitaciones de la vivienda principal. |
| `VALOREST` | Valor estimado de la vivienda principal (solo aplica a hogares en propiedad). |
| `CALEF` | Disposición de calefacción en la vivienda principal (`Sí` / `No`). |
| `FUENCALE` | Fuente de energía principal para calefacción (electricidad, gas natural, gas licuado, combustibles líquidos, sólidos, otras). |
| `FUENAGUA` | Fuente de energía principal para el agua caliente en la vivienda principal. |

### 9.3 Composición del hogar e ingresos

| Variable | Descripción |
|---|---|
| `TAMANO` | Tamaño del hogar codificado por tramos (de 1 a 6 o más personas). |
| `NMIEMB` | Número total de miembros del hogar. |
| `NMIEM1` | Número de miembros del hogar de 14 o más años. |
| `NMIEMSD` | Número de miembros del hogar pertenecientes al servicio doméstico. |
| `NMIEMHU` | Número de miembros del hogar que son huéspedes. |
| `NMIEMIN` | Número de miembros del hogar que son invitados. |
| `UC2` | Tamaño equivalente del hogar según la escala OCDE modificada. |
| `PF2TEO` | Número de libretas de cuentas individuales teóricas del hogar. |
| `TIPHOGAR1` | Tipo de hogar según la primera clasificación del INE (estructura familiar y edad de los miembros). |
| `TIPHOGAR7` | Tipo de hogar según la séptima clasificación del INE (agrupación alternativa). |
| `TIPHOGAR8` | Tipo de hogar según la octava clasificación del INE (versión reducida derivada de `TIPHOGAR1`). |
| `IMPEXAC` | Importe exacto de los ingresos mensuales netos totales del hogar. |
| `INTERIN` | Intervalo de ingresos mensuales netos totales del hogar. |

### 9.4 Características del sustentador principal (SP)

| Variable | Descripción |
|---|---|
| `NORDEN*` | Variables con el número de orden de miembros del hogar (`NORDENSP`, `NORDENCOSP`, `NORDENP1SP`, `NORDENP2SP`). |
| `EDADSP` | Edad del sustentador principal. |
| `NACIONASP` | Nacionalidad del sustentador principal (`Sólo española`, `Sólo extranjera`, `Española y extranjera`). |
| `ESTUDREDSP` | Nivel de formación del sustentador principal (clasificación reducida en 4 niveles). |
| `SITUREDSP` | Principal situación en la actividad del sustentador principal (`Activo` / `Inactivo`). |
| `JORNADASP` | Tipo de jornada laboral del sustentador principal (`Completa` / `Parcial`). |
| `OCUSP` | Indicador de si el sustentador principal está ocupado (`Ocupado` / `No ocupado`). |
| `IMPEXACPSP` | Importe exacto de los ingresos mensuales netos del sustentador principal. |
| `INTERINPSP` | Intervalo de ingresos mensuales netos totales del sustentador principal. |
| `OCUPA` | Ocupación del sustentador principal según la clasificación CON-2011. |
| `OCUPARED` | Ocupación del sustentador principal (clasificación reducida en 5 grupos). |
| `ACTESTB` | Actividad económica del establecimiento donde trabaja/trabajó el SP (CNAE-2009). |
| `ACTESTBRED` | Actividad económica del establecimiento (clasificación reducida: agricultura, industria/construcción, servicios). |
| `SITPROF` | Situación profesional del sustentador principal (asalariado, autónomo, empleador, etc.). |
| `CONTRATO` | Tenencia de contrato laboral del sustentador principal (`Con contrato` / `Sin contrato`). |
| `SITSOCI` | Situación socioeconómica detallada del sustentador principal. |
| `SITSOCIRE` | Situación socioeconómica del sustentador principal (clasificación reducida). |