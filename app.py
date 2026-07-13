"""
App de Streamlit — Preproceso + EDA para Marketing Mix Modeling (Meridian)
Ejecuta:  streamlit run app.py
Objetivo: cargar la base, elegir KPI/medios/control, revisar salud de datos con
alertas y generar el bloque de código para pegar en el notebook de Colab.
"""
import io
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import streamlit as st

sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------------
# FUNCIONES DE APOYO (reutilizadas del notebook, ya validadas)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def leer_datos(contenido: bytes, nombre: str) -> pd.DataFrame:
    buf = io.BytesIO(contenido)
    if nombre.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    return pd.read_csv(buf)

def _racha_max_ceros(s: pd.Series) -> int:
    best = cur = 0
    for v in (s == 0).astype(int).values:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best

def _picos(s: pd.Series, f: float):
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    lo, hi = q1 - f * iqr, q3 + f * iqr
    n = int(((s < lo) | (s > hi)).sum())
    return n, round(n / len(s) * 100, 1)

def _ccf_best(x: pd.Series, y: pd.Series, max_lag: int = 8):
    best_lag, best_r = 0, 0.0
    for L in range(0, max_lag + 1):
        r = x.shift(L).corr(y, method="spearman")
        if pd.notna(r) and abs(r) > abs(best_r):
            best_r, best_lag = r, L
    return best_lag, round(best_r, 3)

def construir_tablero(df, media, kpi_col, iqr_factor):
    y = df[kpi_col]
    mi = mutual_info_regression(
        df[media].fillna(df[media].median()).values, y.values, random_state=0)
    soi = (df[media].sum() / df[media].sum().sum() * 100).round(1)
    filas = []
    for i, c in enumerate(media):
        npk, _ = _picos(df[c], iqr_factor)
        blag, bcorr = _ccf_best(df[c], y)
        filas.append({
            "canal": c,
            "SOI_%": float(soi[c]),
            "%ceros": round((df[c] == 0).mean() * 100, 1),
            "racha_ceros": _racha_max_ceros(df[c]),
            "picos": npk,
            "corr_KPI": round(df[c].corr(y, method="spearman"), 3),
            "MI_KPI": round(float(mi[i]), 3),
            "mejor_lag": blag,
            "corr_en_lag": bcorr,
        })
    return pd.DataFrame(filas)

def generar_alertas(tablero, n_filas, U):
    alertas = []
    def a(sev, cat, var, msg, fix):
        alertas.append({"sev": sev, "tema": cat, "variable": var, "detalle": msg, "sugerencia": fix})
    for _, r in tablero.iterrows():
        v = r["canal"]
        if r["SOI_%"] < U["soi_minimo_pct"]:
            a("warning", "SOI bajo", v,
              f"Pesa solo {r['SOI_%']}% de la inversión total (< {U['soi_minimo_pct']}%).",
              "Agrúpalo con un canal parecido o valida si aporta señal suficiente.")
        if r["%ceros"] > U["pct_ceros"]:
            a("warning", "Muchos silencios", v,
              f"El {r['%ceros']}% de las semanas está en cero.",
              "Canal intermitente: su adstock será inestable. Considera agruparlo.")
        if r["racha_ceros"] >= U["racha_ceros"]:
            a("error", "Apagón prolongado", v,
              f"Estuvo {int(r['racha_ceros'])} semanas seguidas en cero.",
              "Confirma si fue pausa real de campaña o datos faltantes.")
        if (r["picos"] / n_filas * 100) > U["pct_picos"]:
            a("warning", "Picos atípicos", v,
              f"Tiene {int(r['picos'])} semanas con valores muy fuera de rango.",
              "Revisa esas semanas: suelen ser errores de carga o promociones.")
        debil = abs(r["corr_KPI"]) < U["corr_kpi_minima"]
        fuerte_lag = abs(r["corr_en_lag"]) >= U["corr_kpi_minima"]
        if debil and fuerte_lag:
            a("info", "Efecto con rezago", v,
              f"Casi no correlaciona hoy, pero sí {r['corr_en_lag']} con {int(r['mejor_lag'])} semanas de rezago.",
              "Normal en TV/branding. Súbele el adstock (alpha_m) en la Parte 4.")
        elif debil:
            a("warning", "Relación débil", v,
              f"Correlación con el KPI muy baja ({r['corr_KPI']}) incluso con rezago.",
              "Puede aportar poco. Revisa si vale la pena mantenerlo aparte.")
    return alertas

def vif_table(df, cols):
    X = df[cols].fillna(df[cols].median())
    Xs = StandardScaler().fit_transform(X)
    out = []
    for i, c in enumerate(cols):
        y = Xs[:, i]
        Xo = np.delete(Xs, i, axis=1)
        r2 = LinearRegression().fit(Xo, y).score(Xo, y)
        out.append({"variable": c, "VIF": round(1 / (1 - r2), 2) if r2 < 1 else np.inf})
    return pd.DataFrame(out).sort_values("VIF", ascending=False).reset_index(drop=True)

def descomposicion_varianza(df, kpi_col, media, date_col):
    d = df.reset_index(drop=True)
    t = np.arange(len(d))
    mes = pd.to_datetime(d[date_col]).dt.month
    X_ts = np.column_stack([t, t**2, pd.get_dummies(mes, drop_first=True).values]).astype(float)
    y = d[kpi_col].values.astype(float)
    resid = y - LinearRegression().fit(X_ts, y).predict(X_ts)
    Xm = d[media].fillna(d[media].median()).values
    r2m = LinearRegression().fit(Xm, resid).score(Xm, resid)
    var_total, var_resid = np.var(y), np.var(resid)
    pct_ts = round((1 - var_resid / var_total) * 100, 1)
    pct_media_resid = round(r2m * 100, 1)
    techo = round(max(r2m, 0) * (var_resid / var_total) * 100, 1)
    return pct_ts, pct_media_resid, techo

def eficiencia(tablero):
    ef = tablero[["canal", "SOI_%", "corr_KPI", "MI_KPI", "mejor_lag"]].copy()
    ef["fuerza_senal"] = ef["corr_KPI"].abs()
    ef["rank_SOI"] = ef["SOI_%"].rank(ascending=False)
    ef["rank_senal"] = ef["fuerza_senal"].rank(ascending=False)
    ef["lectura"] = np.where(ef["rank_senal"] < ef["rank_SOI"], "🟢 aporta más de lo que gasta",
                     np.where(ef["rank_senal"] > ef["rank_SOI"], "🔴 gasta más de lo que aporta", "≈ equilibrado"))
    return ef.sort_values("fuerza_senal", ascending=False)

def lectura_curvas(df, media, kpi_col):
    y = df[kpi_col]
    filas, overrides = [], {}
    for c in media:
        blag, bcorr = _ccf_best(df[c], y)
        cv = round(df[c].std() / df[c].mean(), 2) if df[c].mean() > 0 else np.nan
        if blag <= 1:
            fam, maxlag, lo, hi = "geométrico (rápido)", "2 a 8", 0.1, 0.5
        else:
            fam, maxlag, lo, hi = "binomial (pico retardado)", "4 a 20", 0.4, 0.9
        nota = "rango estrecho: apóyate en el prior" if (pd.notna(cv) and cv < 0.4) else "rango ok"
        filas.append({"canal": c, "mejor_lag": blag, "corr_en_lag": bcorr,
                      "adstock_sugerido": fam, "max_lag_sugerido": maxlag,
                      "alpha_m_sugerido": f"{lo} - {hi}", "saturación": "cóncava (Hill)",
                      "rango_gasto": nota})
        overrides[c] = {"adstock_low": lo, "adstock_high": hi}
    return pd.DataFrame(filas), overrides

def densidad_senal(n_obs, n_media, n_control, n_knots):
    params = n_media * 3 + n_control + 1 + n_knots
    return round(n_obs / params, 1), params

def generar_codigo_colab(config, umbrales, overrides):
    return (
        "# === Generado por la app de EDA — pégalo en tu notebook de Colab ===\n"
        "# 1) Reemplaza el bloque CONFIG de la Parte 1.2-bis por este:\n"
        f"CONFIG = {json.dumps(config, ensure_ascii=False, indent=4)}\n\n"
        "# 2) (Opcional) Umbrales de EDA usados:\n"
        f"UMBRALES = {json.dumps(umbrales, ensure_ascii=False, indent=4)}\n\n"
        "# 3) Sugerencias de priors por canal (de la Sección B) para el bloque HP de la Parte 4.\n"
        "#    Cópialo dentro de HP['overrides'] = {...}\n"
        f"HP_OVERRIDES = {json.dumps(overrides, ensure_ascii=False, indent=4)}\n"
    )

# ----------------------------------------------------------------------------
# INTERFAZ
# ----------------------------------------------------------------------------
st.set_page_config(page_title="EDA MMM — Meridian", page_icon="📊", layout="wide")
st.title("📊 Preproceso + EDA para Marketing Mix Modeling")
st.caption("Carga tu base, elige KPI / medios / control, revisa la salud de los datos y "
           "genera el código para continuar en Colab con Meridian.")

with st.sidebar:
    st.header("1 · Datos")
    archivo = st.file_uploader("Sube tu base (.xlsx o .csv)", type=["xlsx", "xls", "csv"])

if archivo is None:
    st.info("👈 Sube un archivo para empezar. Debe tener una columna de fecha y el resto numéricas.")
    st.stop()

df = leer_datos(archivo.getvalue(), archivo.name)
numeric = df.select_dtypes(include=[np.number]).columns.tolist()
all_cols = df.columns.tolist()

# ---- Selección de columnas (sidebar) ----
with st.sidebar:
    st.header("2 · Columnas")
    _guess = next((c for c in all_cols if str(c).lower() in
                   ("fecha", "date", "semana", "week", "periodo", "mes")), all_cols[0])
    date_col = st.selectbox("Columna de fecha", all_cols, index=all_cols.index(_guess))

    st.subheader("KPI")
    kpi_mode = st.radio("Cómo definir el KPI", ["Una sola columna", "Sumar varias columnas"], index=0)
    if kpi_mode == "Una sola columna":
        kpi_col = st.selectbox("Columna KPI", numeric)
        kpi_source, kpi_name, kpi_mode_key = [kpi_col], kpi_col, "single"
    else:
        kpi_name = st.text_input("Nombre del KPI", "KPI")
        kpi_source = st.multiselect("Columnas a sumar", numeric)
        kpi_col, kpi_mode_key = kpi_name, "combine"

    st.subheader("Medios (inversión)")
    if st.checkbox("Elegir medios por patrón de texto"):
        pat = st.text_input("Patrón (ej: INVR)", "")
        media_cols = [c for c in numeric if pat and pat.lower() in c.lower()]
        st.caption(f"{len(media_cols)} columnas coinciden")
    else:
        media_cols = st.multiselect("Medios", numeric)

    st.subheader("Control")
    if st.checkbox("Elegir control por patrón de texto"):
        patc = st.text_input("Patrón (ej: POBLAC)", "")
        ctrl_cols = [c for c in numeric if patc and patc.lower() in c.lower()]
        st.caption(f"{len(ctrl_cols)} columnas coinciden")
    else:
        ctrl_cols = st.multiselect("Control", [c for c in numeric if c not in media_cols])

    st.header("3 · Umbrales de alertas")
    with st.expander("Ajustar (opcional)"):
        UMBRALES = {
            "soi_minimo_pct": st.number_input("SOI mínimo %", 0.0, 100.0, 10.0, 1.0),
            "pct_ceros": st.number_input("% ceros para 'silencios'", 0.0, 100.0, 30.0, 5.0),
            "racha_ceros": int(st.number_input("Racha de ceros (semanas)", 1, 52, 6, 1)),
            "iqr_factor": st.number_input("Sensibilidad de picos (IQR)", 0.5, 3.0, 1.5, 0.1),
            "pct_picos": st.number_input("% de picos para alertar", 0.0, 100.0, 4.0, 1.0),
            "corr_kpi_minima": st.number_input("Correlación mínima con KPI", 0.0, 1.0, 0.15, 0.05),
            "vif_alerta": st.number_input("VIF de alerta", 1.0, 50.0, 5.0, 1.0),
        }
    if "UMBRALES" not in dir():
        UMBRALES = {"soi_minimo_pct": 10.0, "pct_ceros": 30.0, "racha_ceros": 6,
                    "iqr_factor": 1.5, "pct_picos": 4.0, "corr_kpi_minima": 0.15, "vif_alerta": 5.0}
    n_knots = int(st.number_input("Knots estimados (densidad de señal)", 2, 60, 12, 1))

# ---- Validaciones ----
errores = []
if not kpi_source:
    errores.append("Elige al menos una columna para el KPI.")
if not media_cols:
    errores.append("Elige al menos un medio (inversión).")
if set(kpi_source) & (set(media_cols) | set(ctrl_cols)):
    errores.append("Una columna del KPI está también en medios/control.")
if set(media_cols) & set(ctrl_cols):
    errores.append("Hay columnas en medios y control a la vez.")

if errores:
    for e in errores:
        st.error("⚠️ " + e)
    st.stop()

# ---- Construir df_model ----
work = df.copy()
work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
if kpi_mode_key == "combine":
    work[kpi_col] = work[kpi_source].sum(axis=1)
df_model = work[[date_col, kpi_col] + media_cols + ctrl_cols].sort_values(date_col).reset_index(drop=True)
predictores = media_cols + ctrl_cols

CONFIG = {"date_col": date_col, "kpi_mode": kpi_mode_key, "kpi_source_cols": kpi_source,
          "kpi_col": kpi_col, "media_cols": media_cols, "control_cols": ctrl_cols}

tab_datos, tab_a, tab_b, tab_code = st.tabs(
    ["📄 Datos", "🩺 Sección A · Salud", "🎯 Sección B · Modelador", "🧩 Código para Colab"])

# ===== TAB DATOS =====
with tab_datos:
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas (semanas)", df_model.shape[0])
    c2.metric("Medios", len(media_cols))
    c3.metric("Controles", len(ctrl_cols))
    nulos = df_model.isna().sum()
    if nulos.any():
        st.warning("Hay nulos: " + ", ".join(f"{k}={v}" for k, v in nulos[nulos > 0].items()))
    st.dataframe(df_model.head(20), use_container_width=True)

# ===== TAB A =====
with tab_a:
    tablero = construir_tablero(df_model, media_cols, kpi_col, UMBRALES["iqr_factor"])
    alertas = generar_alertas(tablero, len(df_model), UMBRALES)
    n_rojo = sum(1 for a in alertas if a["sev"] == "error")

    c1, c2 = st.columns(2)
    c1.metric("Alertas importantes 🔴", n_rojo)
    c2.metric("Para revisar 🟡", len(alertas) - n_rojo)

    st.subheader("Tablero por canal")
    def _estilo(col):
        out = []
        for v in col:
            s = ""
            if col.name == "SOI_%" and v < UMBRALES["soi_minimo_pct"]: s = "background-color:#fff3cd"
            if col.name == "%ceros" and v > UMBRALES["pct_ceros"]: s = "background-color:#fff3cd"
            if col.name == "racha_ceros" and v >= UMBRALES["racha_ceros"]: s = "background-color:#f8d7da"
            if col.name == "picos" and (v / len(df_model) * 100) > UMBRALES["pct_picos"]: s = "background-color:#ffe0b2"
            if col.name == "corr_KPI" and abs(v) < UMBRALES["corr_kpi_minima"]: s = "background-color:#fff3cd"
            out.append(s)
        return out
    try:
        st.dataframe(tablero.style.apply(_estilo), use_container_width=True)
    except Exception:
        st.dataframe(tablero, use_container_width=True)

    st.subheader("Alertas")
    if not alertas:
        st.success("✅ Sin alertas: los canales se ven sanos.")
    else:
        orden = {"error": 0, "warning": 1, "info": 2}
        for a in sorted(alertas, key=lambda x: orden.get(x["sev"], 3)):
            txt = f"**{a['tema']} — {a['variable']}**  \n{a['detalle']}  \n➜ {a['sugerencia']}"
            getattr(st, a["sev"])(txt)

    st.subheader("Distribuciones")
    cols_plot = [kpi_col] + predictores
    ncols = 3
    nrows = int(np.ceil(len(cols_plot) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, c in zip(axes, cols_plot):
        sns.histplot(df_model[c].dropna(), kde=True, ax=ax, color="#4575b4")
        ax.set_title(c, fontsize=9)
        ax.set_xlabel("")
    for ax in axes[len(cols_plot):]:
        ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Redundancia (VIF) y correlación")
    vif = vif_table(df_model, predictores)
    cc1, cc2 = st.columns([1, 1.3])
    with cc1:
        st.dataframe(vif, use_container_width=True)
        redundantes = vif[vif["VIF"] > UMBRALES["vif_alerta"]]["variable"].tolist()
        if redundantes:
            st.warning(f"VIF alto (> {UMBRALES['vif_alerta']}) en: {redundantes}. "
                       "Se pisan entre sí; considera quitar una.")
        else:
            st.success("Sin redundancia preocupante.")
    with cc2:
        figc, axc = plt.subplots(figsize=(1.0 * len(predictores) + 2, 0.8 * len(predictores) + 2))
        sns.heatmap(df_model[predictores].corr(method="spearman"), annot=True, fmt=".2f",
                    cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=axc)
        st.pyplot(figc)

    st.subheader("¿Manda la tendencia? (ADF / KPSS)")
    try:
        import warnings; warnings.filterwarnings("ignore")
        from statsmodels.tsa.stattools import adfuller, kpss
        rows = []
        for c in [kpi_col] + predictores:
            s = df_model[c].dropna()
            try: padf = adfuller(s, autolag="AIC")[1]
            except Exception: padf = np.nan
            try: pk = kpss(s, regression="c", nlags="auto")[1]
            except Exception: pk = np.nan
            domina = pd.notna(padf) and padf > 0.05
            rows.append({"variable": c, "p_ADF": round(padf, 3) if pd.notna(padf) else None,
                         "p_KPSS": round(pk, 3) if pd.notna(pk) else None,
                         "tendencia_fuerte": "⚠️ sí" if domina else "no"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("En Meridian NO se diferencia: la tendencia la maneja con knots. Esto es solo alerta.")
    except ImportError:
        st.info("Instala statsmodels para ADF/KPSS:  pip install statsmodels")

# ===== TAB B =====
with tab_b:
    st.subheader("¿Cuánto peso esperar del total de medios?")
    pct_ts, pct_media, techo = descomposicion_varianza(df_model, kpi_col, media_cols, date_col)
    c1, c2, c3 = st.columns(3)
    c1.metric("Tendencia + estacionalidad", f"{pct_ts}%")
    c2.metric("Medios (del residual)", f"{pct_media}%")
    c3.metric("Techo de contribución", f"~{techo}%")
    st.caption("Referencia para tus priors de ROI, no la contribución final. Un techo >50% o ~0% "
               "es señal de revisar controles o calidad de datos.")

    st.subheader("Contribución esperada por canal (SOI vs eficiencia)")
    ef = eficiencia(tablero)
    st.dataframe(ef.round(3), use_container_width=True)
    figE, axE = plt.subplots(figsize=(8, 5))
    axE.scatter(ef["SOI_%"], ef["fuerza_senal"], s=90, color="#1f77b4")
    for _, r in ef.iterrows():
        axE.annotate(r["canal"], (r["SOI_%"], r["fuerza_senal"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=9)
    axE.set_xlabel("SOI % (peso en inversión)")
    axE.set_ylabel("Fuerza de señal (|corr| con KPI)")
    axE.set_title("Arriba-izquierda = mucha señal con poca inversión")
    axE.grid(alpha=.3)
    st.pyplot(figE)

    st.subheader("Forma de curva a esperar (adstock + saturación)")
    curvas, overrides = lectura_curvas(df_model, media_cols, kpi_col)
    st.dataframe(curvas, use_container_width=True)
    ncols = 3
    nrows = int(np.ceil(len(media_cols) / ncols))
    figS, axesS = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
    axesS = np.array(axesS).reshape(-1)
    for ax, c in zip(axesS, media_cols):
        ax.scatter(df_model[c], df_model[kpi_col], s=12, alpha=.5, color="#4575b4")
        ax.set_title(c, fontsize=9)
    for ax in axesS[len(media_cols):]:
        ax.axis("off")
    figS.suptitle("Inversión vs KPI — si la nube se aplana a la derecha, hay saturación", y=1.02)
    figS.tight_layout()
    st.pyplot(figS)

    st.subheader("¿Tengo datos suficientes? (densidad de señal)")
    ratio, params = densidad_senal(len(df_model), len(media_cols), len(ctrl_cols), n_knots)
    st.metric("Observaciones por parámetro", ratio, help=f"{len(df_model)} obs / {params} parámetros")
    if ratio >= 15: st.success("🟢 Cómodo: los datos mandan sobre los priors.")
    elif ratio >= 8: st.warning("🟡 Aceptable: los priors influyen; que sean razonables.")
    elif ratio >= 5: st.warning("🟠 Justo: tus priors pesarán bastante. Reduce canales/knots.")
    else: st.error("🔴 Insuficiente: el modelo se pegará a los priors. Agrupa canales o pasa a geo-level.")

# ===== TAB CODE =====
with tab_code:
    st.subheader("Código listo para pegar en Colab")
    st.write("Cópialo con el botón de la esquina del bloque, o descárgalo. Reemplaza el `CONFIG` "
             "de la Parte 1.2-bis del notebook y pega los `HP_OVERRIDES` en la Parte 4.")
    _, overrides = lectura_curvas(df_model, media_cols, kpi_col)
    codigo = generar_codigo_colab(CONFIG, UMBRALES, overrides)
    st.code(codigo, language="python")
    st.download_button("⬇️ Descargar config_colab.py", codigo,
                       file_name="config_colab.py", mime="text/x-python")
