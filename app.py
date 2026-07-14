"""
App de Streamlit — Preproceso + EDA para Marketing Mix Modeling (Meridian)
Cliente: Tigo-Millicom  ·  Consumer Science & Analytics (CSA)
Ejecuta:  streamlit run app.py
"""
import os
import io
import json
import base64
from datetime import datetime
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
import streamlit.components.v1 as components

sns.set_theme(style="whitegrid")

# ============================================================================
# MARCA (Tigo-Millicom / CSA)
# ============================================================================
CLIENTE = "Tigo – Millicom"
LOGO_PATH = "logo.png"       # coloca aquí el logo oficial del cliente
NAVY = "#00263A"
BLUE = "#0088CE"
RED = "#E4002B"

CSS = """
<style>
:root { --navy:#00263A; --blue:#0088CE; --red:#E4002B; }
.block-container { padding-top: 1.2rem; }
.csa-banner {
  background: linear-gradient(120deg, #00263A 0%, #003A5C 60%, #0088CE 140%);
  border-radius: 14px; padding: 20px 26px; color: #fff; margin-bottom: 8px;
}
.csa-kicker { color:#FF5A6E; font-weight:800; letter-spacing:.14em; font-size:.78rem; text-transform:uppercase; }
.csa-title { font-size:1.9rem; font-weight:800; line-height:1.1; margin:2px 0 4px 0; }
.csa-sub { color:#Cfe3ef; font-size:.95rem; }
.csa-badge { display:inline-block; background:#E4002B; color:#fff; font-weight:700;
  padding:3px 12px; border-radius:20px; font-size:.8rem; margin-top:8px; }
h2, h3 { color:#00263A; }
div[data-testid="stMetric"] {
  background:#F4F8FB; border:1px solid #e3edf3; border-left:5px solid #0088CE;
  border-radius:10px; padding:12px 14px;
}
.stTabs [data-baseweb="tab-list"] { gap:4px; }
.stTabs [data-baseweb="tab"] { font-weight:600; }
.csa-foot { color:#8aa0ad; font-size:.8rem; text-align:right; margin-top:8px; }
</style>
"""

def banner():
    st.markdown(CSS, unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown(
            '<div class="csa-banner">'
            '<div class="csa-kicker">Consumer Science &amp; Analytics · Marketing Mix Modeling</div>'
            '<div class="csa-title">Preproceso + EDA para MMM</div>'
            '<div class="csa-sub">Carga tu base, define KPI / medios / control, revisa alertas y '
            'genera el código para Meridian en Colab.</div>'
            f'<div class="csa-badge">Cliente: {CLIENTE}</div>'
            '</div>', unsafe_allow_html=True)
    with c2:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.markdown(
                '<div style="border:1px dashed #9bb3c1;border-radius:12px;padding:22px 8px;'
                'text-align:center;color:#7f97a5;font-size:.82rem;margin-top:6px;">'
                'Coloca el logo oficial en<br><code>assets/logo.png</code></div>',
                unsafe_allow_html=True)

# ============================================================================
# FUNCIONES DE APOYO (validadas)
# ============================================================================
@st.cache_data(show_spinner=False)
def leer_datos(contenido: bytes, nombre: str) -> pd.DataFrame:
    buf = io.BytesIO(contenido)
    if nombre.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    return pd.read_csv(buf)

def _racha_max_ceros(s):
    best = cur = 0
    for v in (s == 0).astype(int).values:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best

def _picos(s, f):
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    lo, hi = q1 - f * iqr, q3 + f * iqr
    n = int(((s < lo) | (s > hi)).sum())
    return n, round(n / len(s) * 100, 1)

def _ccf_best(x, y, max_lag=8):
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
        filas.append({"canal": c, "SOI_%": float(soi[c]),
                      "%ceros": round((df[c] == 0).mean() * 100, 1),
                      "racha_ceros": _racha_max_ceros(df[c]), "picos": npk,
                      "corr_KPI": round(df[c].corr(y, method="spearman"), 3),
                      "MI_KPI": round(float(mi[i]), 3), "mejor_lag": blag, "corr_en_lag": bcorr})
    return pd.DataFrame(filas)

def generar_alertas(tablero, n_filas, U):
    alertas = []
    def a(sev, cat, var, msg, fix):
        alertas.append({"sev": sev, "tema": cat, "variable": var, "detalle": msg, "sugerencia": fix})
    for _, r in tablero.iterrows():
        v = r["canal"]
        if r["SOI_%"] < U["soi_minimo_pct"]:
            a("warning", "SOI bajo", v, f"Pesa solo {r['SOI_%']}% de la inversión total (< {U['soi_minimo_pct']}%).",
              "Agrúpalo con un canal parecido o valida si aporta señal suficiente.")
        if r["%ceros"] > U["pct_ceros"]:
            a("warning", "Muchos silencios", v, f"El {r['%ceros']}% de las semanas está en cero.",
              "Canal intermitente: su adstock será inestable. Considera agruparlo.")
        if r["racha_ceros"] >= U["racha_ceros"]:
            a("error", "Apagón prolongado", v, f"Estuvo {int(r['racha_ceros'])} semanas seguidas en cero.",
              "Confirma si fue pausa real de campaña o datos faltantes.")
        if (r["picos"] / n_filas * 100) > U["pct_picos"]:
            a("warning", "Picos atípicos", v, f"Tiene {int(r['picos'])} semanas con valores muy fuera de rango.",
              "Revisa esas semanas: suelen ser errores de carga o promociones.")
        debil = abs(r["corr_KPI"]) < U["corr_kpi_minima"]
        fuerte_lag = abs(r["corr_en_lag"]) >= U["corr_kpi_minima"]
        if debil and fuerte_lag:
            a("info", "Efecto con rezago", v,
              f"Casi no correlaciona hoy, pero sí {r['corr_en_lag']} con {int(r['mejor_lag'])} semanas de rezago.",
              "Normal en TV/branding. Súbele el adstock (alpha_m) en la Parte 4.")
        elif debil:
            a("warning", "Relación débil", v, f"Correlación con el KPI muy baja ({r['corr_KPI']}) incluso con rezago.",
              "Puede aportar poco. Revisa si vale la pena mantenerlo aparte.")
    return alertas

def vif_table(df, cols):
    X = df[cols].fillna(df[cols].median())
    Xs = StandardScaler().fit_transform(X)
    out = []
    for i, c in enumerate(cols):
        y = Xs[:, i]; Xo = np.delete(Xs, i, axis=1)
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
    return (round((1 - var_resid / var_total) * 100, 1), round(r2m * 100, 1),
            round(max(r2m, 0) * (var_resid / var_total) * 100, 1))

def eficiencia(tablero):
    ef = tablero[["canal", "SOI_%", "corr_KPI", "MI_KPI", "mejor_lag"]].copy()
    ef["fuerza_senal"] = ef["corr_KPI"].abs()
    ef["rank_SOI"] = ef["SOI_%"].rank(ascending=False)
    ef["rank_senal"] = ef["fuerza_senal"].rank(ascending=False)
    ef["lectura"] = np.where(ef["rank_senal"] < ef["rank_SOI"], "🟢 aporta más de lo que gasta",
                     np.where(ef["rank_senal"] > ef["rank_SOI"], "🔴 gasta más de lo que aporta", "≈ equilibrado"))
    return ef.sort_values("fuerza_senal", ascending=False)

def lectura_curvas(df, media, kpi_col):
    y = df[kpi_col]; filas, overrides = [], {}
    for c in media:
        blag, bcorr = _ccf_best(df[c], y)
        cv = round(df[c].std() / df[c].mean(), 2) if df[c].mean() > 0 else np.nan
        if blag <= 1:
            fam, maxlag, lo, hi = "geométrico (rápido)", "2 a 8", 0.1, 0.5
        else:
            fam, maxlag, lo, hi = "binomial (pico retardado)", "4 a 20", 0.4, 0.9
        nota = "rango estrecho: apóyate en el prior" if (pd.notna(cv) and cv < 0.4) else "rango ok"
        filas.append({"canal": c, "mejor_lag": blag, "corr_en_lag": bcorr, "adstock_sugerido": fam,
                      "max_lag_sugerido": maxlag, "alpha_m_sugerido": f"{lo} - {hi}",
                      "saturación": "cóncava (Hill)", "rango_gasto": nota})
        overrides[c] = {"adstock_low": lo, "adstock_high": hi}
    return pd.DataFrame(filas), overrides

def densidad_senal(n_obs, n_media, n_control, n_knots):
    params = n_media * 3 + n_control + 1 + n_knots
    return round(n_obs / params, 1), params

def generar_codigo_colab(config, umbrales, overrides):
    return ("# === Generado por la app de EDA (CSA · Tigo-Millicom) — pégalo en Colab ===\n"
            "# 1) Reemplaza el bloque CONFIG de la Parte 1.2-bis por este:\n"
            f"CONFIG = {json.dumps(config, ensure_ascii=False, indent=4)}\n\n"
            "# 2) (Opcional) Umbrales de EDA usados:\n"
            f"UMBRALES = {json.dumps(umbrales, ensure_ascii=False, indent=4)}\n\n"
            "# 3) Sugerencias de priors por canal (Sección B) para HP['overrides'] en la Parte 4:\n"
            f"HP_OVERRIDES = {json.dumps(overrides, ensure_ascii=False, indent=4)}\n")

# ============================================================================
# GRÁFICAS DE TENDENCIA
# ============================================================================
def fig_inversion_apilada(df, media, date_col, freq="MS"):
    d = df[[date_col] + media].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    g = d.set_index(date_col)[media].resample(freq).sum()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    bottom = np.zeros(len(g))
    colores = sns.color_palette("crest", len(media))
    for c, col in zip(media, colores):
        ax.bar(g.index, g[c].values, bottom=bottom, width=22, label=c, color=col)
        bottom += g[c].values
    ax.set_title("Inversión por medio a lo largo del tiempo (mensual)", color=NAVY, fontweight="bold")
    ax.set_ylabel("Inversión")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    return fig

def fig_kpi_trend(df, kpi_col, date_col):
    d = df.copy(); d[date_col] = pd.to_datetime(d[date_col]); d = d.sort_values(date_col)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(d[date_col], d[kpi_col], color=NAVY, lw=2)
    ax.fill_between(d[date_col], d[kpi_col], alpha=0.08, color=NAVY)
    ax.set_title(f"Tendencia del KPI: {kpi_col}", color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig

def fig_overlay(df, media, kpi_col, date_col):
    d = df.copy(); d[date_col] = pd.to_datetime(d[date_col]); d = d.sort_values(date_col)
    total = d[media].sum(axis=1)
    fig, ax1 = plt.subplots(figsize=(11, 4.2))
    ax1.bar(d[date_col], total, width=5, alpha=0.35, color=BLUE, label="Inversión total")
    ax1.set_ylabel("Inversión total", color=BLUE)
    ax2 = ax1.twinx()
    ax2.plot(d[date_col], d[kpi_col], color=RED, lw=2, label="KPI")
    ax2.set_ylabel("KPI", color=RED)
    ax1.set_title("Inversión total vs KPI — ¿se mueven juntos?", color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    b = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b

# ============================================================================
# INFORME FINAL (HTML descargable)
# ============================================================================
def generar_informe_html(cliente, meta, alertas, tablero, figs_b64, seccionB):
    color = {"error": "#E4002B", "warning": "#E8A400", "info": "#0088CE"}
    etiqueta = {"error": "CRÍTICO", "warning": "REVISAR", "info": "INFO"}
    items = ""
    if alertas:
        orden = {"error": 0, "warning": 1, "info": 2}
        for a in sorted(alertas, key=lambda x: orden.get(x["sev"], 3)):
            items += (f'<div style="border-left:5px solid {color[a["sev"]]};background:#fff;'
                      f'padding:10px 14px;margin:8px 0;border-radius:6px;">'
                      f'<b style="color:{color[a["sev"]]}">{etiqueta[a["sev"]]} · {a["tema"]} — {a["variable"]}</b><br>'
                      f'{a["detalle"]}<br><i style="color:#556;">➜ {a["sugerencia"]}</i></div>')
    else:
        items = '<p>Sin alertas: los canales se ven sanos. ✅</p>'
    imgs = "".join(f'<h3>{t}</h3><img src="data:image/png;base64,{b}" style="width:100%;max-width:900px;">'
                   for t, b in figs_b64)
    tabla_html = tablero.to_html(index=False, border=0)
    pct_ts, pct_media, techo = seccionB["descomp"]
    ratio, params = seccionB["densidad"]
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>Informe EDA MMM · {cliente}</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;color:#1b2a33;margin:0;padding:0;background:#f4f7f9;}}
.wrap{{max-width:960px;margin:0 auto;padding:24px;}}
.header{{background:linear-gradient(120deg,#00263A,#0088CE);color:#fff;padding:26px 28px;border-radius:12px;}}
.kick{{color:#FF5A6E;font-weight:800;letter-spacing:.14em;font-size:.75rem;text-transform:uppercase;}}
h1{{margin:4px 0;font-size:1.7rem;}} h2{{color:#00263A;border-bottom:2px solid #e0e8ee;padding-bottom:6px;margin-top:28px;}}
h3{{color:#003A5C;}} table{{border-collapse:collapse;width:100%;font-size:.85rem;}}
th{{background:#00263A;color:#fff;padding:6px 8px;text-align:left;}} td{{padding:5px 8px;border-bottom:1px solid #e6edf1;}}
.cards{{display:flex;gap:12px;margin:10px 0;}} .card{{flex:1;background:#fff;border-left:5px solid #0088CE;
border-radius:8px;padding:12px 14px;}} .card b{{font-size:1.4rem;color:#00263A;}}
.foot{{color:#8aa0ad;font-size:.8rem;text-align:center;margin-top:26px;}}
</style></head><body><div class="wrap">
<div class="header"><div class="kick">Consumer Science &amp; Analytics · MMM</div>
<h1>Informe de EDA — {cliente}</h1>
<div>Generado el {meta['fecha']} · {meta['filas']} semanas · {meta['medios']} medios · {meta['controles']} controles</div></div>

<h2>1 · Alertas</h2>{items}

<h2>2 · Tendencias</h2>{imgs}

<h2>3 · Tablero por canal</h2>{tabla_html}

<h2>4 · Lectura para el modelador</h2>
<div class="cards">
<div class="card"><b>{pct_ts}%</b><br>Tendencia + estacionalidad</div>
<div class="card"><b>~{techo}%</b><br>Techo de contribución de medios</div>
<div class="card"><b>{ratio}</b><br>Obs. por parámetro (densidad)</div>
</div>
<p style="color:#556;font-size:.85rem;">El techo de contribución es una referencia para los priors de ROI, no la
contribución final. La densidad indica cuánto pesarán los datos frente a los priors en Meridian.</p>

<div class="foot">Consumer Science &amp; Analytics · Documento de trabajo para {cliente}</div>
</div></body></html>"""

# ============================================================================
# INTERFAZ
# ============================================================================
st.set_page_config(page_title="EDA MMM · Tigo-Millicom", page_icon="📊", layout="wide")
if os.path.exists(LOGO_PATH):
    try: st.logo(LOGO_PATH)
    except Exception: pass
banner()

archivo = st.file_uploader("Sube tu base (.xlsx o .csv)", type=["xlsx", "xls", "csv"])
if archivo is None:
    st.info("👆 Sube un archivo para empezar. Debe tener una columna de fecha y el resto numéricas.")
    st.stop()

df = leer_datos(archivo.getvalue(), archivo.name)
numeric = df.select_dtypes(include=[np.number]).columns.tolist()
all_cols = df.columns.tolist()

# ---- CONFIGURACIÓN EN EL PANEL PRINCIPAL ----
with st.expander("⚙️  Configuración de campos", expanded=True):
    st.markdown("**1 · Fecha y KPI**")
    c1, c2 = st.columns([1, 2])
    _guess = next((c for c in all_cols if str(c).lower() in
                   ("fecha", "date", "semana", "week", "periodo", "mes")), all_cols[0])
    date_col = c1.selectbox("Columna de fecha", all_cols, index=all_cols.index(_guess))
    with c2:
        kpi_mode = st.radio("Cómo definir el KPI", ["Una sola columna", "Sumar varias columnas"],
                            index=0, horizontal=True)
        if kpi_mode == "Una sola columna":
            kpi_col = st.selectbox("Columna KPI", numeric)
            kpi_source, kpi_name, kpi_mode_key = [kpi_col], kpi_col, "single"
        else:
            kpi_name = st.text_input("Nombre del KPI", "KPI")
            kpi_source = st.multiselect("Columnas a sumar", numeric)
            kpi_col, kpi_mode_key = kpi_name, "combine"

    st.markdown("**2 · Medios y control**")
    c3, c4 = st.columns(2)
    with c3:
        if st.checkbox("Elegir medios por patrón de texto"):
            pat = st.text_input("Patrón para medios (ej: INVR)", "")
            media_cols = [c for c in numeric if pat and pat.lower() in c.lower()]
            st.caption(f"{len(media_cols)} columnas coinciden")
        else:
            media_cols = st.multiselect("Medios (inversión)", numeric)
    with c4:
        if st.checkbox("Elegir control por patrón de texto"):
            patc = st.text_input("Patrón para control (ej: POBLAC)", "")
            ctrl_cols = [c for c in numeric if patc and patc.lower() in c.lower()]
            st.caption(f"{len(ctrl_cols)} columnas coinciden")
        else:
            ctrl_cols = st.multiselect("Control", [c for c in numeric if c not in media_cols])

    with st.expander("Umbrales de alertas y densidad (opcional)"):
        u1, u2, u3, u4 = st.columns(4)
        UMBRALES = {
            "soi_minimo_pct": u1.number_input("SOI mínimo %", 0.0, 100.0, 10.0, 1.0),
            "pct_ceros": u2.number_input("% ceros 'silencios'", 0.0, 100.0, 30.0, 5.0),
            "racha_ceros": int(u3.number_input("Racha ceros (sem)", 1, 52, 6, 1)),
            "iqr_factor": u4.number_input("Sensib. picos (IQR)", 0.5, 3.0, 1.5, 0.1),
            "pct_picos": u1.number_input("% picos alerta", 0.0, 100.0, 4.0, 1.0),
            "corr_kpi_minima": u2.number_input("Corr. mínima KPI", 0.0, 1.0, 0.15, 0.05),
            "vif_alerta": u3.number_input("VIF de alerta", 1.0, 50.0, 5.0, 1.0),
        }
        n_knots = int(u4.number_input("Knots estimados", 2, 60, 12, 1))

# ---- Validaciones ----
errores = []
if not kpi_source: errores.append("Elige al menos una columna para el KPI.")
if not media_cols: errores.append("Elige al menos un medio (inversión).")
if set(kpi_source) & (set(media_cols) | set(ctrl_cols)): errores.append("Una columna del KPI está también en medios/control.")
if set(media_cols) & set(ctrl_cols): errores.append("Hay columnas en medios y control a la vez.")
if errores:
    for e in errores: st.error("⚠️ " + e)
    st.stop()

# ---- df_model ----
work = df.copy()
work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
if kpi_mode_key == "combine":
    work[kpi_col] = work[kpi_source].sum(axis=1)
df_model = work[[date_col, kpi_col] + media_cols + ctrl_cols].sort_values(date_col).reset_index(drop=True)
predictores = media_cols + ctrl_cols
CONFIG = {"date_col": date_col, "kpi_mode": kpi_mode_key, "kpi_source_cols": kpi_source,
          "kpi_col": kpi_col, "media_cols": media_cols, "control_cols": ctrl_cols}

# cálculos base compartidos
tablero = construir_tablero(df_model, media_cols, kpi_col, UMBRALES["iqr_factor"])
alertas = generar_alertas(tablero, len(df_model), UMBRALES)
seccionB = {"descomp": descomposicion_varianza(df_model, kpi_col, media_cols, date_col),
            "densidad": densidad_senal(len(df_model), len(media_cols), len(ctrl_cols), n_knots)}

t_datos, t_tend, t_alert, t_model, t_inf, t_code = st.tabs(
    ["📄 Datos", "📈 Tendencias", "🚨 Alertas", "🎯 Modelador", "📑 Informe", "🧩 Código Colab"])

# ===== DATOS =====
with t_datos:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Semanas", df_model.shape[0])
    c2.metric("Medios", len(media_cols))
    c3.metric("Controles", len(ctrl_cols))
    c4.metric("KPI", kpi_col)
    nulos = df_model.isna().sum()
    if nulos.any():
        st.warning("Nulos: " + ", ".join(f"{k}={v}" for k, v in nulos[nulos > 0].items()))
    st.dataframe(df_model.head(20), use_container_width=True)

# ===== TENDENCIAS =====
with t_tend:
    st.subheader("Inversión por medio a lo largo del tiempo")
    f1 = fig_inversion_apilada(df_model, media_cols, date_col)
    st.pyplot(f1); plt.close(f1)
    st.subheader("Tendencia del KPI")
    f2 = fig_kpi_trend(df_model, kpi_col, date_col)
    st.pyplot(f2); plt.close(f2)
    st.subheader("Inversión total vs KPI")
    st.caption("Dónde suben juntos y dónde se separan: divergencias sostenidas sugieren saturación o efecto de otros factores.")
    f3 = fig_overlay(df_model, media_cols, kpi_col, date_col)
    st.pyplot(f3); plt.close(f3)

# ===== ALERTAS =====
with t_alert:
    n_rojo = sum(1 for a in alertas if a["sev"] == "error")
    c1, c2 = st.columns(2)
    c1.metric("Alertas críticas 🔴", n_rojo)
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
            getattr(st, a["sev"])(f"**{a['tema']} — {a['variable']}**  \n{a['detalle']}  \n➜ {a['sugerencia']}")

    st.subheader("Distribuciones")
    cols_plot = [kpi_col] + predictores
    ncols = 3; nrows = int(np.ceil(len(cols_plot) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, c in zip(axes, cols_plot):
        sns.histplot(df_model[c].dropna(), kde=True, ax=ax, color="#0088CE")
        ax.set_title(c, fontsize=9); ax.set_xlabel("")
    for ax in axes[len(cols_plot):]: ax.axis("off")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.subheader("Redundancia (VIF) y correlación")
    vif = vif_table(df_model, predictores)
    cc1, cc2 = st.columns([1, 1.3])
    with cc1:
        st.dataframe(vif, use_container_width=True)
        red = vif[vif["VIF"] > UMBRALES["vif_alerta"]]["variable"].tolist()
        st.warning(f"VIF alto en: {red}. Se pisan entre sí; considera quitar una.") if red else st.success("Sin redundancia preocupante.")
    with cc2:
        figc, axc = plt.subplots(figsize=(1.0 * len(predictores) + 2, 0.8 * len(predictores) + 2))
        sns.heatmap(df_model[predictores].corr(method="spearman"), annot=True, fmt=".2f",
                    cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=axc)
        st.pyplot(figc); plt.close(figc)

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
            rows.append({"variable": c, "p_ADF": round(padf, 3) if pd.notna(padf) else None,
                         "p_KPSS": round(pk, 3) if pd.notna(pk) else None,
                         "tendencia_fuerte": "⚠️ sí" if (pd.notna(padf) and padf > 0.05) else "no"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("En Meridian NO se diferencia: la tendencia la maneja con knots. Esto es solo alerta.")
    except ImportError:
        st.info("Instala statsmodels para ADF/KPSS:  pip install statsmodels")

# ===== MODELADOR =====
with t_model:
    st.subheader("¿Cuánto peso esperar del total de medios?")
    pct_ts, pct_media, techo = seccionB["descomp"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Tendencia + estacionalidad", f"{pct_ts}%")
    c2.metric("Medios (del residual)", f"{pct_media}%")
    c3.metric("Techo de contribución", f"~{techo}%")
    st.caption("Referencia para tus priors de ROI, no la contribución final.")

    st.subheader("Contribución esperada por canal (SOI vs eficiencia)")
    ef = eficiencia(tablero)
    st.dataframe(ef.round(3), use_container_width=True)
    figE, axE = plt.subplots(figsize=(8, 5))
    axE.scatter(ef["SOI_%"], ef["fuerza_senal"], s=90, color=BLUE)
    for _, r in ef.iterrows():
        axE.annotate(r["canal"], (r["SOI_%"], r["fuerza_senal"]), textcoords="offset points", xytext=(6, 4), fontsize=9)
    axE.set_xlabel("SOI % (peso en inversión)"); axE.set_ylabel("Fuerza de señal (|corr| con KPI)")
    axE.set_title("Arriba-izquierda = mucha señal con poca inversión"); axE.grid(alpha=.3)
    st.pyplot(figE); plt.close(figE)

    st.subheader("Forma de curva a esperar")
    curvas, overrides = lectura_curvas(df_model, media_cols, kpi_col)
    st.dataframe(curvas, use_container_width=True)

    st.subheader("¿Tengo datos suficientes? (densidad de señal)")
    ratio, params = seccionB["densidad"]
    st.metric("Observaciones por parámetro", ratio, help=f"{len(df_model)} obs / {params} parámetros")
    if ratio >= 15: st.success("🟢 Cómodo: los datos mandan sobre los priors.")
    elif ratio >= 8: st.warning("🟡 Aceptable: los priors influyen; que sean razonables.")
    elif ratio >= 5: st.warning("🟠 Justo: tus priors pesarán bastante. Reduce canales/knots.")
    else: st.error("🔴 Insuficiente: el modelo se pegará a los priors. Agrupa canales o pasa a geo-level.")

# ===== INFORME =====
with t_inf:
    st.subheader("Informe final")
    st.write("Reúne alertas, tendencias y lectura para el modelador en un solo documento descargable.")
    meta = {"fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), "filas": len(df_model),
            "medios": len(media_cols), "controles": len(ctrl_cols)}
    figs_b64 = [
        ("Inversión por medio a lo largo del tiempo", fig_to_b64(fig_inversion_apilada(df_model, media_cols, date_col))),
        ("Tendencia del KPI", fig_to_b64(fig_kpi_trend(df_model, kpi_col, date_col))),
        ("Inversión total vs KPI", fig_to_b64(fig_overlay(df_model, media_cols, kpi_col, date_col))),
    ]
    html = generar_informe_html(CLIENTE, meta, alertas, tablero, figs_b64, seccionB)
    st.download_button("⬇️ Descargar informe (HTML)", html,
                       file_name=f"informe_eda_{datetime.now():%Y%m%d}.html", mime="text/html")
    st.caption("Ábrelo en el navegador y usa Imprimir → Guardar como PDF si lo necesitas en PDF.")
    st.markdown("**Vista previa:**")
    components.html(html, height=650, scrolling=True)

# ===== CÓDIGO =====
with t_code:
    st.subheader("Código listo para pegar en Colab")
    st.write("Cópialo con el botón del bloque o descárgalo. `CONFIG` va en la Parte 1.2-bis y "
             "`HP_OVERRIDES` en `HP['overrides']` de la Parte 4.")
    _, overrides = lectura_curvas(df_model, media_cols, kpi_col)
    codigo = generar_codigo_colab(CONFIG, UMBRALES, overrides)
    st.code(codigo, language="python")
    st.download_button("⬇️ Descargar config_colab.py", codigo, file_name="config_colab.py", mime="text/x-python")

st.markdown('<div class="csa-foot">Consumer Science &amp; Analytics · Herramienta interna para '
            f'{CLIENTE}</div>', unsafe_allow_html=True)
