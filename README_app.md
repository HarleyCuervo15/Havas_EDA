# App de Preproceso + EDA para MMM (Meridian)
### Consumer Science & Analytics · Cliente: Tigo–Millicom

App de Streamlit con branding CSA/Tigo para cargar la base, definir **KPI / medios / control**
en el panel principal, revisar **tendencias y alertas**, y generar tanto un **informe final
descargable** como el **código para continuar en Colab** con Google Meridian. No entrena el
modelo (eso se queda en Colab/GPU): es rápida y liviana.

## Cómo correrla

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Se abre en http://localhost:8501

## Logo del cliente
Coloca el logo oficial en `assets/logo.png`. La app lo detecta solo y lo muestra en el
encabezado; si no está, deja un espacio indicándolo. (No se incluye el logo por ser marca
registrada del cliente.)

## Qué hace, pestaña por pestaña
- **Configuración (panel principal):** fecha, KPI (una columna o suma de varias), medios y
  control. Con +1000 columnas usa "elegir por patrón de texto" (ej. `INVR`, `POBLAC`).
- **📄 Datos:** resumen y vista previa.
- **📈 Tendencias:** inversión por medio en el tiempo (barras apiladas mensuales), tendencia
  del KPI, e inversión total vs KPI (para ver dónde se mueven juntos y dónde se separan).
- **🚨 Alertas:** tablero por canal con semáforo + alertas en lenguaje claro (SOI bajo,
  silencios, picos, relación débil), distribuciones, VIF y estacionariedad.
- **🎯 Modelador:** techo de contribución, SOI vs eficiencia, forma de curva y densidad de señal.
- **📑 Informe:** documento HTML descargable con alertas, gráficas y lectura para el modelador
  (ábrelo e imprime a PDF si lo necesitas).
- **🧩 Código Colab:** bloque `CONFIG` + `UMBRALES` + `HP_OVERRIDES` para pegar en el notebook.

## Despliegue
- **Local:** pasos de arriba.
- **Streamlit Community Cloud (gratis):** sube esta carpeta (incluida `assets/`) a un repo de
  GitHub y conéctalo en streamlit.io/cloud. La app es liviana, así que corre sin problema.
- **Servidor interno:** `streamlit run app.py --server.address 0.0.0.0 --server.port 8501`.

## Nota
El entrenamiento de Meridian (MCMC, pesado, ideal con GPU) se hace en Colab con el `CONFIG`
que exporta esta app.
