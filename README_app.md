# App de Preproceso + EDA para MMM (Meridian)

App de Streamlit para cargar tu base, elegir **KPI / medios / control**, revisar la
**salud de los datos con alertas** y **generar el código** para continuar en Colab con
Google Meridian. No entrena el modelo (eso se queda en Colab/GPU): es rápida y ligera.

## Cómo correrla

```bash
# 1. (recomendado) crea un entorno virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. instala dependencias
pip install -r requirements.txt

# 3. lanza la app
streamlit run app.py
```

Se abre en el navegador (por defecto http://localhost:8501).

## Cómo se usa

1. **Sube** tu archivo `.xlsx` o `.csv` (una columna de fecha + columnas numéricas).
2. En la barra lateral elige la **fecha**, el **KPI** (una columna o la suma de varias),
   los **medios** y los **controles**. Con +1000 columnas usa la opción
   *"Elegir por patrón de texto"* (ej. `INVR`, `POBLAC`).
3. Revisa las pestañas:
   - **Sección A · Salud:** tablero por canal con semáforo + alertas (SOI bajo, silencios,
     picos atípicos, relación débil), distribuciones, VIF y estacionariedad.
   - **Sección B · Modelador:** techo de contribución de medios, SOI vs eficiencia,
     forma de curva sugerida por canal y densidad de señal.
4. En **Código para Colab** copia el bloque generado (`CONFIG`, `UMBRALES`,
   `HP_OVERRIDES`) y pégalo en el notebook: `CONFIG` en la Parte 1.2-bis y
   `HP_OVERRIDES` dentro de `HP['overrides']` en la Parte 4.

## Despliegue

- **Local:** con los pasos de arriba.
- **Streamlit Community Cloud (gratis):** sube esta carpeta a un repo de GitHub y
  conéctalo. Funciona bien porque la app es liviana (no incluye Meridian/TensorFlow).
- **Servidor propio:** `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`.

## Nota
El entrenamiento del modelo Meridian (MCMC, pesado, ideal con GPU) se hace en el
notebook de Colab usando el `CONFIG` que exporta esta app.
