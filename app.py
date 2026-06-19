import streamlit as st
import numpy as np
import joblib
import os

# ─── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Valor de Vivienda",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Estilos personalizados ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo principal */
    .stApp { background-color: #0f1117; }

    /* Tarjeta de resultado */
    .result-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #3d5a99;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-label {
        color: #8fa8d0;
        font-size: 0.85rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .result-value {
        color: #4fc3f7;
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1;
    }
    .result-sub {
        color: #546e8a;
        font-size: 0.78rem;
        margin-top: 0.6rem;
    }

    /* Tarjetas de métricas secundarias */
    .metric-row { display: flex; gap: 1rem; margin-top: 1.2rem; flex-wrap: wrap; }
    .metric-card {
        flex: 1; min-width: 120px;
        background: #1a1f2e;
        border: 1px solid #2a3550;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .metric-card .m-label { color: #546e8a; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-card .m-val   { color: #90caf9; font-size: 1.1rem; font-weight: 600; margin-top: 2px; }

    /* Sección de sidebar */
    .sidebar-header {
        color: #4fc3f7;
        font-size: 0.7rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        padding: 0.5rem 0 0.2rem;
        border-bottom: 1px solid #2a3550;
        margin-bottom: 0.6rem;
    }

    /* Divisor */
    hr { border-color: #1e2d40 !important; }

    /* Botón principal */
    .stButton > button {
        background: linear-gradient(90deg, #1565c0, #0d47a1);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Info box */
    .info-box {
        background: #131929;
        border-left: 3px solid #3d5a99;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        color: #8fa8d0;
        font-size: 0.82rem;
        margin-top: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Carga del modelo ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Carga el modelo guardado. Ajusta la ruta y el método según tu archivo."""
    model_path = "model.pkl"          # ← cambia al nombre real de tu archivo
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()


# ─── Constantes de referencia (basadas en estadísticas del dataset) ────────────
STATS = {
    "longitud":              {"mean": -119.57, "std": 2.00,    "min": -124.35, "max": -114.31},
    "latitud":               {"mean":   35.63, "std": 2.14,    "min":   32.54, "max":   41.95},
    "edad_mediana_vivienda": {"mean":   28.63, "std": 12.59,   "min":    1.0,  "max":   52.0},
    "total_habitaciones":    {"mean": 2636.50, "std": 2185.22, "min":    2.0,  "max":39320.0},
    "total_dormitorios":     {"mean":  537.87, "std":  421.37, "min":    1.0,  "max": 6445.0},
    "poblacion":             {"mean": 1424.95, "std": 1133.18, "min":    3.0,  "max":35682.0},
    "hogares":               {"mean":  499.43, "std":  382.29, "min":    1.0,  "max": 6082.0},
    "ingreso_mediano":       {"mean":    3.87, "std":   1.90,  "min":    0.5,  "max":   15.0},
}

PROXIMIDAD_OPTIONS = [
    "<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"
]

PROXIMIDAD_LABELS = {
    "<1H OCEAN":  "< 1 hora del océano",
    "INLAND":     "Interior / Tierra adentro",
    "NEAR OCEAN": "Cerca del océano",
    "NEAR BAY":   "Cerca de la bahía",
    "ISLAND":     "Isla",
}


# ─── Sidebar – Controles de entrada ───────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏡 Variables del inmueble")
    st.markdown("Ajusta cada parámetro para obtener la predicción.")
    st.markdown("---")

    # — Ubicación —
    st.markdown('<p class="sidebar-header">📍 Ubicación</p>', unsafe_allow_html=True)
    longitud = st.slider(
        "Longitud", min_value=-124.35, max_value=-114.31,
        value=float(STATS["longitud"]["mean"]), step=0.01,
        help="Coordenada de longitud del bloque censal"
    )
    latitud = st.slider(
        "Latitud", min_value=32.54, max_value=41.95,
        value=float(STATS["latitud"]["mean"]), step=0.01,
        help="Coordenada de latitud del bloque censal"
    )
    proximidad_oceano = st.selectbox(
        "Proximidad al océano",
        options=PROXIMIDAD_OPTIONS,
        format_func=lambda x: PROXIMIDAD_LABELS[x],
        help="Categoría de distancia al océano más cercano"
    )

    st.markdown("---")

    # — Características de la vivienda —
    st.markdown('<p class="sidebar-header">🏠 Características</p>', unsafe_allow_html=True)
    edad_mediana_vivienda = st.slider(
        "Edad mediana de vivienda (años)", min_value=1, max_value=52,
        value=int(STATS["edad_mediana_vivienda"]["mean"]),
        help="Edad mediana de las viviendas del bloque"
    )
    total_habitaciones = st.number_input(
        "Total de habitaciones", min_value=2, max_value=39320,
        value=int(STATS["total_habitaciones"]["mean"]),
        step=50,
        help="Número total de habitaciones en el bloque censal"
    )
    total_dormitorios = st.number_input(
        "Total de dormitorios", min_value=1, max_value=6445,
        value=int(STATS["total_dormitorios"]["mean"]),
        step=10,
        help="Número total de dormitorios en el bloque censal"
    )

    st.markdown("---")

    # — Demografía —
    st.markdown('<p class="sidebar-header">👥 Demografía</p>', unsafe_allow_html=True)
    poblacion = st.number_input(
        "Población", min_value=3, max_value=35682,
        value=int(STATS["poblacion"]["mean"]),
        step=100,
        help="Población total del bloque censal"
    )
    hogares = st.number_input(
        "Hogares", min_value=1, max_value=6082,
        value=int(STATS["hogares"]["mean"]),
        step=10,
        help="Número de hogares en el bloque censal"
    )
    ingreso_mediano = st.slider(
        "Ingreso mediano (×$10,000)", min_value=0.5, max_value=15.0,
        value=float(STATS["ingreso_mediano"]["mean"]),
        step=0.1,
        help="Ingreso mediano del hogar (en decenas de miles de USD)"
    )

    st.markdown("---")
    predict_btn = st.button("🔮 Predecir valor")


# ─── Área principal ────────────────────────────────────────────────────────────
st.markdown("# Predictor de Valor Mediano de Vivienda")
st.markdown("Modelo entrenado con datos del **California Housing Dataset**. "
            "Ajusta las variables en el panel izquierdo y presiona **Predecir valor**.")
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

# — Columna izquierda: resumen de inputs —
with col1:
    st.markdown("### 📋 Resumen de parámetros seleccionados")

    data_display = {
        "Variable": [
            "Longitud", "Latitud", "Proximidad al océano",
            "Edad mediana vivienda", "Total habitaciones", "Total dormitorios",
            "Población", "Hogares", "Ingreso mediano"
        ],
        "Valor": [
            f"{longitud:.2f}", f"{latitud:.2f}", PROXIMIDAD_LABELS[proximidad_oceano],
            f"{edad_mediana_vivienda} años", f"{total_habitaciones:,}", f"{total_dormitorios:,}",
            f"{poblacion:,}", f"{hogares:,}", f"${ingreso_mediano * 10_000:,.0f}"
        ],
    }

    # Tabla simple con st.dataframe
    import pandas as pd
    df_display = pd.DataFrame(data_display)
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Métricas derivadas útiles
    st.markdown("#### 📊 Métricas derivadas")
    hab_por_hogar = total_habitaciones / max(hogares, 1)
    dorm_por_hogar = total_dormitorios / max(hogares, 1)
    personas_por_hogar = poblacion / max(hogares, 1)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Hab. / Hogar",      f"{hab_por_hogar:.1f}")
    mc2.metric("Dorm. / Hogar",     f"{dorm_por_hogar:.1f}")
    mc3.metric("Personas / Hogar",  f"{personas_por_hogar:.1f}")

# — Columna derecha: predicción —
with col2:
    st.markdown("### 🔮 Resultado de la predicción")

    if predict_btn:
        # Codificación one-hot de proximidad_oceano
        # Ajusta el orden de columnas exactamente al que usaste en entrenamiento
        prox_encoded = {opt: 0 for opt in PROXIMIDAD_OPTIONS}
        prox_encoded[proximidad_oceano] = 1

        # Vector de features — AJUSTA EL ORDEN a tu pipeline de entrenamiento
        features = np.array([[
            longitud,
            latitud,
            edad_mediana_vivienda,
            total_habitaciones,
            total_dormitorios,
            poblacion,
            hogares,
            ingreso_mediano,
            prox_encoded["<1H OCEAN"],
            prox_encoded["INLAND"],
            prox_encoded["ISLAND"],
            prox_encoded["NEAR BAY"],
            prox_encoded["NEAR OCEAN"],
        ]])

        if model is not None:
            prediction = model.predict(features)[0]
        else:
            # ── Simulación cuando no hay modelo cargado ──────────────────────
            # Elimina este bloque cuando tengas tu model.pkl
            base = (ingreso_mediano * 25_000
                    + (52 - edad_mediana_vivienda) * 1500
                    + (hab_por_hogar * 8000)
                    + (100_000 if proximidad_oceano in ["<1H OCEAN", "NEAR OCEAN", "NEAR BAY"] else 0))
            prediction = np.clip(base + np.random.normal(0, 5000), 14999, 500001)
            st.warning("⚠️ Modelo no encontrado — mostrando estimación de ejemplo. "
                       "Coloca tu archivo `model.pkl` en el mismo directorio que `app.py`.",
                       icon="🤖")

        # Tarjeta de resultado principal
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Valor mediano estimado</div>
            <div class="result-value">${prediction:,.0f}</div>
            <div class="result-sub">USD · Valor mediano del bloque censal</div>
        </div>
        """, unsafe_allow_html=True)

        # Contexto relativo
        pct_max = prediction / 500_001 * 100
        st.markdown("---")
        st.markdown(f"**Posición en el rango del dataset**")
        st.progress(min(pct_max / 100, 1.0))
        st.caption(f"${14_999:,} mín. — ${500_001:,} máx. · Predicción en el percentil ~{pct_max:.0f}%")

        # Ingreso en contexto
        ratio = prediction / max(ingreso_mediano * 10_000, 1)
        st.markdown(f"""
        <div class="info-box">
            💡 El valor predicho es <strong>{ratio:.1f}×</strong> el ingreso mediano anual del hogar
            (${ingreso_mediano * 10_000:,.0f}), lo que da una relación
            precio/ingreso de <strong>{ratio:.1f}</strong>.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-card" style="opacity:0.5;">
            <div class="result-label">Valor mediano estimado</div>
            <div class="result-value" style="color:#2a3550;">$ — — —</div>
            <div class="result-sub">Ajusta los parámetros y presiona "Predecir valor"</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            👈 Usa el panel lateral para configurar las variables del inmueble y obtener
            una predicción del valor mediano de vivienda.
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Modelo predictivo · California Housing Dataset · "
           "Variable objetivo: `valor_mediano_vivienda`")
