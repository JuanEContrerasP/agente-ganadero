"""
app.py — Interfaz web del Agente Ganadero (Streamlit).

Ejecutar:
    streamlit run app.py

Características:
  • Chat conversacional con historial en pantalla
  • Panel lateral con preguntas de ejemplo y botón para nueva sesión
  • Expansores para ver las herramientas usadas y las fuentes RAG
  • Botón para re-ingestar documentos sin salir de la app
"""

import streamlit as st
from pathlib import Path

# Auto-ingestión: si el vectorstore no existe (ej: primer despliegue en la nube),
# lo genera antes de que cualquier componente intente cargarlo.
_vs = Path("vectorstore")
if not _vs.exists() or not any(_vs.iterdir()):
    with st.spinner("Indexando documentos por primera vez..."):
        from ingest import ingestar
        ingestar(limpiar=False)

# ── Configuración general de la página ────────────────────────────────────────
st.set_page_config(
    page_title="Agente Ganadero IA",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS mínimo para mejorar legibilidad de los bloques de herramientas
st.markdown(
    """
    <style>
    .tool-box { background:#f0f4f8; border-left:3px solid #2e86ab;
                padding:8px 12px; border-radius:4px; font-size:0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Carga del agente (una sola vez por sesión de Streamlit) ───────────────────
@st.cache_resource(show_spinner="Cargando base de conocimiento y modelos...")
def _cargar_agente():
    """Instancia AgenteGanadero una sola vez; se reutiliza entre reruns."""
    from agent import AgenteGanadero
    return AgenteGanadero()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Cow_female_black_white.jpg/320px-Cow_female_black_white.jpg",
        use_container_width=True,
    )
    st.title("Agente Ganadero IA")
    st.caption("Monitoreo inteligente de ganado y tierras agrícolas")

    st.divider()
    st.subheader("Tecnologías")
    st.markdown(
        """
        | Componente | Tecnología |
        |---|---|
        | LLM | llama-3.3-70b (Groq) |
        | Vector DB | ChromaDB |
        | Embeddings | sentence-transformers |
        | Clima | Open-Meteo API |
        | UI | Streamlit |
        """
    )

    st.divider()
    st.subheader("Preguntas de ejemplo")

    EJEMPLOS = [
        "¿Cuáles son los síntomas de la fiebre aftosa en bovinos?",
        "¿Qué vacunas necesita un bovino según el calendario colombiano?",
        "¿Qué antibióticos hay disponibles para bovinos y en qué presentaciones?",
        "Tengo un potrero de 8 ha con Brachiaria. ¿Cuántas vacas de 400 kg puedo tener?",
        "¿Cuál es el clima hoy en Bogotá? (lat: 4.71, lon: -74.07)",
        "¿Qué cultivos predominan en la vereda Páramo según el registro de suelos?",
        "¿Qué antiparasitarios están disponibles para equinos?",
        "¿Qué es la carga animal y cómo afecta la pradera?",
    ]

    for ejemplo in EJEMPLOS:
        if st.button(ejemplo, use_container_width=True, key=f"ej_{ejemplo[:25]}"):
            st.session_state["pregunta_rapida"] = ejemplo

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Nueva sesión", use_container_width=True):
            st.session_state["mensajes"] = []
            try:
                _cargar_agente().limpiar_historial()
            except Exception:
                pass
            st.rerun()

    with col2:
        if st.button("🔄 Re-ingestar", use_container_width=True):
            with st.spinner("Procesando documentos..."):
                try:
                    from ingest import ingestar
                    ingestar(limpiar=True)
                    st.cache_resource.clear()
                    st.success("Listo.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error: {exc}")


# ── Helpers de renderizado ─────────────────────────────────────────────────────

def _mostrar_detalles(msg: dict) -> None:
    """Muestra los expansores de herramientas y fuentes bajo cada mensaje."""
    herramientas = msg.get("herramientas", [])
    fuentes      = msg.get("fuentes", [])

    if herramientas:
        with st.expander(f"🔧 Herramientas utilizadas ({len(herramientas)})"):
            for h in herramientas:
                st.markdown(f"**`{h['herramienta']}`**")
                st.markdown("*Argumentos:*")
                st.json(h["argumentos"])
                st.markdown("*Resultado:*")
                st.json(h["resultado"])

    if fuentes:
        with st.expander(f"📚 Fuentes RAG ({len(fuentes)})"):
            for f in fuentes:
                st.markdown(f"- `{f}`")


# ── Área principal ─────────────────────────────────────────────────────────────
st.title("🐄 Agente Inteligente de Monitoreo Ganadero")
st.caption("Consulta sobre salud animal, praderas, suelos y productos veterinarios")

if "mensajes" not in st.session_state:
    st.session_state["mensajes"] = []

for msg in st.session_state["mensajes"]:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["contenido"])
        if "fuentes" in msg or "herramientas" in msg:
            _mostrar_detalles(msg)


# ── Capturar input: teclado o botón del sidebar ───────────────────────────────
pregunta_sidebar = st.session_state.pop("pregunta_rapida", None)
pregunta_chat    = st.chat_input("Escribe tu pregunta sobre ganado o tierras...")

pregunta = pregunta_sidebar or pregunta_chat

# ── Procesar pregunta ─────────────────────────────────────────────────────────
if pregunta:
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(pregunta)
    st.session_state["mensajes"].append({"rol": "user", "contenido": pregunta})

    # Obtener respuesta del agente
    with st.chat_message("assistant"):
        with st.spinner("Consultando base de conocimiento..."):
            try:
                agente = _cargar_agente()
                respuesta, fuentes, herramientas = agente.responder(pregunta)
            except RuntimeError as exc:
                # Error común: colección no indexada todavía
                respuesta   = f"⚠️ **Error de configuración:** {exc}"
                fuentes     = []
                herramientas = []
            except Exception as exc:
                respuesta   = f"⚠️ **Error inesperado:** {exc}"
                fuentes     = []
                herramientas = []

        st.markdown(respuesta)

        # Mostrar detalles de la respuesta recién generada
        msg_actual = {"herramientas": herramientas, "fuentes": fuentes}
        _mostrar_detalles(msg_actual)

    # Guardar en historial de la sesión
    st.session_state["mensajes"].append({
        "rol":          "assistant",
        "contenido":    respuesta,
        "fuentes":      fuentes,
        "herramientas": herramientas,
    })
