"""
app.py — Interfaz web del Agente Ganadero (Streamlit).

Ejecutar:
    streamlit run app.py
"""

import streamlit as st

# set_page_config DEBE ser el primer comando Streamlit — sin excepciones
st.set_page_config(
    page_title="Agente Ganadero IA",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Carga del agente + auto-ingestión (una sola vez por sesión) ───────────────
@st.cache_resource(show_spinner="Cargando modelos e indexando documentos...")
def _cargar_agente():
    """
    Instancia el agente. Si el vectorstore no existe (primer despliegue),
    ejecuta la ingestión automáticamente antes de crear el agente.
    """
    from pathlib import Path
    from ingest import ingestar, VECTORSTORE_DIR, COLLECTION_NAME
    import chromadb

    # Verificar si la colección ya existe en disco
    coleccion_lista = False
    if VECTORSTORE_DIR.exists():
        try:
            cliente = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
            nombres = [c.name for c in cliente.list_collections()]
            coleccion_lista = COLLECTION_NAME in nombres
        except Exception:
            coleccion_lista = False

    if not coleccion_lista:
        ingestar(limpiar=True)

    from agent import AgenteGanadero
    return AgenteGanadero()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🐄 Agente Ganadero IA")
    st.caption("Monitoreo inteligente de ganado y tierras agrícolas")

    st.divider()
    st.subheader("Stack tecnológico")
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
        "¿Cuál es el clima hoy en Bogotá? lat=4.71, lon=-74.07",
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
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error: {exc}")


# ── Helper de renderizado ──────────────────────────────────────────────────────

def _mostrar_detalles(msg: dict) -> None:
    """Muestra expansores de herramientas y fuentes bajo un mensaje."""
    herramientas = msg.get("herramientas", [])
    fuentes      = msg.get("fuentes", [])

    if herramientas:
        with st.expander(f"🔧 Herramientas utilizadas ({len(herramientas)})"):
            for h in herramientas:
                st.markdown(f"**`{h['herramienta']}`**")
                st.json({"argumentos": h["argumentos"], "resultado": h["resultado"]})

    if fuentes:
        with st.expander(f"📚 Fuentes consultadas ({len(fuentes)})"):
            for fuente in fuentes:
                st.markdown(f"- `{fuente}`")


# ── Área principal ─────────────────────────────────────────────────────────────
st.title("🐄 Agente Inteligente de Monitoreo Ganadero")
st.caption("Consulta sobre salud animal, praderas, suelos y productos veterinarios")

if "mensajes" not in st.session_state:
    st.session_state["mensajes"] = []

# Renderizar historial de la conversación
for msg in st.session_state["mensajes"]:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["contenido"])
        if "fuentes" in msg or "herramientas" in msg:
            _mostrar_detalles(msg)

# ── Input del usuario ─────────────────────────────────────────────────────────
pregunta_sidebar = st.session_state.pop("pregunta_rapida", None)
pregunta_chat    = st.chat_input("Escribe tu pregunta sobre ganado o tierras...")
pregunta         = pregunta_sidebar or pregunta_chat

# ── Procesar pregunta ─────────────────────────────────────────────────────────
if pregunta:
    with st.chat_message("user"):
        st.markdown(pregunta)
    st.session_state["mensajes"].append({"rol": "user", "contenido": pregunta})

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                agente = _cargar_agente()
                respuesta, fuentes, herramientas = agente.responder(pregunta)
            except Exception as exc:
                respuesta    = f"⚠️ **Error:** {exc}"
                fuentes      = []
                herramientas = []

        st.markdown(respuesta)
        _mostrar_detalles({"herramientas": herramientas, "fuentes": fuentes})

    st.session_state["mensajes"].append({
        "rol":          "assistant",
        "contenido":    respuesta,
        "fuentes":      fuentes,
        "herramientas": herramientas,
    })
