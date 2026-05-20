"""
app.py — Interfaz web del Agente Ganadero (Streamlit).

Tabs:
  💬 Chat         — Conversación con el agente
  📊 Evaluación   — Análisis crítico del sistema
  🏗️ Arquitectura — Justificación y diseño del sistema
"""

import streamlit as st

# set_page_config DEBE ser el primer comando Streamlit — sin excepciones
st.set_page_config(
    page_title="Agente Ganadero IA",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Carga del agente + auto-ingestión ─────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelos e indexando documentos (puede tardar ~2 min la primera vez)...")
def _cargar_agente():
    from ingest import ingestar, EMBEDDINGS_FILE
    if not EMBEDDINGS_FILE.exists():
        ingestar(limpiar=True)
    from agent import AgenteGanadero
    return AgenteGanadero()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🐄 Agente Ganadero IA")
    st.caption("Monitoreo inteligente de ganado y tierras agrícolas")

    st.divider()
    st.subheader("Stack tecnológico")
    st.markdown("""
| Componente | Tecnología |
|---|---|
| LLM | llama-3.3-70b (Groq) |
| Embeddings | sentence-transformers |
| Vector store | NumPy coseno |
| Clima | Open-Meteo API |
| UI | Streamlit |
""")

    st.divider()
    st.subheader("Preguntas de ejemplo")
    EJEMPLOS = [
        "¿Qué vacunas hay para bovinos?",
        "¿Qué antibióticos se usan en porcinos?",
        "Potrero 8 ha, kikuyo, vacas 400 kg. ¿Cuántas caben?",
        "Clima en Bogotá hoy. lat=4.71, lon=-74.07",
        "¿Qué cultivos hay en la vereda Páramo?",
        "¿Qué antiparasitarios hay para equinos?",
        "¿Qué vitaminas existen para bovinos?",
        "¿Qué es la fiebre aftosa y cómo se previene?",
    ]
    for ejemplo in EJEMPLOS:
        if st.button(ejemplo, use_container_width=True, key=f"ej_{ejemplo[:20]}"):
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
            with st.spinner("Procesando..."):
                try:
                    from ingest import ingestar
                    ingestar(limpiar=True)
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))


# ── Helper de renderizado ──────────────────────────────────────────────────────
def _mostrar_detalles(msg: dict) -> None:
    herramientas = msg.get("herramientas", [])
    fuentes      = msg.get("fuentes", [])
    if herramientas:
        with st.expander(f"🔧 Herramientas utilizadas ({len(herramientas)})"):
            for h in herramientas:
                st.markdown(f"**`{h['herramienta']}`**")
                st.json({"argumentos": h["argumentos"], "resultado": h["resultado"]})
    if fuentes:
        with st.expander(f"📚 Fuentes RAG ({len(fuentes)})"):
            for f in fuentes:
                st.markdown(f"- `{f}`")


# ── Tabs principales ───────────────────────────────────────────────────────────
tab_chat, tab_eval, tab_arq = st.tabs(["💬 Chat", "📊 Evaluación del Agente", "🏗️ Arquitectura y Justificación"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.title("🐄 Agente Inteligente de Monitoreo Ganadero")
    st.caption("Consulta sobre salud animal, praderas, suelos y productos veterinarios")

    if "mensajes" not in st.session_state:
        st.session_state["mensajes"] = []

    for msg in st.session_state["mensajes"]:
        with st.chat_message(msg["rol"]):
            st.markdown(msg["contenido"])
            if "fuentes" in msg or "herramientas" in msg:
                _mostrar_detalles(msg)

    pregunta_sidebar = st.session_state.pop("pregunta_rapida", None)
    pregunta_chat    = st.chat_input("Escribe tu pregunta sobre ganado o tierras...")
    pregunta         = pregunta_sidebar or pregunta_chat

    if pregunta:
        with st.chat_message("user"):
            st.markdown(pregunta)
        st.session_state["mensajes"].append({"rol": "user", "contenido": pregunta})

        with st.chat_message("assistant"):
            with st.spinner("Consultando base de conocimiento..."):
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
            "rol": "assistant", "contenido": respuesta,
            "fuentes": fuentes, "herramientas": herramientas,
        })


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.title("📊 Evaluación del Agente Ganadero")
    st.markdown("""
Esta sección analiza el desempeño del agente en cuatro dimensiones críticas:
**recuperación RAG**, **uso de herramientas**, **manejo de límites del conocimiento**
y **riesgo de alucinación**.
""")

    from evaluacion import CASOS_PRUEBA, LIMITACIONES, MEJORAS_FUTURAS, ejecutar_retrieval

    # ── 2.1 Casos de prueba ───────────────────────────────────────────────
    st.subheader("2.1  Casos de prueba y comportamiento esperado")

    categorias = sorted(set(c["categoria"] for c in CASOS_PRUEBA))
    col_filtro, _ = st.columns([2, 3])
    with col_filtro:
        cat_sel = st.selectbox("Filtrar por categoría", ["Todas"] + categorias)

    casos_filtrados = CASOS_PRUEBA if cat_sel == "Todas" else [
        c for c in CASOS_PRUEBA if c["categoria"] == cat_sel
    ]

    for caso in casos_filtrados:
        icono = {"R": "🔍", "T": "🔧", "L": "🚧", "H": "⚠️"}.get(caso["id"][0], "•")
        with st.expander(f"{icono} [{caso['id']}] {caso['categoria']} — *{caso['pregunta'][:60]}...*"):
            st.markdown(f"**Pregunta completa:** {caso['pregunta']}")
            st.markdown(f"**Comportamiento esperado:** {caso['comportamiento_esperado']}")
            if caso.get("tool_esperada"):
                st.info(f"🔧 Herramienta esperada: `{caso['tool_esperada']}`")
            if caso.get("palabras_clave"):
                st.markdown(f"**Palabras clave en retrieval:** `{', '.join(caso['palabras_clave'])}`")

    # ── 2.2 Análisis de retrieval ─────────────────────────────────────────
    st.divider()
    st.subheader("2.2  Análisis cuantitativo del retrieval")
    st.markdown("""
Evaluación de la calidad del retrieval semántico para los casos RAG.
Se mide similitud coseno y cobertura de palabras clave sin consumir la API del LLM.
""")

    if st.button("▶ Ejecutar análisis de retrieval", type="primary"):
        with st.spinner("Ejecutando retrieval en todos los casos de prueba..."):
            try:
                from rag import RecuperadorRAG
                rag = RecuperadorRAG()
                resultados = ejecutar_retrieval(rag)
                st.session_state["resultados_eval"] = resultados
            except RuntimeError as e:
                st.error(f"Vectorstore no disponible: {e}. Carga el Chat primero para inicializar.")
                resultados = []

    resultados = st.session_state.get("resultados_eval", [])

    if resultados:
        # Métricas globales
        casos_rag = [r for r in resultados if r.get("similitud_maxima") is not None]
        if casos_rag:
            sim_prom_global = sum(r["similitud_promedio"] for r in casos_rag) / len(casos_rag)
            cob_prom_global = sum(r["cobertura_pct"] for r in casos_rag) / len(casos_rag)

            m1, m2, m3 = st.columns(3)
            m1.metric("Similitud promedio (RAG)", f"{sim_prom_global:.2%}")
            m2.metric("Cobertura de palabras clave", f"{cob_prom_global:.1f}%")
            m3.metric("Casos RAG evaluados", len(casos_rag))

        st.markdown("**Detalle por caso:**")
        for r in resultados:
            if r.get("similitud_maxima") is None:
                continue
            color = "🟢" if r["cobertura_pct"] >= 80 else ("🟡" if r["cobertura_pct"] >= 50 else "🔴")
            with st.expander(
                f"{color} [{r['id']}] {r['categoria']} — "
                f"Similitud máx: {r['similitud_maxima']:.2%} | Cobertura: {r['cobertura_pct']:.0f}%"
            ):
                st.markdown(f"**Palabras buscadas:** `{', '.join(r.get('palabras_buscadas', []))}`")
                st.markdown(f"**Palabras encontradas:** `{', '.join(r.get('palabras_encontradas', []))}`")
                st.markdown("**Top 3 chunks recuperados:**")
                for i, chunk in enumerate(r.get("chunks", [])[:3], 1):
                    st.markdown(
                        f"*Chunk {i}* | Fuente: `{chunk['fuente']}` | "
                        f"Similitud: `{chunk['similitud']:.2%}`"
                    )
                    st.text(chunk["texto"][:300] + "..." if len(chunk["texto"]) > 300 else chunk["texto"])

        # Casos de tools y límites
        st.markdown("**Casos de herramientas y límites (evaluación cualitativa):**")
        for r in resultados:
            if r.get("similitud_maxima") is not None:
                continue
            icono = "🔧" if r["id"].startswith("T") else ("🚧" if r["id"].startswith("L") else "⚠️")
            with st.expander(f"{icono} [{r['id']}] {r['categoria']}"):
                st.markdown(f"**Pregunta:** {r['pregunta']}")
                st.info(r["nota"])

    # ── 2.3 Limitaciones ──────────────────────────────────────────────────
    st.divider()
    st.subheader("2.3  Limitaciones identificadas")
    st.markdown("Análisis crítico de las debilidades del sistema en su estado actual.")

    impacto_color = {"Alto": "🔴", "Medio": "🟡", "Bajo": "🟢"}
    for lim in LIMITACIONES:
        with st.expander(
            f"{impacto_color[lim['impacto']]} [{lim['impacto']}] {lim['titulo']} — *{lim['categoria']}*"
        ):
            st.markdown(lim["descripcion"])

    # ── 2.4 Mejoras futuras ────────────────────────────────────────────────
    st.divider()
    st.subheader("2.4  Mejoras futuras propuestas")

    prioridad_color = {"Alta": "🔴", "Media": "🟡", "Baja": "🟢"}
    for mej in MEJORAS_FUTURAS:
        st.markdown(
            f"{prioridad_color[mej['prioridad']]} **{mej['mejora']}** *(prioridad {mej['prioridad']})*  \n"
            f"{mej['descripcion']}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ARQUITECTURA Y JUSTIFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════
with tab_arq:
    st.title("🏗️ Arquitectura y Justificación del Agente")

    # ── 3.1 Definición del problema ───────────────────────────────────────
    st.header("3.1  Definición del problema")
    st.markdown("""
El sector ganadero y agrícola colombiano enfrenta una brecha de acceso a información técnica
especializada. Los productores rurales —en su mayoría pequeños y medianos ganaderos— necesitan
tomar decisiones diarias sobre:

- **Sanidad animal**: qué medicamento usar, cuándo vacunar, cómo identificar enfermedades
- **Manejo de praderas**: cuántos animales puede soportar un potrero, cuándo rotar
- **Análisis de suelos**: qué cultivos son viables según la zona y el tipo de suelo

El acceso a veterinarios especializados es costoso y geográficamente limitado en zonas rurales.
Los documentos técnicos (manuales del ICA, guías de praderas) existen pero son difíciles de
consultar rápidamente en campo.

### ¿Por qué no basta con un chatbot simple?

| Enfoque | Limitación principal |
|---|---|
| Chatbot basado en reglas (FAQ) | No escala ante la variedad de preguntas; requiere programar cada respuesta manualmente |
| Búsqueda de palabras clave | Devuelve documentos completos sin sintetizar la respuesta relevante |
| LLM sin RAG | Alucina datos veterinarios específicos (dosificaciones, nombres de productos) que no están en su entrenamiento |
| RAG sin tools | No puede responder preguntas que requieren datos externos (clima, cálculos dinámicos) |

### ¿Por qué RAG + LLM + Tools es la solución adecuada?

- **RAG** ancla las respuestas del modelo en documentos verificados, reduciendo alucinaciones
- **LLM** entiende preguntas en lenguaje natural y sintetiza información de múltiples fragmentos
- **Tools** permiten datos en tiempo real (clima) y cálculos precisos (carga animal)
- **Arquitectura de agente** permite al modelo decidir cuándo usar cada herramienta según el contexto
""")

    # ── 3.2 Diagrama de arquitectura ──────────────────────────────────────
    st.header("3.2  Diagrama de arquitectura")
    st.code("""
┌─────────────────────────────────────────────────────────────────────┐
│                        AGENTE GANADERO IA                           │
└─────────────────────────────────────────────────────────────────────┘

 PIPELINE DE INGESTIÓN (ingest.py)         PIPELINE DE CONSULTA
 ─────────────────────────────────         ─────────────────────────
                                            Pregunta del usuario
 data/                                             │
 ├── *.pdf   → pypdf                               ▼
 ├── *.txt   → read_text              ┌────────────────────────┐
 └── *.csv   → pandas                │  sentence-transformers │
      │                              │  (embedding consulta)  │
      ▼                              └────────────┬───────────┘
 Texto plano                                      │
      │                                           ▼
      ▼                              ┌────────────────────────┐
 Chunking (350 palabras,             │  Similitud coseno      │
 solapamiento 60)                    │  sobre embeddings.npy  │◄──── vectorstore/
      │                              └────────────┬───────────┘      embeddings.npy
      ▼                                           │                   metadata.json
 sentence-transformers               Top-K chunks relevantes
 (multilingual embeddings)                        │
      │                                           ▼
      ▼                              ┌────────────────────────┐
 vectorstore/                        │  Prompt del sistema    │
 ├── embeddings.npy  (N × 384)       │  + contexto RAG        │
 └── metadata.json                   │  + historial (4 turnos)│
                                     └────────────┬───────────┘
                                                  │
                                                  ▼
                                     ┌────────────────────────┐
                                     │  Groq API              │
                                     │  llama-3.3-70b-versatile│
                                     │  (tool_choice="auto")  │
                                     └────────────┬───────────┘
                                          │       │
                              ┌───────────┘       └───────────┐
                              ▼                               ▼
                   ┌──────────────────┐           ┌──────────────────┐
                   │ obtener_clima()  │           │calcular_carga_   │
                   │ Open-Meteo API   │           │animal()          │
                   │ (sin API key)    │           │(fórmula UA/ha)   │
                   └────────┬─────────┘           └────────┬─────────┘
                            │                              │
                            └──────────────┬───────────────┘
                                           ▼
                                  Resultado de tool(s)
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │  Respuesta final       │
                               │  en español            │
                               └───────────────────────┘
""", language=None)

    # ── 3.3 Decisiones de diseño justificadas ─────────────────────────────
    st.header("3.3  Decisiones de diseño justificadas")

    with st.expander("📦 Chunking: 350 palabras, solapamiento de 60"):
        st.markdown("""
**¿Por qué 350 palabras?**
- Chunks muy pequeños (<100 palabras) pierden contexto clínico; por ejemplo, una descripción
  de vacuna que ocupa 3 líneas quedaría dividida y el retriever podría no recuperarla completa.
- Chunks muy grandes (>500 palabras) incluyen información irrelevante que "contamina" el contexto
  del LLM y reduce la precisión de las respuestas.
- 350 palabras representa un equilibrio testado empíricamente en literatura RAG para documentos
  técnicos de extensión media.

**¿Por qué 60 palabras de solapamiento?**
- Evita que información crítica (una dosis, un nombre de producto) quede partido exactamente en
  el borde entre dos chunks consecutivos.
- El 17% de solapamiento (60/350) es el valor estándar recomendado en la mayoría de frameworks RAG.
""")

    with st.expander("🧠 Modelo de embeddings: paraphrase-multilingual-MiniLM-L12-v2"):
        st.markdown("""
**¿Por qué este modelo?**
- **Multilingüe**: soporta español nativo sin necesidad de traducir los documentos al inglés.
- **Ligero**: 120 MB vs. ~1 GB de modelos más grandes; viable en servidores con poca RAM.
- **Calidad**: en benchmarks MTEB para español supera a modelos en inglés como `all-MiniLM-L6-v2`
  en tareas de similitud semántica.
- **Sin GPU**: corre en CPU, compatible con Streamlit Cloud free tier.

**Alternativas consideradas y descartadas:**
- `all-MiniLM-L6-v2`: inglés únicamente, degradaría la recuperación de texto en español.
- `text-embedding-ada-002` (OpenAI): requiere API key de pago por cada embedding generado.
- Modelos grandes (>500M parámetros): exceden la memoria de Streamlit Cloud free tier.
""")

    with st.expander("📐 Métrica de similitud: coseno en lugar de distancia euclidiana"):
        st.markdown("""
**¿Por qué similitud coseno?**
- Los embeddings de texto tienen distancias euclidianas que dependen de la magnitud del vector,
  la cual no refleja similitud semántica.
- La similitud coseno mide el ángulo entre vectores, invariante a la longitud, capturando
  mejor si dos textos hablan de lo mismo independientemente de su extensión.
- Es el estándar en todos los sistemas RAG modernos (ChromaDB, Pinecone, FAISS).

**Implementación eficiente:** los embeddings se pre-normalizan a norma 1 al cargar el vectorstore;
la búsqueda se reduce a un producto matricial `E @ q` (O(N·D)), ejecutable en milisegundos
para N < 10.000 chunks.
""")

    with st.expander("🤖 LLM: llama-3.3-70b-versatile vía Groq"):
        st.markdown("""
**¿Por qué Groq + llama-3.3-70b?**
- **Gratuito**: Groq ofrece un tier free con límites generosos para proyectos académicos.
- **Velocidad**: la infraestructura LPU de Groq genera tokens ~10× más rápido que GPUs convencionales,
  crucial para una experiencia conversacional fluida.
- **Tool calling nativo**: llama-3.3-70b soporta function calling en el formato OpenAI,
  permitiendo que el agente decida autónomamente cuándo usar herramientas.
- **Contexto de 128K tokens**: suficiente para conversaciones largas con contexto RAG.

**¿Por qué no GPT-4 / Claude / Gemini?**
- Todos requieren API key de pago. Para un proyecto académico desplegado públicamente,
  Groq es la única opción gratuita con un modelo de calidad equivalente.
""")

    with st.expander("🔄 Historial de conversación: 4 turnos (8 mensajes)"):
        st.markdown("""
**Trade-off memoria vs. costo de contexto:**
- Conservar todo el historial haría crecer el prompt indefinidamente, aumentando latencia y
  riesgo de superar el límite de tokens con el contexto RAG incluido.
- 4 turnos (≈ 8 mensajes) permiten preguntas de seguimiento naturales ("¿y para equinos?",
  "¿cuánto de ese antibiótico se usa?") sin saturar el contexto.
- En una versión productiva se implementaría memoria con compresión semántica (MemGPT/LangMem).
""")

    with st.expander("🛠️ Arquitectura de agente con tool calling en lugar de pipeline fijo"):
        st.markdown("""
**¿Por qué tool calling en lugar de un pipeline determinista?**
- Un pipeline fijo llamaría SIEMPRE a clima y SIEMPRE a la calculadora, incluso cuando
  la pregunta es sobre vacunas y no necesita ninguna herramienta. Esto añade latencia innecesaria.
- Con `tool_choice="auto"`, el LLM decide dinámicamente si la pregunta requiere clima,
  cálculo, solo RAG, o ninguno de los anteriores.
- Esto hace el agente más eficiente y sus respuestas más coherentes con la pregunta.

**Ciclo de ejecución (ReAct simplificado):**
```
Pregunta → RAG context → LLM
  Si LLM llama herramienta:
    ejecutar herramienta → resultado → LLM (hasta 6 iteraciones)
  Si no:
    respuesta final directa
```
""")

    # ── 3.4 Fuentes de datos ──────────────────────────────────────────────
    st.header("3.4  Fuentes de datos indexadas")
    st.markdown("""
| Archivo | Contenido | Registros | Relevancia |
|---|---|---|---|
| `Datos_Abiertos_Animales_de_Pro.csv` | Catálogo de productos veterinarios: vacunas, antibióticos, antiparasitarios, hormonales, vitaminas, analgésicos | ~160 productos × especie | Alta: permite responder qué producto usar según especie y enfermedad |
| `REGISTRO_ANALISIS_DE_SUELOS.csv` | Registro de predios con análisis de suelos: vereda, predio, tipo de cultivo | 121 registros | Media: permite contextualizar cultivos por zona geográfica |
| `manual_bovinos.pdf` *(pendiente)* | Manual veterinario bovino | — | Alta: cubriría enfermedades, síntomas y protocolos clínicos |
| `guia_praderas.txt` *(pendiente)* | Guía de manejo de praderas y rotación | — | Alta: cubriría forrajes, establecimiento y fertilización |

> Los dos últimos archivos se procesan automáticamente al colocarlos en `data/` y ejecutar Re-ingestar.
""")
