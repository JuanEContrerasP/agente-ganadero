"""
evaluacion.py — Marco de evaluación del Agente Ganadero.

Organiza 9 casos de prueba en 4 categorías:
  R — Recuperación RAG (base de conocimiento)
  T — Uso de herramientas externas (tools)
  L — Límites del conocimiento (out-of-scope)
  H — Riesgo de alucinación

La función ejecutar_retrieval() evalúa la calidad del retrieval
SIN consumir la API de Groq, calculando métricas sobre los chunks devueltos.
"""

from __future__ import annotations
from dataclasses import dataclass, field

# ── Casos de prueba ────────────────────────────────────────────────────────────

CASOS_PRUEBA: list[dict] = [
    # ── Categoría R: Recuperación de la base de conocimiento ──────────────
    {
        "id": "R01",
        "categoria": "RAG — Vacunas",
        "pregunta": "¿Qué vacunas están disponibles para bovinos?",
        "tool_esperada": None,
        "palabras_clave": ["vacuna", "bovinos", "aftogan", "rayovacuna", "rabigan"],
        "comportamiento_esperado": (
            "Listar vacunas del catálogo para bovinos: AFTOGAN, RAYOVACUNA, "
            "RABIGAN, CEPA 19, BLINDAGAN, HEXAGAN, VACUNA TRIPLE HA."
        ),
    },
    {
        "id": "R02",
        "categoria": "RAG — Antibióticos por especie",
        "pregunta": "¿Qué antibióticos se pueden usar en porcinos y en qué presentaciones?",
        "tool_esperada": None,
        "palabras_clave": ["antibiótico", "porcinos", "enrofloxacina", "longicilina", "oxitetraciclina"],
        "comportamiento_esperado": (
            "Mencionar antibióticos disponibles para porcinos con sus presentaciones: "
            "ENROFLOXACINA 10% (50-100 ml), LONGICILINA 200 (500 ml), OXITETRACICLINA, "
            "TILOSINA, TRIPEN L.A., SULFAMETAZINA."
        ),
    },
    {
        "id": "R03",
        "categoria": "RAG — Análisis de suelos",
        "pregunta": "¿Qué cultivos predominan en la vereda Páramo según el registro de suelos?",
        "tool_esperada": None,
        "palabras_clave": ["paramo", "mora", "papa", "tomate", "arveja"],
        "comportamiento_esperado": (
            "Reportar cultivos registrados en vereda Páramo: mora uva, papa, "
            "tomate de árbol, arveja, lulo, hortalizas. Mencionar predios específicos."
        ),
    },
    {
        "id": "R04",
        "categoria": "RAG — Antiparasitarios equinos",
        "pregunta": "¿Qué productos antiparasitarios hay disponibles para equinos?",
        "tool_esperada": None,
        "palabras_clave": ["antiparasitario", "equinos", "iverquinos", "fenbendazol"],
        "comportamiento_esperado": (
            "Mencionar IVERQUINOS (jeringa plástica 6,42 g) y FENBENDAZOL 25% "
            "como antiparasitarios registrados para equinos."
        ),
    },
    # ── Categoría T: Uso de herramientas ──────────────────────────────────
    {
        "id": "T01",
        "categoria": "Tool — Calculadora carga animal",
        "pregunta": (
            "Tengo un potrero de 10 hectáreas con pasto kikuyo "
            "(disponibilidad 2500 kg MS/ha). ¿Cuántas vacas de 450 kg puedo sostener?"
        ),
        "tool_esperada": "calcular_carga_animal",
        "palabras_clave": [],
        "comportamiento_esperado": (
            "Invocar calcular_carga_animal(area_ha=10, peso_promedio_kg=450, "
            "disponibilidad_forraje_kg_ha=2500). "
            "Resultado esperado: ~18–19 animales por 30 días, carga ~0.83 UA/ha (adecuada)."
        ),
    },
    {
        "id": "T02",
        "categoria": "Tool — Consulta de clima",
        "pregunta": "¿Cómo está el clima hoy en Medellín? lat=6.22, lon=-75.57",
        "tool_esperada": "obtener_clima",
        "palabras_clave": [],
        "comportamiento_esperado": (
            "Invocar obtener_clima(latitud=6.22, longitud=-75.57) y reportar "
            "temperatura, humedad relativa, precipitación y pronóstico de 3 días."
        ),
    },
    # ── Categoría L: Límites del conocimiento ─────────────────────────────
    {
        "id": "L01",
        "categoria": "Límite — Precios en tiempo real",
        "pregunta": "¿Cuál es el precio actual de la ivermectina 1% en Colombia?",
        "tool_esperada": None,
        "palabras_clave": [],
        "comportamiento_esperado": (
            "Reconocer que no dispone de información de precios en tiempo real. "
            "NO debe inventar un precio. Puede orientar sobre dónde consultarlo (ICA, proveedores)."
        ),
    },
    {
        "id": "L02",
        "categoria": "Límite — Predicción futura",
        "pregunta": "¿Cuánto valdrá el kilo de carne de res en diciembre?",
        "tool_esperada": None,
        "palabras_clave": [],
        "comportamiento_esperado": (
            "Indicar que no puede predecir precios futuros. "
            "Puede mencionar factores que influyen (clima, demanda, exportaciones) "
            "pero no debe fabricar una cifra."
        ),
    },
    # ── Categoría H: Riesgo de alucinación ────────────────────────────────
    {
        "id": "H01",
        "categoria": "Alucinación — Dosis clínicas",
        "pregunta": "¿Cuál es la dosis exacta de enrofloxacina para un bovino de 300 kg con neumonía?",
        "tool_esperada": None,
        "palabras_clave": ["enrofloxacina", "bovinos"],
        "comportamiento_esperado": (
            "Puede orientar sobre rangos típicos (2.5–5 mg/kg SC una vez al día), "
            "pero DEBE recomendar consultar un médico veterinario. "
            "Riesgo: el modelo podría inventar una dosis o protocolo no verificado."
        ),
    },
]

# ── Análisis de limitaciones ───────────────────────────────────────────────────

LIMITACIONES: list[dict] = [
    {
        "titulo": "Base de conocimiento pequeña",
        "descripcion": (
            "Actualmente solo se indexan 2 archivos CSV (productos veterinarios y análisis de suelos). "
            "No se incluyen manuales clínicos, guías del ICA ni literatura científica. "
            "El agente no puede responder sobre enfermedades, procedimientos quirúrgicos "
            "ni protocolos de manejo que no están en los datos."
        ),
        "impacto": "Alto",
        "categoria": "Conocimiento",
    },
    {
        "titulo": "Riesgo de alucinación en dosis y procedimientos",
        "descripcion": (
            "Los LLMs pueden generar dosis, nombres de principios activos o protocolos "
            "aparentemente correctos pero incorrectos. El prompt del sistema instruye al "
            "modelo a recomendar veterinario, pero no elimina el riesgo."
        ),
        "impacto": "Alto",
        "categoria": "Confiabilidad",
    },
    {
        "titulo": "Sin datos en tiempo real (precios, epidemias)",
        "descripcion": (
            "El agente no tiene acceso a precios del SIPSA/DANE, alertas epidemiológicas "
            "del ICA ni boletines de enfermedades emergentes. La herramienta de clima "
            "es la única fuente de datos en tiempo real."
        ),
        "impacto": "Medio",
        "categoria": "Actualidad",
    },
    {
        "titulo": "Vectorstore no persistente en Streamlit Cloud",
        "descripcion": (
            "Streamlit Cloud no garantiza almacenamiento permanente en disco. "
            "En cada reinicio frío el vectorstore se regenera (~60 s). "
            "En producción real debería usarse un servicio de almacenamiento externo."
        ),
        "impacto": "Medio",
        "categoria": "Infraestructura",
    },
    {
        "titulo": "Historial de conversación limitado",
        "descripcion": (
            "El agente conserva solo los últimos 4 turnos (8 mensajes) para evitar "
            "exceder el contexto del LLM. En conversaciones largas puede perder "
            "referencias a preguntas anteriores."
        ),
        "impacto": "Bajo",
        "categoria": "Experiencia de usuario",
    },
    {
        "titulo": "Tool de clima requiere coordenadas exactas",
        "descripcion": (
            "El usuario debe conocer latitud y longitud de su ubicación. "
            "Una mejora sería integrar geocodificación para aceptar el nombre del municipio."
        ),
        "impacto": "Bajo",
        "categoria": "Usabilidad",
    },
]

MEJORAS_FUTURAS: list[dict] = [
    {
        "mejora": "Integración con API del ICA",
        "descripcion": "Consultar alertas sanitarias y registros de medicamentos vigentes en tiempo real.",
        "prioridad": "Alta",
    },
    {
        "mejora": "Base de conocimiento expandida",
        "descripcion": "Incorporar manuales clínicos bovinos, guías de praderas del CIAT y artículos del AGROSAVIA.",
        "prioridad": "Alta",
    },
    {
        "mejora": "Precios de mercado (SIPSA/DANE)",
        "descripcion": "Tool adicional para consultar precios de insumos y productos pecuarios.",
        "prioridad": "Alta",
    },
    {
        "mejora": "Geocodificación en herramienta de clima",
        "descripcion": "Aceptar nombre de municipio colombiano y resolver coordenadas automáticamente.",
        "prioridad": "Media",
    },
    {
        "mejora": "Evaluación con retroalimentación del usuario",
        "descripcion": "Botones de 👍/👎 por respuesta para construir un dataset de evaluación humana.",
        "prioridad": "Media",
    },
    {
        "mejora": "Diagnóstico por imagen",
        "descripcion": "Módulo multimodal para analizar fotos de animales enfermos o lesiones en praderas.",
        "prioridad": "Baja",
    },
    {
        "mejora": "Fine-tuning con datos ganaderos colombianos",
        "descripcion": "Ajustar el modelo con terminología local, razas criollas y condiciones del trópico.",
        "prioridad": "Baja",
    },
]

# ── Evaluación de retrieval (sin API de Groq) ──────────────────────────────────

def ejecutar_retrieval(rag) -> list[dict]:
    """
    Evalúa la calidad del retrieval para los casos con palabras_clave definidas.
    No consume la API de Groq: solo mide qué tan bien el vector store
    recupera información relevante para cada pregunta.

    Returns:
        Lista de dicts con: id, pregunta, chunks_recuperados,
        similitud_maxima, similitud_promedio, palabras_encontradas, cobertura_pct
    """
    resultados = []

    for caso in CASOS_PRUEBA:
        if not caso["palabras_clave"]:
            # Casos de tools o límites: no evalúan retrieval
            resultados.append({
                "id":                caso["id"],
                "categoria":         caso["categoria"],
                "pregunta":          caso["pregunta"],
                "chunks":            [],
                "similitud_maxima":  None,
                "similitud_promedio": None,
                "cobertura_pct":     None,
                "nota":              "Evaluado por uso de herramienta o comportamiento del LLM",
            })
            continue

        try:
            chunks = rag.recuperar(caso["pregunta"], k=5)
            sims   = [c["similitud"] for c in chunks]

            # Contar cuántas palabras clave aparecen en los chunks recuperados
            texto_recuperado = " ".join(c["texto"] for c in chunks).lower()
            encontradas = [
                kw for kw in caso["palabras_clave"]
                if kw.lower() in texto_recuperado
            ]
            cobertura = len(encontradas) / len(caso["palabras_clave"]) * 100

            resultados.append({
                "id":                 caso["id"],
                "categoria":          caso["categoria"],
                "pregunta":           caso["pregunta"],
                "chunks":             chunks,
                "similitud_maxima":   round(max(sims), 4),
                "similitud_promedio": round(sum(sims) / len(sims), 4),
                "cobertura_pct":      round(cobertura, 1),
                "palabras_encontradas": encontradas,
                "palabras_buscadas":  caso["palabras_clave"],
                "nota":               caso["comportamiento_esperado"],
            })
        except Exception as exc:
            resultados.append({
                "id":       caso["id"],
                "categoria": caso["categoria"],
                "pregunta":  caso["pregunta"],
                "chunks":    [],
                "error":     str(exc),
                "nota":      "Error durante evaluación",
            })

    return resultados
