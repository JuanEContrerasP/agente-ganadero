"""
agent.py — Núcleo del Agente Ganadero.

Arquitectura:
  Pregunta del usuario
       ↓
  RAG (ChromaDB) → contexto inyectado en el sistema
       ↓
  Groq LLM (llama-3.3-70b-versatile) con tool calling
       ↓
  [Si el modelo llama una herramienta] → ejecutar → devolver resultado al LLM
       ↓
  Respuesta final en español

El historial de conversación se mantiene por sesión para preguntas de seguimiento.
"""

import json
import os

from dotenv import load_dotenv
from groq import Groq

from rag import RecuperadorRAG
from tools import DEFINICIONES_HERRAMIENTAS, MAPA_HERRAMIENTAS

load_dotenv()

# ── Constantes ─────────────────────────────────────────────────────────────────
MODELO_GROQ    = "llama-3.3-70b-versatile"
MAX_ITERACIONES = 6   # ciclos herramienta máximos antes de forzar respuesta
HISTORIAL_TURNOS = 4  # turnos (usuario+asistente) que se conservan en contexto

# ── Prompt del sistema ─────────────────────────────────────────────────────────
# El contexto RAG se inyecta dinámicamente antes de cada llamada al LLM.
_SISTEMA_TEMPLATE = """\
Eres un asistente experto en ganadería bovina y agricultura colombiana, \
diseñado para apoyar a ganaderos y técnicos agropecuarios.

Áreas de conocimiento:
• Salud animal: enfermedades, vacunas, antibióticos, antiparasitarios, dosificación
• Praderas y forrajes: establecimiento, rotación, carga animal, fertilización
• Suelos: interpretación de análisis, cultivos por zona, enmiendas
• Productos veterinarios: clases, presentaciones, especies objetivo

Instrucciones:
1. Responde siempre en español claro y práctico para ganaderos colombianos.
2. Cuando la pregunta involucre una región o municipio, llama a obtener_clima.
3. Cuando pregunten cuántos animales caben en un potrero, usa calcular_carga_animal.
4. Cita siempre la fuente del contexto si la usas (ej: "Según {fuentes}...").
5. Para medicamentos, recuerda recomendar consultar un médico veterinario.
6. Si el contexto no tiene información suficiente, respóndelo honestamente.

━━━ CONTEXTO RECUPERADO DE LA BASE DE CONOCIMIENTO ━━━
{contexto}

Fuentes: {fuentes}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


class AgenteGanadero:
    """
    Agente conversacional RAG + herramientas sobre Groq.

    Uso:
        agente = AgenteGanadero()
        respuesta, fuentes, herramientas_usadas = agente.responder("¿Qué vacunas...")
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "tu_clave_aqui":
            raise ValueError(
                "GROQ_API_KEY no configurada. "
                "Edita el archivo .env y coloca tu clave de https://console.groq.com/keys"
            )

        self._groq    = Groq(api_key=api_key)
        self._rag     = RecuperadorRAG()
        self._historial: list[dict] = []

    # ── API pública ────────────────────────────────────────────────────────────

    def responder(self, pregunta: str) -> tuple[str, list[str], list[dict]]:
        """
        Procesa una pregunta y devuelve la respuesta con metadatos.

        Returns:
            respuesta       — texto final del asistente
            fuentes         — lista de archivos consultados por RAG
            herr_usadas     — lista de dicts {herramienta, argumentos, resultado}
        """
        # 1. Recuperar contexto semántico relevante para esta pregunta
        contexto, fuentes = self._rag.construir_contexto(pregunta)

        # 2. Construir prompt de sistema con contexto inyectado
        sistema = _SISTEMA_TEMPLATE.format(
            contexto=contexto,
            fuentes=", ".join(fuentes) if fuentes else "ninguna",
        )

        # 3. Armar la lista de mensajes: sistema + historial reciente + pregunta
        mensajes: list[dict] = [{"role": "system", "content": sistema}]
        # Solo los últimos N turnos para no exceder el contexto
        mensajes.extend(self._historial[-(HISTORIAL_TURNOS * 2):])
        mensajes.append({"role": "user", "content": pregunta})

        herr_usadas: list[dict] = []
        respuesta_final = ""

        # 4. Ciclo agente: LLM → (opcional) herramienta → LLM → ...
        for _ in range(MAX_ITERACIONES):
            completado = self._groq.chat.completions.create(
                model=MODELO_GROQ,
                messages=mensajes,
                tools=DEFINICIONES_HERRAMIENTAS,
                tool_choice="auto",
                temperature=0.25,   # más determinista para datos técnicos
                max_tokens=1200,
            )

            msg = completado.choices[0].message

            # Si no hay llamadas a herramientas → el agente terminó
            if not msg.tool_calls:
                respuesta_final = msg.content or ""
                break

            # Agregar la respuesta del asistente con las tool_calls al hilo
            mensajes.append(msg)

            # Ejecutar cada herramienta solicitada y devolver el resultado
            for tc in msg.tool_calls:
                nombre   = tc.function.name
                args     = json.loads(tc.function.arguments)
                funcion  = MAPA_HERRAMIENTAS.get(nombre)

                if funcion:
                    resultado = funcion(**args)
                else:
                    resultado = {"error": f"Herramienta desconocida: '{nombre}'"}

                herr_usadas.append({
                    "herramienta": nombre,
                    "argumentos":  args,
                    "resultado":   resultado,
                })

                mensajes.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(resultado, ensure_ascii=False),
                })

        else:
            # Se alcanzó el límite de iteraciones sin respuesta final
            respuesta_final = (
                "No pude completar la respuesta en el número máximo de pasos. "
                "Por favor reformula la pregunta o intenta de nuevo."
            )

        # 5. Actualizar historial (sin el mensaje de sistema para no duplicarlo)
        self._historial.append({"role": "user",      "content": pregunta})
        self._historial.append({"role": "assistant", "content": respuesta_final})

        return respuesta_final, fuentes, herr_usadas

    def limpiar_historial(self) -> None:
        """Reinicia la conversación eliminando el historial de turnos previos."""
        self._historial.clear()
