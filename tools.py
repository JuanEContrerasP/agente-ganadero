"""
tools.py — Herramientas externas del Agente Ganadero.

Herramienta 1: obtener_clima
  - Fuente: Open-Meteo (https://open-meteo.com) — gratuita, sin clave
  - Devuelve condiciones actuales + pronóstico de 3 días

Herramienta 2: calcular_carga_animal
  - Calcula Unidades Animal por hectárea (UA/ha) según disponibilidad de forraje
  - Clasifica si la carga es subutilizada, adecuada, alta o en sobrecarga

Al final del módulo se definen DEFINICIONES_HERRAMIENTAS (formato Groq/OpenAI)
y MAPA_HERRAMIENTAS para despachar llamadas dinámicamente desde el agente.
"""

import requests


# ── Herramienta 1: Clima ───────────────────────────────────────────────────────

def obtener_clima(latitud: float, longitud: float) -> dict:
    """
    Consulta el clima actual y el pronóstico de 3 días para las coordenadas dadas.

    Args:
        latitud:  Latitud decimal. Ejemplos colombianos: Bogotá=4.71, Medellín=6.22
        longitud: Longitud decimal. Ejemplos: Bogotá=-74.07, Medellín=-75.57

    Returns:
        Dict con sección 'actual' y lista 'pronostico_3dias', o 'estado': 'error'.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  latitud,
        "longitude": longitud,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "weather_code",
        ],
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
        ],
        "timezone":     "America/Bogota",
        "forecast_days": 3,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        datos = resp.json()
    except requests.RequestException as exc:
        return {"estado": "error", "mensaje": f"No se pudo conectar a Open-Meteo: {exc}"}

    actual  = datos["current"]
    diario  = datos["daily"]

    pronostico = []
    for i in range(len(diario["time"])):
        pronostico.append({
            "fecha":           diario["time"][i],
            "temp_max_c":      diario["temperature_2m_max"][i],
            "temp_min_c":      diario["temperature_2m_min"][i],
            "precipitacion_mm": diario["precipitation_sum"][i],
        })

    return {
        "estado": "éxito",
        "actual": {
            "temperatura_c":    actual["temperature_2m"],
            "humedad_pct":      actual["relative_humidity_2m"],
            "precipitacion_mm": actual["precipitation"],
            "viento_kmh":       actual["wind_speed_10m"],
            "descripcion":      _wmo_a_texto(actual.get("weather_code", 0)),
        },
        "pronostico_3dias": pronostico,
    }


def _wmo_a_texto(codigo: int) -> str:
    """Convierte el código WMO de clima a descripción en español."""
    tabla = {
        0: "Despejado",
        1: "Principalmente despejado", 2: "Parcialmente nublado", 3: "Nublado",
        45: "Niebla", 48: "Niebla con escarcha",
        51: "Llovizna ligera", 53: "Llovizna moderada", 55: "Llovizna densa",
        61: "Lluvia ligera", 63: "Lluvia moderada", 65: "Lluvia fuerte",
        71: "Nevada ligera", 73: "Nevada moderada", 75: "Nevada fuerte",
        80: "Aguaceros ligeros", 81: "Aguaceros moderados", 82: "Aguaceros fuertes",
        95: "Tormenta eléctrica", 96: "Tormenta con granizo leve", 99: "Tormenta con granizo fuerte",
    }
    return tabla.get(codigo, f"Condición código {codigo}")


# ── Herramienta 2: Carga animal ───────────────────────────────────────────────

def calcular_carga_animal(
    area_ha: float,
    peso_promedio_kg: float,
    disponibilidad_forraje_kg_ha: float,
    tasa_consumo_pct: float = 3.0,
) -> dict:
    """
    Calcula la capacidad de carga animal de un potrero para 30 días de pastoreo.

    Fórmula base:
      UA = peso_animal / 450 kg  (1 Unidad Animal = bovino adulto de 450 kg)
      Forraje total  = area_ha × disponibilidad_kg_ha
      Consumo diario = tasa_consumo_pct% × peso_promedio_kg
      Animales_30d   = forraje_total / (consumo_diario × 30)
      Carga (UA/ha)  = (animales × UA_por_animal) / area_ha

    Args:
        area_ha:                     Área del potrero en hectáreas.
        peso_promedio_kg:            Peso promedio del animal en kg.
        disponibilidad_forraje_kg_ha: Materia seca de forraje disponible (kg MS/ha).
                                      Kikuyo: 2000-3000 | Brachiaria: 1500-2500 | Natural: 800-1500
        tasa_consumo_pct:            % del peso vivo consumido/día en MS (default 3 %).

    Returns:
        Dict con métricas de carga y clasificación agronómica.
    """
    if any(v <= 0 for v in (area_ha, peso_promedio_kg, disponibilidad_forraje_kg_ha)):
        return {"estado": "error", "mensaje": "Todos los valores numéricos deben ser positivos."}

    UA_ESTANDAR = 450.0  # kg — referencia internacional para Colombia (ICA)

    ua_por_animal    = peso_promedio_kg / UA_ESTANDAR
    forraje_total_kg = area_ha * disponibilidad_forraje_kg_ha
    consumo_diario_kg = (tasa_consumo_pct / 100.0) * peso_promedio_kg

    # Animales que puede sostener el potrero durante 30 días
    animales_30d = forraje_total_kg / (consumo_diario_kg * 30)

    ua_totales       = animales_30d * ua_por_animal
    carga_ua_ha      = ua_totales / area_ha

    clasificacion, recomendacion = _clasificar_carga(carga_ua_ha, forraje_total_kg)

    return {
        "estado":                    "éxito",
        "area_ha":                   area_ha,
        "forraje_disponible_kg":     round(forraje_total_kg, 1),
        "consumo_diario_por_animal_kg": round(consumo_diario_kg, 1),
        "animales_soportados_30_dias": round(animales_30d, 1),
        "ua_por_animal":             round(ua_por_animal, 3),
        "carga_ua_ha":               round(carga_ua_ha, 2),
        "clasificacion":             clasificacion,
        "recomendacion":             recomendacion,
    }


def _clasificar_carga(carga_ua_ha: float, forraje_total_kg: float) -> tuple[str, str]:
    """Devuelve (clasificación, recomendación) según la carga en UA/ha."""
    if carga_ua_ha < 0.8:
        return (
            "Subutilizado",
            f"Forraje disponible suficiente ({forraje_total_kg:.0f} kg). "
            "Puede incrementar la carga o reducir el área por rotación.",
        )
    elif carga_ua_ha <= 1.5:
        return (
            "Carga adecuada",
            "Carga óptima para pasturas mejoradas. "
            "Mantenga rotación cada 28-35 días y monitoree mensualmente.",
        )
    elif carga_ua_ha <= 2.5:
        return (
            "Carga alta",
            "Implemente rotación estricta cada 21-28 días, "
            "siembre leguminosas (kudzú, maní forrajero) y fertilice.",
        )
    else:
        return (
            "SOBRECARGA CRÍTICA",
            "Reduzca el número de animales o amplíe el área de inmediato. "
            "Riesgo de compactación del suelo y degradación irreversible de la pradera.",
        )


# ── Definiciones de herramientas para la API de Groq ─────────────────────────
# Formato compatible con OpenAI function calling

DEFINICIONES_HERRAMIENTAS = [
    {
        "type": "function",
        "function": {
            "name": "obtener_clima",
            "description": (
                "Consulta el clima actual y el pronóstico de 3 días para una ubicación geográfica. "
                "Úsala cuando el usuario mencione una región, municipio o coordenadas, "
                "o cuando la pregunta sobre ganado o praderas pueda verse afectada por condiciones climáticas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitud": {
                        "type": "number",
                        "description": "Latitud decimal de la ubicación. Ej: 4.71 (Bogotá), 6.22 (Medellín), 3.43 (Cali).",
                    },
                    "longitud": {
                        "type": "number",
                        "description": "Longitud decimal de la ubicación. Ej: -74.07 (Bogotá), -75.57 (Medellín).",
                    },
                },
                "required": ["latitud", "longitud"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_carga_animal",
            "description": (
                "Calcula la capacidad de carga animal (UA/ha) de un potrero y determina "
                "si la carga actual es subutilizada, adecuada, alta o en sobrecarga crítica. "
                "Úsala cuando el usuario pregunte cuántas vacas caben en un potrero, "
                "o quiera saber si está manejando bien su pradera."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "area_ha": {
                        "type": "number",
                        "description": "Área del potrero en hectáreas.",
                    },
                    "peso_promedio_kg": {
                        "type": "number",
                        "description": "Peso promedio de los animales en kg. Bovino adulto colombiano: 350-500 kg.",
                    },
                    "disponibilidad_forraje_kg_ha": {
                        "type": "number",
                        "description": (
                            "Materia seca de forraje disponible en kg/ha. "
                            "Kikuyo: 2000-3000 | Brachiaria decumbens: 1500-2500 | "
                            "Estrella: 2000-3000 | Pasto natural: 800-1500."
                        ),
                    },
                    "tasa_consumo_pct": {
                        "type": "number",
                        "description": "Porcentaje del peso vivo consumido/día en materia seca. Default: 3.0 %.",
                    },
                },
                "required": ["area_ha", "peso_promedio_kg", "disponibilidad_forraje_kg_ha"],
            },
        },
    },
]

# Mapa nombre → función para despacho dinámico en el agente
MAPA_HERRAMIENTAS = {
    "obtener_clima":        obtener_clima,
    "calcular_carga_animal": calcular_carga_animal,
}
