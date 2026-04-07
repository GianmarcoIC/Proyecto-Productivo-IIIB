import time
import requests
from config import (
    PERPLEXITY_API_KEY, PERPLEXITY_API_URL,
    REGIONES_PERU, ALLOWED_EXTENSIONS
)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def determinar_region_peru(frutas_detectadas, departamento=None):
    conteo_regiones = {'Costa': 0, 'Sierra': 0, 'Selva': 0}

    if departamento:
        for region, info in REGIONES_PERU.items():
            if departamento in info['departamentos']:
                return region

    for fruta in frutas_detectadas:
        fruta_lower = fruta.lower()
        for region, info in REGIONES_PERU.items():
            for fruta_reg in info['frutas']:
                if fruta_reg in fruta_lower or fruta_lower in fruta_reg:
                    conteo_regiones[region] += 1

    if sum(conteo_regiones.values()) == 0:
        return "Costa"

    return max(conteo_regiones.items(), key=lambda x: x[1])[0]


def consultar_perplexity_mejorado(prompt):
    try:
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        data = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Eres un experto agrónomo peruano especializado en fruticultura. "
                        "Proporciona información precisa, práctica y específica para agricultores peruanos. "
                        "Sé natural, evita lenguaje robótico. Usa ejemplos locales y términos peruanos. "
                        "Responde en español peruano coloquial pero profesional."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0.8,
            "top_p": 0.9,
            "stream": False
        }

        response = requests.post(PERPLEXITY_API_URL, headers=headers, json=data, timeout=45)

        if response.status_code != 200:
            print(f"Error API Perplexity: {response.status_code} - {response.text}")
            return None

        return response.json()['choices'][0]['message']['content']

    except requests.exceptions.Timeout:
        print("Timeout en consulta Perplexity")
        return None
    except Exception as e:
        print(f"Error consultando Perplexity: {str(e)}")
        return None


def generar_analisis_completo(frutas_detectadas, confianzas, region, departamento=""):
    if not frutas_detectadas:
        return generar_respuesta_por_defecto()

    frutas_str = ", ".join(frutas_detectadas)
    confianza_promedio = sum(confianzas) / len(confianzas) if confianzas else 0

    if confianza_promedio >= 0.9:
        estado, porcentaje = "Óptimamente maduro", "90-100%"
    elif confianza_promedio >= 0.7:
        estado, porcentaje = "Maduro", "70-90%"
    elif confianza_promedio >= 0.5:
        estado, porcentaje = "Semi-maduro", "50-70%"
    elif confianza_promedio >= 0.3:
        estado, porcentaje = "Verde/Inmaduro", "30-50%"
    else:
        estado, porcentaje = "Muy verde", "0-30%"

    prompt = f"""Analiza estas frutas peruanas detectadas: {frutas_str}

Contexto:
- Región: {region}, Perú
- Departamento: {departamento if departamento else 'No especificado'}
- Estado de maduración: {estado} ({porcentaje})
- Confianza promedio: {confianza_promedio:.1%}

Necesito que me des información NATURAL como un experto agrónomo peruano:

1. DESCRIPCIÓN (sé específico y usa términos locales):
Describe estas frutas en el contexto peruano, menciona variedades locales si las conoces.

2. RECOMENDACIONES DE MADURACIÓN (prácticas y realistas):
Si está verde, ¿cómo acelerar la maduración de forma natural?
Si está maduro, ¿cómo mantenerlo en buen estado?

3. CLIMA Y SUELO IDEAL:
¿Qué condiciones específicas necesita en {region}?
¿Epoca de siembra y cosecha en Perú?

4. CONSEJOS DE CULTIVO:
3-4 tips prácticos para pequeños agricultores.

5. TIEMPO DE MADURACIÓN:
¿Cuánto falta aproximadamente? ¿Días o semanas?

6. ALMACENAMIENTO:
¿Cómo guardar para que dure más? ¿Temperatura ideal?

7. MERCADO LOCAL:
¿Dónde se comercializa mejor en Perú? ¿Precio aproximado?

IMPORTANTE: No uses frases genéricas. Sé específico para Perú. Usa ejemplos reales."""

    respuesta = consultar_perplexity_mejorado(prompt)
    if respuesta:
        return parsear_respuesta_ia(respuesta, estado, porcentaje, region)

    time.sleep(1)
    return generar_respuesta_por_defecto(region, frutas_str)


def parsear_respuesta_ia(respuesta_texto, estado, porcentaje, region):
    secciones = {
        'descripcion': '', 'recomendaciones': '', 'clima': '',
        'consejos_cultivo': '', 'tiempo_maduracion': '',
        'almacenamiento': '', 'mercado_local': ''
    }

    seccion_actual = None
    for linea in respuesta_texto.split('\n'):
        linea_lower = linea.lower().strip()

        if any(k in linea_lower for k in ['descripción', 'descripcion', 'contexto']):
            seccion_actual = 'descripcion'
        elif any(k in linea_lower for k in ['recomendación', 'recomendacion', 'maduración', 'maduracion']):
            seccion_actual = 'recomendaciones'
        elif any(k in linea_lower for k in ['clima', 'suelo', 'condiciones']):
            seccion_actual = 'clima'
        elif any(k in linea_lower for k in ['cultivo', 'consejos', 'tips', 'práctico']):
            seccion_actual = 'consejos_cultivo'
        elif any(k in linea_lower for k in ['tiempo', 'días', 'semanas', 'fecha']):
            seccion_actual = 'tiempo_maduracion'
        elif any(k in linea_lower for k in ['almacenamiento', 'guardar', 'temperatura']):
            seccion_actual = 'almacenamiento'
        elif any(k in linea_lower for k in ['mercado', 'comercial', 'precio', 'venta']):
            seccion_actual = 'mercado_local'

        if seccion_actual and linea.strip() and not linea_lower.startswith(
                ('1.', '2.', '3.', '4.', '5.', '6.', '7.')):
            if secciones[seccion_actual]:
                secciones[seccion_actual] += " "
            secciones[seccion_actual] += linea.strip()

    if not any(secciones.values()):
        secciones['descripcion'] = respuesta_texto[:500] + "..."
        secciones['recomendaciones'] = (
            "Para recomendaciones específicas, consulta con el SENASA o un ingeniero agrónomo local."
        )

    for key in secciones:
        if not secciones[key]:
            secciones[key] = obtener_contenido_por_defecto(key, region)

    return {
        'descripcion':        secciones['descripcion'],
        'recomendaciones':    secciones['recomendaciones'],
        'clima':              secciones['clima'],
        'consejos_cultivo':   secciones['consejos_cultivo'],
        'tiempo_maduracion':  secciones['tiempo_maduracion'],
        'almacenamiento':     secciones['almacenamiento'],
        'mercado_local':      secciones['mercado_local'],
        'porcentaje_maduracion': porcentaje,
        'estado_maduracion':  estado
    }


def obtener_contenido_por_defecto(seccion, region):
    defaults = {
        'descripcion':       f'Fruta típica de la región {region} de Perú.',
        'recomendaciones':   'Para maduración uniforme, mantener a temperatura ambiente.',
        'clima':             f'Clima adecuado para la región {region}: {REGIONES_PERU.get(region, {}).get("clima", "Variado")}',
        'consejos_cultivo':  'Realizar podas sanitarias, control natural de plagas y fertilización orgánica.',
        'tiempo_maduracion': 'Entre 2-4 semanas dependiendo de las condiciones climáticas.',
        'almacenamiento':    'Almacenar en lugar fresco y ventilado. No refrigerar si está verde.',
        'mercado_local':     'Se comercializa en mercados mayoristas como Villa María o Mercado Central.'
    }
    return defaults.get(seccion, 'Información en proceso de actualización.')


def generar_respuesta_por_defecto(region="Perú", frutas=""):
    return {
        'descripcion':        f'Frutas peruanas detectadas: {frutas}.',
        'recomendaciones':    f'Para {frutas if frutas else "estas frutas"}: madurar a temperatura ambiente, no apilar, revisar diariamente.',
        'clima':              f'En la región {region}, clima cálido de día y fresco de noche. Evitar heladas.',
        'consejos_cultivo':   '1) Usar abono orgánico  2) Riego por goteo  3) Control manual de plagas  4) Cosechar en horas frescas.',
        'tiempo_maduracion':  'Aproximadamente 7-15 días dependiendo de la temperatura y humedad.',
        'almacenamiento':     'Guardar en cajas de madera ventiladas. Temperatura ideal: 18-22 °C.',
        'mercado_local':      'Se vende bien en ferias agroecológicas y mercados locales.',
        'porcentaje_maduracion': '50-70%',
        'estado_maduracion':  'Semi-maduro'
    }
