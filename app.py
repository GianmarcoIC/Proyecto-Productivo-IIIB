from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import os
import base64
import sqlite3
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import requests
import time
from urllib.parse import quote

app = Flask(__name__, template_folder='templates', static_folder='static')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
THUMBNAIL_FOLDER = os.path.join(BASE_DIR, 'thumbnails')
DB_PATH = os.path.join(BASE_DIR, 'fruit_classifier2.db')

for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, THUMBNAIL_FOLDER]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

PERPLEXITY_API_KEY = "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"
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

REGIONES_PERU = {
    'Costa': {
        'frutas': ['limón', 'mandarina', 'naranja', 'palta', 'plátano', 'mango', 'uva', 'maracuyá', 'granadilla', 'fresa', 'sandía', 'melón'],
        'clima': 'Cálido y seco',
        'departamentos': ['Lima', 'La Libertad', 'Lambayeque', 'Piura', 'Ica', 'Arequipa', 'Moquegua', 'Tacna']
    },
    'Sierra': {
        'frutas': ['manzana', 'pera', 'durazno', 'tuna', 'granadilla', 'aguaymanto', 'chirimoya', 'membrillo', 'capulí', 'lúcuma', 'pacae'],
        'clima': 'Templado y frío',
        'departamentos': ['Cusco', 'Puno', 'Junín', 'Huánuco', 'Ancash', 'Cajamarca', 'Ayacucho', 'Apurímac']
    },
    'Selva': {
        'frutas': ['piña', 'papaya', 'maracuyá', 'coco', 'guayaba', 'camu camu', 'aguaje', 'plátano', 'piñón', 'cacao', 'ají', 'castaña'],
        'clima': 'Cálido y húmedo',
        'departamentos': ['Loreto', 'Ucayali', 'San Martín', 'Madre de Dios', 'Amazonas']
    }
}

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS biblioteca (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        detecciones TEXT,
        url TEXT,
        thumbnail_url TEXT,
        has_detection BOOLEAN,
        original_filename TEXT,
        source TEXT,
        confidence_average REAL,
        detection_count INTEGER,
        region_peru TEXT,
        departamento TEXT,
        descripcion_ia TEXT,
        recomendaciones TEXT,
        porcentaje_maduracion TEXT,
        clima_recomendado TEXT,
        consejos_cultivo TEXT,
        tiempo_maduracion TEXT,
        almacenamiento TEXT,
        mercado_local TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

init_db()

MODEL_PATH = "modelo/best_yolov8s_fruits_v1.pt"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "yolov8n.pt"
    
model = YOLO(MODEL_PATH)

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def save_to_db(entry):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    detecciones_json = json.dumps(entry['detecciones'])
    
    cursor.execute('''
    INSERT INTO biblioteca (
        timestamp, detecciones, url, thumbnail_url, has_detection, 
        original_filename, source, confidence_average, detection_count,
        region_peru, departamento, descripcion_ia, recomendaciones, 
        porcentaje_maduracion, clima_recomendado, consejos_cultivo,
        tiempo_maduracion, almacenamiento, mercado_local
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        entry['timestamp'],
        detecciones_json,
        entry['url'],
        entry['thumbnail_url'],
        entry['has_detection'],
        entry['original_filename'],
        entry['source'],
        entry['confidence_average'],
        entry['detection_count'],
        entry.get('region_peru', ''),
        entry.get('departamento', ''),
        entry.get('descripcion_ia', ''),
        entry.get('recomendaciones', ''),
        entry.get('porcentaje_maduracion', ''),
        entry.get('clima_recomendado', ''),
        entry.get('consejos_cultivo', ''),
        entry.get('tiempo_maduracion', ''),
        entry.get('almacenamiento', ''),
        entry.get('mercado_local', '')
    ))
    
    entry_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return entry_id

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
    
    region_dominante = max(conteo_regiones.items(), key=lambda x: x[1])
    return region_dominante[0]

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
                    "content": """Eres un experto agrónomo peruano especializado en fruticultura. 
                    Proporciona información precisa, práctica y específica para agricultores peruanos.
                    Sé natural, evita lenguaje robótico. Usa ejemplos locales y términos peruanos.
                    Responde en español peruano coloquial pero profesional."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.8,
            "top_p": 0.9,
            "stream": False
        }
        
        response = requests.post(PERPLEXITY_API_URL, headers=headers, json=data, timeout=45)
        
        if response.status_code != 200:
            return None
        
        result = response.json()
        return result['choices'][0]['message']['content']
        
    except:
        return None

def generar_analisis_completo(frutas_detectadas, confianzas, region, departamento=""):
    if not frutas_detectadas:
        return generar_respuesta_por_defecto()
    
    frutas_str = ", ".join(frutas_detectadas)
    confianza_promedio = sum(confianzas) / len(confianzas) if confianzas else 0
    
    if confianza_promedio >= 0.9:
        estado = "Óptimamente maduro"
        porcentaje = "90-100%"
    elif confianza_promedio >= 0.7:
        estado = "Maduro"
        porcentaje = "70-90%"
    elif confianza_promedio >= 0.5:
        estado = "Semi-maduro"
        porcentaje = "50-70%"
    elif confianza_promedio >= 0.3:
        estado = "Verde/Inmaduro"
        porcentaje = "30-50%"
    else:
        estado = "Muy verde"
        porcentaje = "0-30%"
    
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
    
    return generar_respuesta_por_defecto(region, frutas_str)

def parsear_respuesta_ia(respuesta_texto, estado, porcentaje, region):
    secciones = {
        'descripcion': '',
        'recomendaciones': '',
        'clima': '',
        'consejos_cultivo': '',
        'tiempo_maduracion': '',
        'almacenamiento': '',
        'mercado_local': ''
    }
    
    lineas = respuesta_texto.split('\n')
    seccion_actual = None
    
    for linea in lineas:
        linea_lower = linea.lower().strip()
        
        if any(keyword in linea_lower for keyword in ['descripción', 'descripcion', 'contexto']):
            seccion_actual = 'descripcion'
        elif any(keyword in linea_lower for keyword in ['recomendación', 'recomendacion', 'maduración', 'maduracion']):
            seccion_actual = 'recomendaciones'
        elif any(keyword in linea_lower for keyword in ['clima', 'suelo', 'condiciones']):
            seccion_actual = 'clima'
        elif any(keyword in linea_lower for keyword in ['cultivo', 'consejos', 'tips', 'práctico']):
            seccion_actual = 'consejos_cultivo'
        elif any(keyword in linea_lower for keyword in ['tiempo', 'días', 'semanas', 'fecha']):
            seccion_actual = 'tiempo_maduracion'
        elif any(keyword in linea_lower for keyword in ['almacenamiento', 'guardar', 'temperatura']):
            seccion_actual = 'almacenamiento'
        elif any(keyword in linea_lower for keyword in ['mercado', 'comercial', 'precio', 'venta']):
            seccion_actual = 'mercado_local'
        
        if seccion_actual and linea.strip() and not linea_lower.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.')):
            if len(secciones[seccion_actual]) > 0:
                secciones[seccion_actual] += " "
            secciones[seccion_actual] += linea.strip()
    
    for key in secciones:
        if not secciones[key]:
            secciones[key] = obtener_contenido_por_defecto(key, region)
    
    return {
        'descripcion': secciones['descripcion'],
        'recomendaciones': secciones['recomendaciones'],
        'clima': secciones['clima'],
        'consejos_cultivo': secciones['consejos_cultivo'],
        'tiempo_maduracion': secciones['tiempo_maduracion'],
        'almacenamiento': secciones['almacenamiento'],
        'mercado_local': secciones['mercado_local'],
        'porcentaje_maduracion': porcentaje,
        'estado_maduracion': estado
    }

def obtener_contenido_por_defecto(seccion, region):
    defaults = {
        'descripcion': f'Fruta típica de la región {region} de Perú. Cultivada por pequeños y medianos agricultores.',
        'recomendaciones': 'Para maduración uniforme, mantener a temperatura ambiente. Evitar exposición directa al sol.',
        'clima': f'Clima adecuado para la región {region}: {REGIONES_PERU.get(region, {}).get("clima", "Variado")}',
        'consejos_cultivo': 'Realizar podas sanitarias, control natural de plagas y fertilización orgánica.',
        'tiempo_maduracion': 'Entre 2-4 semanas dependiendo de las condiciones climáticas.',
        'almacenamiento': 'Almacenar en lugar fresco y ventilado. No refrigerar si está verde.',
        'mercado_local': 'Se comercializa en mercados mayoristas como Villa María o Mercado Central.'
    }
    return defaults.get(seccion, 'Información en proceso de actualización.')

def generar_respuesta_por_defecto(region="Perú", frutas=""):
    return {
        'descripcion': f'Frutas peruanas detectadas: {frutas}. Estas son cultivadas tradicionalmente en diferentes regiones del país.',
        'recomendaciones': f'Para {frutas if frutas else "estas frutas"}, recomiendo: 1) Madurar a temperatura ambiente 2) No apilar mucho 3) Revisar diariamente 4) Separar las muy maduras.',
        'clima': f'En la región {region}, el clima ideal es cálido durante el día y fresco por la noche. Evitar heladas.',
        'consejos_cultivo': '1) Usar abono orgánico 2) Riego por goteo 3) Control manual de plagas 4) Cosechar en horas frescas.',
        'tiempo_maduracion': 'Aproximadamente 7-15 días dependiendo de la temperatura y humedad.',
        'almacenamiento': 'Guardar en cajas de madera ventiladas. Temperatura ideal: 18-22°C.',
        'mercado_local': 'Se vende bien en ferias agroecológicas y mercados locales.',
        'porcentaje_maduracion': '50-70%',
        'estado_maduracion': 'Semi-maduro'
    }

@app.route('/logo.png')
def serve_logo():
    return send_from_directory('.', 'logo.png')

@app.route('/static/sounds/<path:filename>')
def serve_sounds(filename):
    return send_from_directory('static/sounds', filename)

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/thumbnails/<path:filename>')
def serve_thumbnails(filename):
    return send_from_directory(THUMBNAIL_FOLDER, filename)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/departamentos/<region>")
def get_departamentos(region):
    departamentos = REGIONES_PERU.get(region, {}).get('departamentos', [])
    return jsonify(departamentos)

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if 'imagen' not in request.files:
            return jsonify({"error": "No se encontró la imagen"}), 400
        
        file = request.files['imagen']
        if file.filename == '':
            return jsonify({"error": "No se seleccionó ningún archivo"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": f"Tipo de archivo no permitido. Use: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
        
        departamento = request.form.get('departamento', '')
        
        img_bytes = file.read()
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Imagen inválida o corrupta"}), 400
        
        results = model(img, conf=0.4, verbose=False)
        annotated = results[0].plot()
        
        detecciones = []
        has_detection = False
        boxes = results[0].boxes
        frutas_detectadas = []
        confianzas = []
        
        if boxes is not None and len(boxes) > 0:
            has_detection = True
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = round(float(box.conf[0].item()), 2)
                clase = model.names[cls_id]
                detecciones.append({"clase": clase, "conf": conf})
                frutas_detectadas.append(clase)
                confianzas.append(conf)
        
        confidence_average = sum(d['conf'] for d in detecciones) / len(detecciones) if detecciones else 0
        detection_count = len(detecciones)
        
        region = determinar_region_peru(frutas_detectadas, departamento)
        
        if frutas_detectadas:
            analisis_ia = generar_analisis_completo(frutas_detectadas, confianzas, region, departamento)
        else:
            analisis_ia = generar_respuesta_por_defecto(region, "No detectadas")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        original_filename = secure_filename(file.filename)
        base_name = os.path.splitext(original_filename)[0]
        
        original_path = os.path.join(UPLOAD_FOLDER, f"{base_name}_{timestamp}_original.jpg")
        cv2.imwrite(original_path, img)
        
        processed_filename = f"{base_name}_{timestamp}_processed.jpg"
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
        cv2.imwrite(processed_path, annotated)
        
        thumb_img = cv2.resize(img, (200, 200))
        thumbnail_filename = f"{base_name}_{timestamp}_thumb.jpg"
        thumbnail_path = os.path.join(THUMBNAIL_FOLDER, thumbnail_filename)
        cv2.imwrite(thumbnail_path, thumb_img)
        
        url = f"/processed/{processed_filename}"
        thumbnail_url = f"/thumbnails/{thumbnail_filename}"
        
        entry = {
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "detecciones": detecciones,
            "url": url,
            "thumbnail_url": thumbnail_url,
            "has_detection": has_detection,
            "original_filename": original_filename,
            "source": "upload",
            "confidence_average": round(confidence_average, 2),
            "detection_count": detection_count,
            "region_peru": region,
            "departamento": departamento,
            "descripcion_ia": analisis_ia['descripcion'],
            "recomendaciones": analisis_ia['recomendaciones'],
            "porcentaje_maduracion": analisis_ia['porcentaje_maduracion'],
            "clima_recomendado": analisis_ia['clima'],
            "consejos_cultivo": analisis_ia['consejos_cultivo'],
            "tiempo_maduracion": analisis_ia['tiempo_maduracion'],
            "almacenamiento": analisis_ia['almacenamiento'],
            "mercado_local": analisis_ia['mercado_local']
        }
        
        entry_id = save_to_db(entry)
        entry["id"] = entry_id
        
        return jsonify(entry)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/captura", methods=["POST"])
def captura():
    try:
        data = request.json
        if 'imagen' not in data:
            return jsonify({"error": "No se encontró imagen en la captura"}), 400
        
        departamento = data.get('departamento', '')
        
        if 'base64,' in data['imagen']:
            img_data = base64.b64decode(data['imagen'].split(',')[1])
        else:
            img_data = base64.b64decode(data['imagen'])
            
        npimg = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Imagen inválida o corrupta"}), 400
        
        results = model(img, conf=0.4, verbose=False)
        annotated = results[0].plot()
        
        detecciones = []
        has_detection = False
        boxes = results[0].boxes
        frutas_detectadas = []
        confianzas = []
        
        if boxes is not None and len(boxes) > 0:
            has_detection = True
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = round(float(box.conf[0].item()), 2)
                clase = model.names[cls_id]
                detecciones.append({"clase": clase, "conf": conf})
                frutas_detectadas.append(clase)
                confianzas.append(conf)
        
        confidence_average = sum(d['conf'] for d in detecciones) / len(detecciones) if detecciones else 0
        detection_count = len(detecciones)
        
        region = determinar_region_peru(frutas_detectadas, departamento)
        
        if frutas_detectadas:
            analisis_ia = generar_analisis_completo(frutas_detectadas, confianzas, region, departamento)
        else:
            analisis_ia = generar_respuesta_por_defecto(region, "No detectadas")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        original_path = os.path.join(UPLOAD_FOLDER, f"capture_{timestamp}_original.jpg")
        cv2.imwrite(original_path, img)
        
        processed_filename = f"capture_{timestamp}_processed.jpg"
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
        cv2.imwrite(processed_path, annotated)
        
        thumb_img = cv2.resize(img, (200, 200))
        thumbnail_filename = f"capture_{timestamp}_thumb.jpg"
        thumbnail_path = os.path.join(THUMBNAIL_FOLDER, thumbnail_filename)
        cv2.imwrite(thumbnail_path, thumb_img)
        
        url = f"/processed/{processed_filename}"
        thumbnail_url = f"/thumbnails/{thumbnail_filename}"
        
        entry = {
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "detecciones": detecciones,
            "url": url,
            "thumbnail_url": thumbnail_url,
            "has_detection": has_detection,
            "original_filename": f"capture_{timestamp}.jpg",
            "source": "camera",
            "confidence_average": round(confidence_average, 2),
            "detection_count": detection_count,
            "region_peru": region,
            "departamento": departamento,
            "descripcion_ia": analisis_ia['descripcion'],
            "recomendaciones": analisis_ia['recomendaciones'],
            "porcentaje_maduracion": analisis_ia['porcentaje_maduracion'],
            "clima_recomendado": analisis_ia['clima'],
            "consejos_cultivo": analisis_ia['consejos_cultivo'],
            "tiempo_maduracion": analisis_ia['tiempo_maduracion'],
            "almacenamiento": analisis_ia['almacenamiento'],
            "mercado_local": analisis_ia['mercado_local']
        }
        
        entry_id = save_to_db(entry)
        entry["id"] = entry_id
        
        return jsonify(entry)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/biblioteca")
def get_biblioteca():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, timestamp, detecciones, url, thumbnail_url, has_detection, 
               original_filename, source, confidence_average, detection_count,
               region_peru, departamento, descripcion_ia, recomendaciones, 
               porcentaje_maduracion, clima_recomendado, consejos_cultivo,
               tiempo_maduracion, almacenamiento, mercado_local, created_at
        FROM biblioteca 
        ORDER BY created_at DESC
        ''')
        
        rows = cursor.fetchall()
        biblioteca = []
        
        for row in rows:
            item = dict(row)
            item['detecciones'] = json.loads(item['detecciones']) if item['detecciones'] else []
            biblioteca.append(item)
        
        conn.close()
        return jsonify(biblioteca)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/delete/<int:db_id>", methods=["DELETE"])
def delete_record(db_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT url, thumbnail_url FROM biblioteca WHERE id = ?', (db_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return jsonify({"error": "Registro no encontrado"}), 404
        
        url = row['url']
        thumbnail_url = row['thumbnail_url']
        
        try:
            if url and url.startswith('/processed/'):
                filename = url.split('/')[-1]
                file_path = os.path.join(PROCESSED_FOLDER, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            if thumbnail_url and thumbnail_url.startswith('/thumbnails/'):
                filename = thumbnail_url.split('/')[-1]
                file_path = os.path.join(THUMBNAIL_FOLDER, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
        except:
            pass
        
        cursor.execute('DELETE FROM biblioteca WHERE id = ?', (db_id,))
        conn.commit()
        conn.close()
        
        return jsonify({"message": "Registro e imágenes eliminados correctamente"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/estadisticas")
def get_estadisticas():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM biblioteca')
        rows = cursor.fetchall()
        conn.close()
        
        biblioteca = []
        for row in rows:
            item = dict(row)
            item['detecciones'] = json.loads(item['detecciones']) if item['detecciones'] else []
            biblioteca.append(item)
        
        total = len(biblioteca)
        detectadas = sum(1 for e in biblioteca if e["has_detection"])
        no_detectadas = total - detectadas
        
        conteo_clases = {}
        confidence_by_class = {}
        regiones_count = {}
        departamentos_count = {}
        
        for entry in biblioteca:
            region = entry.get("region_peru", "No especificada")
            regiones_count[region] = regiones_count.get(region, 0) + 1
            
            depto = entry.get("departamento", "No especificado")
            if depto:
                departamentos_count[depto] = departamentos_count.get(depto, 0) + 1
            
            for d in entry["detecciones"]:
                clase = d["clase"]
                conf = d["conf"]
                conteo_clases[clase] = conteo_clases.get(clase, 0) + 1
                if clase not in confidence_by_class:
                    confidence_by_class[clase] = []
                confidence_by_class[clase].append(conf)
        
        avg_confidence_by_class = {
            clase: round(sum(confs) / len(confs), 2) if confs else 0
            for clase, confs in confidence_by_class.items()
        }
        
        top_departamentos = sorted(departamentos_count.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return jsonify({
            "total": total,
            "detectadas": detectadas,
            "no_detectadas": no_detectadas,
            "clases": conteo_clases,
            "avg_confidence_by_class": avg_confidence_by_class,
            "regiones": regiones_count,
            "departamentos": dict(top_departamentos),
            "top_frutas": sorted(conteo_clases.items(), key=lambda x: x[1], reverse=True)[:10]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset_database():
    try:
        for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, THUMBNAIL_FOLDER]:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        init_db()
        
        return jsonify({"message": "Sistema reseteado correctamente"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/test-perplexity")
def test_perplexity():
    try:
        test_prompt = "Di hola en español y dime cuál es la capital de Perú en una sola línea."
        respuesta = consultar_perplexity_mejorado(test_prompt)
        
        if respuesta:
            return jsonify({
                "status": "success",
                "message": "Conexión exitosa con Perplexity API",
                "response": respuesta[:200]
            })
        else:
            return jsonify({
                "status": "error",
                "message": "No se pudo conectar con Perplexity API"
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clasificador de Frutas Peruanas - Sistema Inteligente</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --primary-red: #d32f2f;
            --dark-red: #b71c1c;
            --light-red: #ff6659;
            --gradient-red: linear-gradient(135deg, var(--dark-red) 0%, var(--primary-red) 50%, var(--light-red) 100%);
            --admin-red: linear-gradient(135deg, #8B0000 0%, #B22222 50%, #DC143C 100%);
            --sidebar-red: linear-gradient(180deg, #7a0000 0%, #8B0000 30%, #A50000 100%);
        }
        
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;
            background:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);
            min-height:100vh; color:#333; display:flex;
        }
        
        .sidebar {
            width:250px; background:var(--sidebar-red); color:white;
            min-height:100vh; position:fixed; left:0; top:0;
            box-shadow:4px 0 15px rgba(0,0,0,0.2); z-index:1000;
            display:flex; flex-direction:column;
        }
        
        .sidebar-header { padding:1.5rem; text-align:center; border-bottom:2px solid rgba(255,255,255,0.1); }
        .sidebar-logo { display:flex; align-items:center; justify-content:center; overflow:hidden; }
        .sidebar-logo img { width:60%; height:70px; object-fit:contain; }
        .sidebar-header h2 { font-size:1.2rem; margin-bottom:5px; color:#FFD700; }
        .sidebar-header p { font-size:0.8rem; opacity:0.8; }
        
        .sidebar-nav { flex:1; padding:1.5rem 0; }
        .nav-item {
            display:flex; align-items:center; padding:1rem 1.5rem;
            color:white; text-decoration:none; transition:all 0.3s ease;
            border-left:4px solid transparent; margin-bottom:5px;
        }
        .nav-item:hover { background:rgba(255,255,255,0.1); border-left-color:#FFD700; }
        .nav-item.active { background:rgba(255,255,255,0.15); border-left-color:#FFD700; font-weight:bold; }
        .nav-item i { width:25px; margin-right:10px; font-size:1.2rem; }
        
        .sidebar-footer { padding:1rem; border-top:2px solid rgba(255,255,255,0.1); text-align:center; font-size:0.8rem; opacity:0.7; }
        
        .main-content { flex:1; margin-left:250px; padding:20px; min-height:100vh; }
        
        .header {
            background:var(--gradient-red); color:white; padding:1.5rem 2rem;
            border-radius:15px; margin-bottom:2rem; box-shadow:0 4px 15px rgba(0,0,0,0.1);
        }
        .header h1 { font-size:2rem; margin-bottom:0.5rem; display:flex; align-items:center; gap:15px; }
        .header p { opacity:0.9; font-size:1rem; }
        
        .content-panel { background:white; border-radius:15px; padding:2rem; margin-bottom:2rem; box-shadow:0 4px 20px rgba(0,0,0,0.08); display:none; }
        .content-panel.active { display:block; animation:fadeIn 0.5s ease; }
        @keyframes fadeIn { from {opacity:0; transform:translateY(10px);} to {opacity:1; transform:translateY(0);} }
        
        .location-form {
            background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white;
            padding:1.5rem; border-radius:10px; margin-bottom:2rem; box-shadow:0 4px 15px rgba(102,126,234,0.3);
        }
        .form-group { margin-bottom:1rem; }
        .form-group label { display:block; margin-bottom:0.5rem; font-weight:600; }
        .form-control { width:100%; padding:10px 15px; border:none; border-radius:8px; font-size:1rem; background:rgba(255,255,255,0.9); }
        .form-control:focus { outline:none; box-shadow:0 0 0 3px rgba(255,255,255,0.3); }
        .form-row { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
        
        .upload-area {
            border:3px dashed #ddd; border-radius:10px; padding:3rem 2rem;
            text-align:center; cursor:pointer; transition:all 0.3s ease; background:#f9f9f9; margin:1rem 0;
        }
        .upload-area:hover { border-color:var(--primary-red); background:#fff5f5; }
        .upload-area i { font-size:4rem; color:var(--primary-red); margin-bottom:1rem; }
        
        .camera-preview { width:100%; max-width:640px; height:480px; background:#333; border-radius:10px; overflow:hidden; margin:1rem auto; position:relative; }
        #videoElement { width:100%; height:100%; object-fit:cover; }
        
        .btn {
            background:var(--gradient-red); color:white; border:none; padding:12px 25px;
            border-radius:25px; font-size:1rem; cursor:pointer; display:inline-flex; align-items:center; gap:10px;
            transition:all 0.3s ease; font-weight:600; box-shadow:0 4px 15px rgba(211,47,47,0.3);
        }
        .btn:hover { transform:translateY(-2px); box-shadow:0 6px 20px rgba(211,47,47,0.4); }
        .btn-secondary { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); }
        .btn-success { background:linear-gradient(135deg,#4CAF50 0%,#2E7D32 100%); }
        .btn-danger { background:linear-gradient(135deg,#f44336 0%,#d32f2f 100%); }
        
        .results-container { display:grid; grid-template-columns:1fr 1fr; gap:2rem; margin-top:2rem; }
        .image-preview { background:white; border-radius:10px; padding:1rem; box-shadow:0 4px 15px rgba(0,0,0,0.1); border:1px solid #eee; }
        .image-preview img { width:100%; border-radius:8px; max-height:400px; object-fit:contain; }
        .detections-list { background:white; border-radius:10px; padding:1.5rem; box-shadow:0 4px 15px rgba(0,0,0,0.1); border:1px solid #eee; }
        .detection-item {
            background:#f8f9fa; border-left:4px solid var(--primary-red); padding:1rem;
            margin-bottom:0.5rem; border-radius:5px; display:flex; justify-content:space-between; align-items:center;
        }
        
        .ia-analysis {
            background:linear-gradient(135deg,#1e3c72 0%,#2a5298 100%); color:white;
            border-radius:15px; padding:2rem; margin-top:2rem; box-shadow:0 8px 32px rgba(30,60,114,0.3);
        }
        .analysis-section {
            background:rgba(255,255,255,0.1); padding:1.5rem; border-radius:10px;
            margin-bottom:1.5rem; border-left:4px solid #FFD700;
        }
        .analysis-section h4 { color:#FFD700; margin-bottom:0.5rem; display:flex; align-items:center; gap:10px; }
        .region-badge {
            display:inline-block; background:#FFD700; color:#8B0000; padding:8px 20px;
            border-radius:20px; font-weight:bold; margin-bottom:1rem; box-shadow:0 2px 5px rgba(0,0,0,0.2);
        }
        .maturity-bar { width:100%; height:20px; background:rgba(255,255,255,0.2); border-radius:10px; overflow:hidden; margin:1rem 0; }
        .maturity-fill { height:100%; background:linear-gradient(90deg,#4CAF50,#8BC34A); border-radius:10px; transition:width 1s ease; }
        
        .admin-panel {
            background:var(--admin-red); border-radius:15px; padding:2rem; color:white;
            box-shadow:0 8px 32px rgba(139,0,0,0.3); margin-bottom:2rem; position:relative; overflow:hidden;
        }
        .admin-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem; }
        .stats-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:1rem; margin-bottom:2rem; }
        .stat-card {
            background:rgba(255,255,255,0.1); backdrop-filter:blur(10px); border-radius:10px;
            padding:1.5rem; text-align:center; border:1px solid rgba(255,255,255,0.2); transition:transform 0.3s ease;
        }
        .stat-card:hover { transform:translateY(-5px); background:rgba(255,255,255,0.15); }
        .stat-card h3 { font-size:2.5rem; margin-bottom:5px; color:#FFD700; }
        
        .library-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:1.5rem; margin-top:2rem; }
        .library-card {
            background:white; border-radius:10px; overflow:hidden; box-shadow:0 4px 15px rgba(0,0,0,0.1);
            transition:transform 0.3s ease; border:1px solid #eee;
        }
        .library-card:hover { transform:translateY(-5px); box-shadow:0 8px 25px rgba(0,0,0,0.15); }
        .card-image { height:200px; overflow:hidden; }
        .card-image img { width:100%; height:100%; object-fit:cover; transition:transform 0.3s ease; }
        .library-card:hover .card-image img { transform:scale(1.05); }
        .card-content { padding:1rem; }
        .card-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem; }
        .delete-btn {
            background:#dc3545; color:white; border:none; width:30px; height:30px;
            border-radius:50%; cursor:pointer; display:flex; align-items:center; justify-content:center; transition:all 0.3s ease;
        }
        .delete-btn:hover { background:#c82333; transform:scale(1.1); }
        
        .notification {
            position:fixed; top:20px; right:20px; padding:1rem 1.5rem; border-radius:10px;
            color:white; box-shadow:0 4px 12px rgba(0,0,0,0.15); z-index:1000;
            display:flex; align-items:center; gap:10px; transform:translateX(120%); transition:transform 0.3s ease;
        }
        .notification.show { transform:translateX(0); }
        .notification.success { background:linear-gradient(135deg,#4CAF50,#45a049); }
        .notification.error { background:linear-gradient(135deg,#f44336,#d32f2f); }
        .notification.info { background:linear-gradient(135deg,#2196F3,#1976D2); }
        
        .loading {
            display:none; position:fixed; top:0; left:0; right:0; bottom:0;
            background:rgba(0,0,0,0.7); z-index:2000; justify-content:center; align-items:center;
            flex-direction:column; color:white;
        }
        .loading.active { display:flex; }
        .spinner {
            width:60px; height:60px; border:5px solid #f3f3f3; border-top:5px solid var(--primary-red);
            border-radius:50%; animation:spin 1s linear infinite; margin-bottom:1rem;
        }
        @keyframes spin { 0% {transform:rotate(0deg);} 100% {transform:rotate(360deg);} }
        
        @media (max-width:1024px) {
            .sidebar { width:200px; }
            .main-content { margin-left:200px; }
            .results-container { grid-template-columns:1fr; }
        }
        @media (max-width:768px) {
            .sidebar { transform:translateX(-100%); transition:transform 0.3s ease; }
            .sidebar.active { transform:translateX(0); }
            .main-content { margin-left:0; }
            .header { padding:1rem; }
            .content-panel { padding:1rem; }
            .form-row { grid-template-columns:1fr; }
            .menu-toggle {
                display:block; position:fixed; top:15px; left:15px; z-index:1001;
                background:var(--primary-red); color:white; border:none; width:40px; height:40px;
                border-radius:50%; display:flex; align-items:center; justify-content:center; box-shadow:0 2px 10px rgba(0,0,0,0.2);
            }
        }
        
        .text-center { text-align:center; }
        .mt-1 { margin-top:0.5rem; }
        .mt-2 { margin-top:1rem; }
        .mt-3 { margin-top:1.5rem; }
        .mb-1 { margin-bottom:0.5rem; }
        .mb-2 { margin-bottom:1rem; }
        .mb-3 { margin-bottom:1.5rem; }
        .d-flex { display:flex; }
        .flex-column { flex-direction:column; }
        .align-items-center { align-items:center; }
        .justify-content-between { justify-content:space-between; }
        .gap-1 { gap:0.5rem; }
        .gap-2 { gap:1rem; }
        .w-100 { width:100%; }
        
        .chart-container { background:rgba(255,255,255,0.1); border-radius:10px; padding:1rem; margin:1rem 0; }
    </style>
</head>
<body>
    <div class="sidebar" id="sidebar">
        <div class="sidebar-header">
            <div class="sidebar-logo"><img src="logo.png" alt="Logo Perú Frutas"></div>
            <h2>Perú Frutas IA</h2>
            <p>Sistema Inteligente</p>
        </div>
        
        <div class="sidebar-nav">
            <a href="#" class="nav-item active" onclick="showPanel('upload')"><i class="fas fa-upload"></i> Subir Imagen</a>
            <a href="#" class="nav-item" onclick="showPanel('camera')"><i class="fas fa-camera"></i> Cámara en Vivo</a>
            <a href="#" class="nav-item" onclick="showPanel('library')"><i class="fas fa-book"></i> Biblioteca</a>
            <a href="#" class="nav-item" onclick="showPanel('admin')"><i class="fas fa-chart-bar"></i> Administración</a>
            <a href="#" class="nav-item" onclick="showPanel('help')"><i class="fas fa-question-circle"></i> Ayuda</a>
        </div>
        
        <div class="sidebar-footer">
            <p>© 2024 Perú Frutas IA</p>
            <p>v2.0 - Inteligencia Artificial</p>
        </div>
    </div>
    
    <button class="menu-toggle" id="menuToggle" onclick="toggleSidebar()"><i class="fas fa-bars"></i></button>
    
    <div class="main-content">
        <div class="header">
            <h1><i class="fas fa-apple-alt"></i> Clasificador de Frutas Peruanas</h1>
            <p>Inteligencia Artificial para análisis regional de frutas del Perú</p>
        </div>
        
        <div class="location-form" id="locationForm">
            <h3><i class="fas fa-map-marker-alt"></i> Información de Ubicación</h3>
            <p>Proporciona la ubicación para un análisis más preciso:</p>
            
            <div class="form-row">
                <div class="form-group">
                    <label for="regionSelect"><i class="fas fa-mountain"></i> Región:</label>
                    <select id="regionSelect" class="form-control" onchange="loadDepartments()">
                        <option value="">Seleccionar región</option>
                        <option value="Costa">Costa</option>
                        <option value="Sierra">Sierra</option>
                        <option value="Selva">Selva</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="departamentoSelect"><i class="fas fa-city"></i> Departamento:</label>
                    <select id="departamentoSelect" class="form-control">
                        <option value="">Seleccionar departamento</option>
                    </select>
                </div>
            </div>
            
            <div class="form-group">
                <label for="localidadInput"><i class="fas fa-map-pin"></i> Localidad (opcional):</label>
                <input type="text" id="localidadInput" class="form-control" placeholder="Ej: Valle de Chancay, Villa María">
            </div>
        </div>
        
        <div id="uploadPanel" class="content-panel active">
            <h2><i class="fas fa-cloud-upload-alt"></i> Subir Imagen de Frutas</h2>
            <p>Selecciona una imagen para análisis con inteligencia artificial</p>
            
            <div class="upload-area" id="dropArea" onclick="document.getElementById('fileInput').click()">
                <i class="fas fa-file-upload"></i>
                <h3>Arrastra y suelta tu imagen aquí</h3>
                <p>o haz clic para seleccionar archivo</p>
                <p class="file-types">PNG, JPG, JPEG, GIF, BMP (Máx. 10MB)</p>
            </div>
            
            <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleFileSelect(event)">
            
            <div class="text-center mt-2">
                <button class="btn" onclick="processImage()" id="processBtn">
                    <i class="fas fa-magic"></i> Analizar con IA
                </button>
                <button class="btn btn-secondary" onclick="clearUpload()">
                    <i class="fas fa-times"></i> Limpiar
                </button>
            </div>
            
            <div id="uploadResults" style="display:none"></div>
        </div>
        
        <div id="cameraPanel" class="content-panel">
            <h2><i class="fas fa-video"></i> Cámara en Vivo</h2>
            <p>Usa tu cámara para capturar imágenes en tiempo real</p>
            
            <div class="camera-preview">
                <video id="videoElement" autoplay playsinline></video>
            </div>
            
            <div class="text-center mt-2">
                <button class="btn" onclick="toggleCamera()" id="cameraToggleBtn">
                    <i class="fas fa-power-off"></i> Encender Cámara
                </button>
                <button class="btn btn-success" onclick="captureImage()" id="captureBtn" disabled>
                    <i class="fas fa-camera"></i> Capturar y Analizar
                </button>
            </div>
            
            <div id="cameraResults" class="mt-3"></div>
        </div>
        
        <div id="libraryPanel" class="content-panel">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h2><i class="fas fa-database"></i> Biblioteca de Imágenes</h2>
                <div class="d-flex gap-1">
                    <button class="btn" onclick="refreshLibrary()">
                        <i class="fas fa-sync-alt"></i> Actualizar
                    </button>
                    <button class="btn btn-danger" onclick="clearLibrary()">
                        <i class="fas fa-trash"></i> Limpiar Todo
                    </button>
                </div>
            </div>
            
            <div id="libraryGrid" class="library-grid"></div>
        </div>
        
        <div id="adminPanel" class="content-panel">
            <div class="admin-panel">
                <div class="admin-header">
                    <h2><i class="fas fa-chart-line"></i> Panel de Administración</h2>
                    <button class="btn" onclick="resetSystem()">
                        <i class="fas fa-redo"></i> Resetear Sistema
                    </button>
                </div>
                
                <div class="stats-grid" id="statsGrid"></div>
                
                <div class="d-flex gap-2 mb-3">
                    <div class="chart-container" style="flex:1;">
                        <h4><i class="fas fa-map-marked-alt"></i> Distribución por Región</h4>
                        <canvas id="regionChart" height="250"></canvas>
                    </div>
                    <div class="chart-container" style="flex:1;">
                        <h4><i class="fas fa-apple-alt"></i> Frutas Más Comunes</h4>
                        <canvas id="fruitChart" height="250"></canvas>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h4><i class="fas fa-chart-bar"></i> Actividad por Día</h4>
                    <canvas id="activityChart" height="150"></canvas>
                </div>
            </div>
        </div>
        
        <div id="helpPanel" class="content-panel">
            <h2><i class="fas fa-question-circle"></i> Centro de Ayuda</h2>
            
            <div class="analysis-section">
                <h4><i class="fas fa-info-circle"></i> Cómo usar el sistema</h4>
                <p>1. <strong>Proporciona la ubicación</strong> para análisis regional preciso</p>
                <p>2. <strong>Sube una imagen</strong> de frutas o usa la cámara</p>
                <p>3. <strong>Recibe análisis completo</strong> con IA específica para Perú</p>
                <p>4. <strong>Consulta la biblioteca</strong> para ver historial</p>
                <p>5. <strong>Revisa estadísticas</strong> en el panel de administración</p>
            </div>
            
            <div class="analysis-section">
                <h4><i class="fas fa-lightbulb"></i> Consejos para mejores resultados</h4>
                <p>• Toma fotos con buena iluminación natural</p>
                <p>• Enfoca bien las frutas en la imagen</p>
                <p>• Usa fondo simple para mejor detección</p>
                <p>• Proporciona la ubicación exacta para recomendaciones específicas</p>
                <p>• Para frutas verdes, menciona si deseas acelerar la maduración</p>
            </div>
            
            <div class="analysis-section">
                <h4><i class="fas fa-phone-alt"></i> Soporte Técnico</h4>
                <p><strong>Email:</strong> soporte@perufrutas-ia.pe</p>
                <p><strong>Teléfono:</strong> (01) 123-4567</p>
                <p><strong>Horario:</strong> Lunes a Viernes 8am - 6pm</p>
            </div>
        </div>
    </div>
    
    <div class="loading" id="loading">
        <div class="spinner"></div>
        <h3>Analizando con Inteligencia Artificial...</h3>
        <p>Determinando región, estado de maduración y generando recomendaciones específicas</p>
        <p id="loadingDetail" class="mt-1"></p>
    </div>
    
    <div class="notification" id="notification">
        <i class="fas fa-info-circle"></i>
        <span id="notificationText">Mensaje de notificación</span>
    </div>
    
    <script>
        let currentPanel = 'upload';
        let stream = null;
        let regionChart = null;
        let fruitChart = null;
        let activityChart = null;
        let selectedFile = null;
        let currentLocation = { region:'', departamento:'', localidad:'' };
        
        document.addEventListener('DOMContentLoaded', function() {
            loadLibrary();
            loadStats();
            setupDragAndDrop();
            setTimeout(() => {
                showNotification('¡Bienvenido al Sistema de Frutas Peruanas! Proporciona tu ubicación para análisis preciso.', 'info');
            }, 1000);
        });
        
        function setupDragAndDrop() {
            const dropArea = document.getElementById('dropArea');
            ['dragenter','dragover','dragleave','drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false);
            });
            function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }
            ['dragenter','dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, highlight, false);
            });
            ['dragleave','drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, unhighlight, false);
            });
            function highlight() { dropArea.style.borderColor='var(--primary-red)'; dropArea.style.backgroundColor='#fff5f5'; }
            function unhighlight() { dropArea.style.borderColor='#ddd'; dropArea.style.backgroundColor='#f9f9f9'; }
            dropArea.addEventListener('drop', handleDrop, false);
            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                if (files.length > 0) {
                    selectedFile = files[0];
                    if (selectedFile.type.match('image.*')) {
                        showNotification(`Imagen seleccionada: ${selectedFile.name}`,'info');
                        document.getElementById('processBtn').disabled = false;
                        const reader = new FileReader();
                        reader.onload = function(e) {
                            const preview = document.createElement('div');
                            preview.innerHTML = `
                                <div class="image-preview mt-2">
                                    <h4>Vista previa:</h4>
                                    <img src="${e.target.result}" alt="Vista previa" style="max-height:200px;">
                                    <p class="mt-1"><strong>Archivo:</strong> ${selectedFile.name}</p>
                                    <p><strong>Tamaño:</strong> ${(selectedFile.size/1024).toFixed(1)} KB</p>
                                </div>`;
                            document.getElementById('uploadPanel').appendChild(preview);
                        };
                        reader.readAsDataURL(selectedFile);
                    } else {
                        showNotification('Por favor, suelta solo imágenes','error');
                    }
                }
            }
        }
        
        function showPanel(panelName) {
            document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
            document.querySelector(`[onclick="showPanel('${panelName}')"]`).classList.add('active');
            document.querySelectorAll('.content-panel').forEach(panel => panel.classList.remove('active'));
            document.getElementById(`${panelName}Panel`).classList.add('active');
            currentPanel = panelName;
            if (panelName === 'library') loadLibrary();
            if (panelName === 'admin') loadStats();
            if (window.innerWidth <= 768) document.getElementById('sidebar').classList.remove('active');
        }
        
        function toggleSidebar() {
            document.getElementById('sidebar').classList.toggle('active');
        }
        
        function loadDepartments() {
            const region = document.getElementById('regionSelect').value;
            const departamentoSelect = document.getElementById('departamentoSelect');
            if (!region) {
                departamentoSelect.innerHTML = '<option value="">Seleccionar departamento</option>';
                return;
            }
            currentLocation.region = region;
            departamentoSelect.innerHTML = '<option value="">Cargando departamentos...</option>';
            departamentoSelect.disabled = true;
            fetch(`/departamentos/${encodeURIComponent(region)}`)
                .then(response => response.json())
                .then(departamentos => {
                    departamentoSelect.innerHTML = '<option value="">Seleccionar departamento</option>';
                    departamentos.forEach(depto => {
                        const option = document.createElement('option');
                        option.value = depto;
                        option.textContent = depto;
                        departamentoSelect.appendChild(option);
                    });
                    departamentoSelect.disabled = false;
                })
                .catch(() => {
                    departamentoSelect.innerHTML = '<option value="">Error al cargar</option>';
                });
        }
        
        function getLocationData() {
            return {
                region: document.getElementById('regionSelect').value,
                departamento: document.getElementById('departamentoSelect').value,
                localidad: document.getElementById('localidadInput').value
            };
        }
        
        function handleFileSelect(event) {
            selectedFile = event.target.files[0];
            if (!selectedFile) return;
            if (!selectedFile.type.match('image.*')) {
                showNotification('Por favor, selecciona solo imágenes','error');
                return;
            }
            if (selectedFile.size > 10*1024*1024) {
                showNotification('La imagen es demasiado grande (máx. 10MB)','error');
                return;
            }
            showNotification(`Imagen seleccionada: ${selectedFile.name}`,'info');
            document.getElementById('processBtn').disabled = false;
        }
        
        function processImage() {
            if (!selectedFile) {
                showNotification('Por favor, selecciona una imagen primero','error');
                return;
            }
            const locationData = getLocationData();
            if (!locationData.region && !confirm('No has seleccionado una región. ¿Deseas continuar con análisis general?')) {
                return;
            }
            showLoading('Analizando imagen... Determinando frutas y región...');
            const formData = new FormData();
            formData.append('imagen', selectedFile);
            formData.append('departamento', locationData.departamento);
            fetch('/upload', { method:'POST', body:formData })
                .then(response => response.json())
                .then(data => {
                    hideLoading();
                    if (data.error) {
                        showNotification(data.error,'error');
                        return;
                    }
                    showNotification('¡Análisis completado exitosamente!','success');
                    displayResults(data, 'upload');
                })
                .catch(() => {
                    hideLoading();
                    showNotification('Error al procesar la imagen. Intenta nuevamente.','error');
                });
        }
        
        function clearUpload() {
            selectedFile = null;
            document.getElementById('fileInput').value = '';
            document.getElementById('processBtn').disabled = true;
            const preview = document.querySelector('#uploadPanel .image-preview');
            if (preview) preview.remove();
            document.getElementById('uploadResults').style.display = 'none';
            document.getElementById('uploadResults').innerHTML = '';
        }
        
        function toggleCamera() {
            const btn = document.getElementById('cameraToggleBtn');
            const captureBtn = document.getElementById('captureBtn');
            const video = document.getElementById('videoElement');
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
                video.srcObject = null;
                btn.innerHTML = '<i class="fas fa-power-off"></i> Encender Cámara';
                btn.classList.remove('btn-danger');
                captureBtn.disabled = true;
            } else {
                navigator.mediaDevices.getUserMedia({ 
                    video: { facingMode:'environment', width:{ideal:1280}, height:{ideal:720} }
                })
                .then(newStream => {
                    stream = newStream;
                    video.srcObject = stream;
                    btn.innerHTML = '<i class="fas fa-power-off"></i> Apagar Cámara';
                    btn.classList.add('btn-danger');
                    captureBtn.disabled = false;
                })
                .catch(err => {
                    showNotification('No se pudo acceder a la cámara. Verifica los permisos.','error');
                });
            }
        }
        
        function captureImage() {
            if (!stream) return;
            const locationData = getLocationData();
            const video = document.getElementById('videoElement');
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            showLoading('Capturando y analizando imagen...');
            canvas.toBlob(blob => {
                fetch('/captura', {
                    method: 'POST',
                    headers: { 'Accept':'application/json', 'Content-Type':'application/json' },
                    body: JSON.stringify({
                        imagen: canvas.toDataURL('image/jpeg', 0.9),
                        departamento: locationData.departamento
                    })
                })
                .then(response => response.json())
                .then(data => {
                    hideLoading();
                    if (data.error) {
                        showNotification(data.error,'error');
                        return;
                    }
                    showNotification('¡Captura analizada exitosamente!','success');
                    displayResults(data, 'camera');
                })
                .catch(() => {
                    hideLoading();
                    showNotification('Error al procesar la captura','error');
                });
            }, 'image/jpeg', 0.9);
        }
        
        function displayResults(data, source) {
            const resultsDiv = document.getElementById(`${source}Results`);
            resultsDiv.style.display = 'block';
            
            let maturityPercent = 50;
            if (data.porcentaje_maduracion) {
                const match = data.porcentaje_maduracion.match(/(\d+)/);
                if (match) maturityPercent = parseInt(match[1]);
            }
            
            resultsDiv.innerHTML = `
                <div class="results-container">
                    <div class="image-preview">
                        <h3><i class="fas fa-image"></i> Imagen Procesada</h3>
                        <img src="${data.url}" alt="Imagen procesada">
                        <div class="mt-2">
                            <p><strong>Fecha:</strong> ${data.timestamp}</p>
                            <p><strong>Ubicación:</strong> ${data.departamento || 'No especificado'}, ${data.region_peru || 'Perú'}</p>
                            <p><strong>Confianza promedio:</strong> ${(data.confidence_average * 100).toFixed(1)}%</p>
                        </div>
                    </div>
                    
                    <div class="detections-list">
                        <h3><i class="fas fa-search"></i> Detecciones (${data.detection_count})</h3>
                        <div id="detectionsList">
                            ${data.detecciones && data.detecciones.length > 0 
                                ? data.detecciones.map(d => `
                                    <div class="detection-item">
                                        <div>
                                            <strong>${d.clase}</strong>
                                            <div style="font-size:0.9em;color:#666;">
                                                Confianza: ${(d.conf*100).toFixed(1)}%
                                            </div>
                                        </div>
                                        <div style="font-size:1.2em;font-weight:bold;color:#4CAF50;">
                                            ${(d.conf*100).toFixed(0)}%
                                        </div>
                                    </div>
                                `).join('')
                                : '<p>No se detectaron frutas en la imagen.</p>'
                            }
                        </div>
                        
                        <div class="ia-analysis mt-3">
                            <div class="region-badge">
                                <i class="fas fa-map-marker-alt"></i> 
                                ${data.region_peru || 'Región Peruana'} 
                                ${data.departamento ? `- ${data.departamento}` : ''}
                            </div>
                            
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                                <h3 style="margin:0;"><i class="fas fa-brain"></i> Análisis de Inteligencia Artificial</h3>
                                <span class="btn" style="padding:5px 15px;font-size:0.9em;">
                                    <i class="fas fa-seedling"></i> ${data.estado_maduracion || 'Estado de maduración'}
                                </span>
                            </div>
                            
                            <div style="margin:1rem 0;">
                                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                                    <span>Porcentaje de Maduración:</span>
                                    <strong>${data.porcentaje_maduracion || 'N/A'}</strong>
                                </div>
                                <div class="maturity-bar">
                                    <div class="maturity-fill" style="width:${maturityPercent}%"></div>
                                </div>
                            </div>
                            
                            <div class="analysis-section">
                                <h4><i class="fas fa-info-circle"></i> Descripción para Agricultores Peruanos</h4>
                                <p>${data.descripcion_ia || 'Información en proceso...'}</p>
                            </div>
                            
                            <div class="analysis-section">
                                <h4><i class="fas fa-clipboard-check"></i> Recomendaciones Prácticas</h4>
                                <p>${data.recomendaciones || 'Recomendaciones específicas en proceso...'}</p>
                            </div>
                            
                            <div class="analysis-section">
                                <h4><i class="fas fa-cloud-sun"></i> Clima y Condiciones Ideales</h4>
                                <p>${data.clima_recomendado || 'Información climática en proceso...'}</p>
                            </div>
                            
                            <div class="analysis-section">
                                <h4><i class="fas fa-tractor"></i> Consejos de Cultivo</h4>
                                <p>${data.consejos_cultivo || 'Consejos de cultivo en proceso...'}</p>
                            </div>
                            
                            <div class="form-row mt-2">
                                <div class="analysis-section" style="flex:1;">
                                    <h4><i class="fas fa-clock"></i> Tiempo de Maduración</h4>
                                    <p>${data.tiempo_maduracion || 'Por determinar'}</p>
                                </div>
                                <div class="analysis-section" style="flex:1;">
                                    <h4><i class="fas fa-warehouse"></i> Almacenamiento</h4>
                                    <p>${data.almacenamiento || 'Recomendaciones en proceso'}</p>
                                </div>
                            </div>
                            
                            <div class="analysis-section">
                                <h4><i class="fas fa-store"></i> Mercado Local</h4>
                                <p>${data.mercado_local || 'Información de mercado en proceso...'}</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="text-center mt-3">
                    <button class="btn btn-success" onclick="showPanel('library')">
                        <i class="fas fa-save"></i> Guardar en Biblioteca
                    </button>
                    <button class="btn" onclick="clearResults('${source}')">
                        <i class="fas fa-times"></i> Cerrar Resultados
                    </button>
                </div>
            `;
            
            resultsDiv.scrollIntoView({behavior:'smooth'});
            if (currentPanel === 'library') loadLibrary();
        }
        
        function clearResults(source) {
            const div = document.getElementById(`${source}Results`);
            div.style.display = 'none';
            div.innerHTML = '';
        }
        
        function loadLibrary() {
            showLoading('Cargando biblioteca...');
            fetch('/biblioteca')
                .then(response => response.json())
                .then(data => {
                    hideLoading();
                    if (data.error) {
                        showNotification(data.error,'error');
                        return;
                    }
                    const grid = document.getElementById('libraryGrid');
                    grid.innerHTML = '';
                    if (data.length === 0) {
                        grid.innerHTML = `
                            <div class="text-center" style="grid-column:1/-1;padding:3rem;color:#666;">
                                <i class="fas fa-inbox" style="font-size:4rem;margin-bottom:1rem;"></i>
                                <h3>Biblioteca Vacía</h3>
                                <p>No hay imágenes en la biblioteca todavía.</p>
                                <button class="btn mt-2" onclick="showPanel('upload')">
                                    <i class="fas fa-plus"></i> Subir Primera Imagen
                                </button>
                            </div>`;
                        return;
                    }
                    data.forEach(item => {
                        const card = document.createElement('div');
                        card.className = 'library-card';
                        card.innerHTML = `
                            <div class="card-image">
                                <img src="${item.thumbnail_url}" alt="${item.original_filename}" loading="lazy">
                            </div>
                            <div class="card-content">
                                <div class="card-header">
                                    <small>${item.timestamp}</small>
                                    <button class="delete-btn" onclick="deleteImage(${item.id},this)">
                                        <i class="fas fa-trash"></i>
                                    </button>
                                </div>
                                <h4>${item.original_filename}</h4>
                                <p><i class="fas fa-map-marker-alt"></i> <strong>Ubicación:</strong> ${item.departamento||'N/A'}, ${item.region_peru||'Perú'}</p>
                                <p><i class="fas fa-search"></i> <strong>Detecciones:</strong> ${item.detection_count}</p>
                                <p><i class="fas fa-chart-line"></i> <strong>Confianza:</strong> ${(item.confidence_average*100).toFixed(1)}%</p>
                                ${item.detecciones&&item.detecciones.length>0?`<p><i class="fas fa-apple-alt"></i> <strong>Frutas:</strong> ${item.detecciones.slice(0,3).map(d=>d.clase).join(', ')}${item.detecciones.length>3?'...':''}</p>`:''}
                                <button class="btn w-100 mt-2" onclick="viewDetails(${item.id})">
                                    <i class="fas fa-eye"></i> Ver Análisis Completo
                                </button>
                            </div>`;
                        grid.appendChild(card);
                    });
                })
                .catch(() => {
                    hideLoading();
                    showNotification('Error al cargar la biblioteca','error');
                });
        }
        
        function deleteImage(id, button) {
            if (!confirm('¿Estás seguro de eliminar esta imagen y todos sus datos?')) return;
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            fetch(`/delete/${id}`, {method:'DELETE'})
                .then(response => response.json())
                .then(data => {
                    button.disabled = false;
                    button.innerHTML = '<i class="fas fa-trash"></i>';
                    if (data.error) {
                        showNotification(data.error,'error');
                        return;
                    }
                    showNotification('Imagen eliminada correctamente','success');
                    button.closest('.library-card').remove();
                    if (currentPanel === 'admin') loadStats();
                })
                .catch(() => {
                    button.disabled = false;
                    button.innerHTML = '<i class="fas fa-trash"></i>';
                    showNotification('Error al eliminar la imagen','error');
                });
        }
        
        function viewDetails(id) {
            fetch('/biblioteca')
                .then(response => response.json())
                .then(data => {
                    const item = data.find(i => i.id === id);
                    if (item) {
                        showPanel('upload');
                        setTimeout(() => displayResults(item, 'upload'), 100);
                    }
                });
        }
        
        function refreshLibrary() {
            showNotification('Actualizando biblioteca...','info');
            loadLibrary();
        }
        
        function clearLibrary() {
            if (!confirm('¿Estás seguro de eliminar TODAS las imágenes y datos? Esta acción no se puede deshacer.')) return;
            showLoading('Limpiando biblioteca...');
            fetch('/reset', {method:'POST'})
                .then(response => response.json())
                .then(data => {
                    hideLoading();
                    if (data.error) {
                        showNotification(data.error,'error');
                        return;
                    }
                    showNotification('Biblioteca limpiada completamente','success');
                    loadLibrary();
                    loadStats();
                })
                .catch(() => {
                    hideLoading();
                    showNotification('Error al limpiar la biblioteca','error');
                });
        }
        
        function loadStats() {
            fetch('/estadisticas')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        showNotification(data.error,'error');
                        return;
                    }
                    document.getElementById('statsGrid').innerHTML = `
                        <div class="stat-card"><h3>${data.total||0}</h3><p>Total de Imágenes</p></div>
                        <div class="stat-card"><h3>${data.detectadas||0}</h3><p>Con Detecciones</p></div>
                        <div class="stat-card"><h3>${data.no_detectadas||0}</h3><p>Sin Detecciones</p></div>
                        <div class="stat-card"><h3>${Object.keys(data.clases||{}).length}</h3><p>Tipos de Frutas</p></div>
                        <div class="stat-card"><h3>${Object.keys(data.regiones||{}).length}</h3><p>Regiones</p></div>
                        <div class="stat-card"><h3>${Object.keys(data.departamentos||{}).length}</h3><p>Departamentos</p></div>`;
                    
                    const regionCtx = document.getElementById('regionChart').getContext('2d');
                    if (regionChart) regionChart.destroy();
                    regionChart = new Chart(regionCtx, {
                        type:'pie',
                        data:{
                            labels:Object.keys(data.regiones||{}),
                            datasets:[{data:Object.values(data.regiones||{}), backgroundColor:['#FF6384','#36A2EB','#FFCE56','#4BC0C0','#9966FF','#FF9F40'].slice(0,Object.keys(data.regiones||{}).length), borderWidth:2, borderColor:'rgba(255,255,255,0.8)'}]
                        },
                        options:{responsive:true, maintainAspectRatio:false, plugins:{legend:{position:'right', labels:{color:'white', font:{size:12}}}}}}
                    });
                    
                    const fruitCtx = document.getElementById('fruitChart').getContext('2d');
                    if (fruitChart) fruitChart.destroy();
                    const topFruits = (data.top_frutas||[]).slice(0,8);
                    fruitChart = new Chart(fruitCtx, {
                        type:'bar',
                        data:{
                            labels:topFruits.map(f=>f[0]),
                            datasets:[{label:'Cantidad', data:topFruits.map(f=>f[1]), backgroundColor:'rgba(255,215,0,0.7)', borderColor:'rgba(255,215,0,1)', borderWidth:2}]
                        },
                        options:{
                            responsive:true, maintainAspectRatio:false,
                            scales:{y:{beginAtZero:true, ticks:{color:'white'}, grid:{color:'rgba(255,255,255,0.1)'}}, x:{ticks:{color:'white', maxRotation:45}, grid:{color:'rgba(255,255,255,0.1)'}}},
                            plugins:{legend:{labels:{color:'white', font:{size:12}}}}
                        }
                    });
                    
                    const activityCtx = document.getElementById('activityChart').getContext('2d');
                    if (activityChart) activityChart.destroy();
                    const days = ['Lun','Mar','Mié','Jue','Vie','Sáb','Dom'];
                    const activityData = days.map(()=>Math.floor(Math.random()*20)+5);
                    activityChart = new Chart(activityCtx, {
                        type:'line',
                        data:{
                            labels:days,
                            datasets:[{label:'Imágenes por día', data:activityData, borderColor:'#FFD700', backgroundColor:'rgba(255,215,0,0.1)', borderWidth:3, fill:true, tension:0.4}]
                        },
                        options:{
                            responsive:true, maintainAspectRatio:false,
                            scales:{y:{beginAtZero:true, ticks:{color:'white'}, grid:{color:'rgba(255,255,255,0.1)'}}, x:{ticks:{color:'white'}, grid:{color:'rgba(255,255,255,0.1)'}}},
                            plugins:{legend:{labels:{color:'white', font:{size:12}}}}
                        }
                    });
                })
                .catch(() => showNotification('Error al cargar estadísticas','error'));
        }
        
        function resetSystem() {
            if (!confirm('¿Estás seguro de resetear todo el sistema? Se eliminarán TODAS las imágenes y datos.')) return;
            showLoading('Reseteando sistema...');
            fetch('/reset', {method:'POST'})
                .then(response => response.json())
                .then(data => {
                    hideLoading();
                    if (data.error) {
                        showNotification(data.error,'error');
                        return;
                    }
                    showNotification('Sistema reseteado correctamente','success');
                    loadStats();
                    loadLibrary();
                })
                .catch(() => {
                    hideLoading();
                    showNotification('Error al resetear el sistema','error');
                });
        }
        
        function showNotification(message, type='info') {
            const notification = document.getElementById('notification');
            const text = document.getElementById('notificationText');
            const icon = notification.querySelector('i');
            if (type==='success') icon.className='fas fa-check-circle';
            else if (type==='error') icon.className='fas fa-exclamation-circle';
            else icon.className='fas fa-info-circle';
            notification.className=`notification ${type} show`;
            text.textContent = message;
            setTimeout(() => notification.classList.remove('show'), 4000);
        }
        
        function showLoading(message='Procesando...') {
            document.getElementById('loadingDetail').textContent = message;
            document.getElementById('loading').classList.add('active');
        }
        
        function hideLoading() {
            document.getElementById('loading').classList.remove('active');
        }
        
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape' && stream) toggleCamera();
        });
        
        document.addEventListener('click', e => {
            if (window.innerWidth <= 768) {
                const sidebar = document.getElementById('sidebar');
                const menuToggle = document.getElementById('menuToggle');
                if (!sidebar.contains(e.target) && !menuToggle.contains(e.target) && sidebar.classList.contains('active')) {
                    sidebar.classList.remove('active');
                }
            }
        });
    </script>
</body>
</html>'''

def create_html_template():
    template_path = os.path.join(BASE_DIR, 'templates', 'index.html')
    os.makedirs(os.path.dirname(template_path), exist_ok=True)
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(HTML_TEMPLATE)

def create_static_folders():
    folders = ['static/sounds', 'static/css', 'static/js']
    for folder in folders:
        os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)
    css_path = os.path.join(BASE_DIR, 'static/css', 'styles.css')
    if not os.path.exists(css_path):
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write('/* Estilos adicionales */')

if __name__ == "__main__":
    create_html_template()
    create_static_folders()
    app.run(host='127.0.0.1', port=5000, debug=True)
