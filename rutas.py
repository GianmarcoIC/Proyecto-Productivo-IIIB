import os
import json
import base64
import cv2
import numpy as np
from datetime import datetime, timedelta
from flask import request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename

from config import (
    UPLOAD_FOLDER, PROCESSED_FOLDER, THUMBNAIL_FOLDER,
    STATISTICS_MAX_RECORDS
)
from base_datos import get_db_connection, save_to_db, init_db
from funciones import (
    allowed_file, determinar_region_peru,
    generar_analisis_completo, generar_respuesta_por_defecto,
    consultar_perplexity_mejorado
)


def register_routes(app, model):
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
        from config import REGIONES_PERU
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
                from config import ALLOWED_EXTENSIONS
                return jsonify({"error": f"Tipo de archivo no permitido. Use: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

            departamento = request.form.get('departamento', '')

            img_bytes = file.read()
            npimg = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({"error": "Imagen inválida o corrupta"}), 400

            entry = _procesar_imagen(img, model, file.filename, departamento, source="upload")
            entry["id"] = save_to_db(entry)
            return jsonify(entry)

        except Exception as e:
            print(f"Error en upload: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/captura", methods=["POST"])
    def captura():
        try:
            data = request.json
            if 'imagen' not in data:
                return jsonify({"error": "No se encontró imagen en la captura"}), 400

            departamento = data.get('departamento', '')

            img_b64 = data['imagen']
            if 'base64,' in img_b64:
                img_b64 = img_b64.split(',')[1]
            img_data = base64.b64decode(img_b64)
            npimg = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({"error": "Imagen inválida o corrupta"}), 400

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            entry = _procesar_imagen(img, model, f"capture_{timestamp}.jpg", departamento, source="camera")
            entry["id"] = save_to_db(entry)
            return jsonify(entry)

        except Exception as e:
            print(f"Error en captura: {str(e)}")
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
                FROM biblioteca ORDER BY created_at DESC
            ''')
            rows = cursor.fetchall()
            conn.close()

            biblioteca = []
            for row in rows:
                item = dict(row)
                item['detecciones'] = json.loads(item['detecciones']) if item['detecciones'] else []
                biblioteca.append(item)

            return jsonify(biblioteca)

        except Exception as e:
            print(f"Error en biblioteca: {str(e)}")
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

            url           = row['url']
            thumbnail_url = row['thumbnail_url']

            try:
                if url and url.startswith('/processed/'):
                    file_path = os.path.join(PROCESSED_FOLDER, url.split('/')[-1])
                    if os.path.exists(file_path):
                        os.remove(file_path)
                if thumbnail_url and thumbnail_url.startswith('/thumbnails/'):
                    file_path = os.path.join(THUMBNAIL_FOLDER, thumbnail_url.split('/')[-1])
                    if os.path.exists(file_path):
                        os.remove(file_path)
            except Exception as e:
                print(f"Error eliminando archivos: {e}")

            cursor.execute('DELETE FROM biblioteca WHERE id = ?', (db_id,))
            conn.commit()
            conn.close()
            return jsonify({"message": "Registro e imágenes eliminados correctamente"})

        except Exception as e:
            print(f"Error en delete: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/estadisticas")
    def get_estadisticas():
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) AS total_historico FROM biblioteca')
            total_historico = cursor.fetchone()['total_historico']

            cursor.execute('''
                SELECT * FROM biblioteca
                ORDER BY created_at DESC, id DESC
                LIMIT ?
            ''', (STATISTICS_MAX_RECORDS,))
            rows = cursor.fetchall()
            conn.close()

            biblioteca = []
            for row in rows:
                item = dict(row)
                item['detecciones'] = json.loads(item['detecciones']) if item['detecciones'] else []
                biblioteca.append(item)

            total       = len(biblioteca)
            detectadas  = sum(1 for e in biblioteca if e["has_detection"])
            no_detectadas = total - detectadas

            conteo_clases, confidence_by_class = {}, {}
            regiones_count, departamentos_count = {}, {}

            hoy = datetime.now().date()
            dias_actividad    = [hoy - timedelta(days=i) for i in range(6, -1, -1)]
            actividad_por_dia = {dia.strftime("%Y-%m-%d"): 0 for dia in dias_actividad}
            nombres_dias = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom']

            for entry in biblioteca:
                created_at = entry.get("created_at") or ""
                try:
                    fecha_entry = datetime.fromisoformat(created_at.replace("Z", "+00:00")).date()
                    fecha_key = fecha_entry.strftime("%Y-%m-%d")
                    if fecha_key in actividad_por_dia:
                        actividad_por_dia[fecha_key] += 1
                except ValueError:
                    pass

                region = entry.get("region_peru", "No especificada")
                regiones_count[region] = regiones_count.get(region, 0) + 1

                depto = entry.get("departamento", "")
                if depto:
                    departamentos_count[depto] = departamentos_count.get(depto, 0) + 1

                for d in entry["detecciones"]:
                    clase = d["clase"]
                    conf  = d["conf"]
                    conteo_clases[clase] = conteo_clases.get(clase, 0) + 1
                    confidence_by_class.setdefault(clase, []).append(conf)

            avg_confidence_by_class = {
                clase: round(sum(confs) / len(confs), 2)
                for clase, confs in confidence_by_class.items()
            }
            top_departamentos = sorted(departamentos_count.items(), key=lambda x: x[1], reverse=True)[:5]

            return jsonify({
                "total":               total,
                "total_historico":     total_historico,
                "limite_estadisticas": STATISTICS_MAX_RECORDS,
                "detectadas":          detectadas,
                "no_detectadas":       no_detectadas,
                "clases":              conteo_clases,
                "avg_confidence_by_class": avg_confidence_by_class,
                "regiones":            regiones_count,
                "departamentos":       dict(top_departamentos),
                "top_frutas":          sorted(conteo_clases.items(), key=lambda x: x[1], reverse=True)[:10],
                "actividad_labels":    [
                    f"{nombres_dias[dia.weekday()]} {dia.strftime('%d/%m')}"
                    for dia in dias_actividad
                ],
                "actividad_valores": list(actividad_por_dia.values())
            })

        except Exception as e:
            print(f"Error en estadisticas: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/reset", methods=["POST"])
    def reset_database():
        try:
            for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, THUMBNAIL_FOLDER]:
                for fname in os.listdir(folder):
                    fpath = os.path.join(folder, fname)
                    if os.path.isfile(fpath):
                        os.remove(fpath)
            init_db()
            return jsonify({"message": "Sistema reseteado correctamente"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/test-perplexity")
    def test_perplexity():
        try:
            respuesta = consultar_perplexity_mejorado(
                "Di hola en español y dime cuál es la capital de Perú en una sola línea."
            )
            if respuesta:
                return jsonify({"status": "success", "message": "Conexión exitosa con Perplexity API", "response": respuesta[:200]})
            return jsonify({"status": "error", "message": "No se pudo conectar con Perplexity API"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})


def _procesar_imagen(img, model, original_filename, departamento, source):
    results   = model(img, conf=0.4, verbose=False)
    annotated = results[0].plot()

    detecciones      = []
    frutas_detectadas = []
    confianzas       = []
    has_detection    = False
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        has_detection = True
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf   = round(float(box.conf[0].item()), 2)
            clase  = model.names[cls_id]
            detecciones.append({"clase": clase, "conf": conf})
            frutas_detectadas.append(clase)
            confianzas.append(conf)

    confidence_average = sum(d['conf'] for d in detecciones) / len(detecciones) if detecciones else 0
    detection_count    = len(detecciones)
    region             = determinar_region_peru(frutas_detectadas, departamento)

    analisis_ia = (
        generar_analisis_completo(frutas_detectadas, confianzas, region, departamento)
        if frutas_detectadas
        else generar_respuesta_por_defecto(region, "No detectadas")
    )

    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name      = secure_filename(original_filename)
    base_name      = os.path.splitext(safe_name)[0]

    cv2.imwrite(os.path.join(UPLOAD_FOLDER,    f"{base_name}_{timestamp}_original.jpg"), img)

    processed_filename  = f"{base_name}_{timestamp}_processed.jpg"
    cv2.imwrite(os.path.join(PROCESSED_FOLDER, processed_filename), annotated)

    thumbnail_filename  = f"{base_name}_{timestamp}_thumb.jpg"
    cv2.imwrite(os.path.join(THUMBNAIL_FOLDER, thumbnail_filename), cv2.resize(img, (200, 200)))

    return {
        "timestamp":          datetime.now().strftime("%d/%m/%Y %H:%M"),
        "detecciones":        detecciones,
        "url":                f"/processed/{processed_filename}",
        "thumbnail_url":      f"/thumbnails/{thumbnail_filename}",
        "has_detection":      has_detection,
        "original_filename":  safe_name,
        "source":             source,
        "confidence_average": round(confidence_average, 2),
        "detection_count":    detection_count,
        "region_peru":        region,
        "departamento":       departamento,
        "descripcion_ia":     analisis_ia['descripcion'],
        "recomendaciones":    analisis_ia['recomendaciones'],
        "porcentaje_maduracion": analisis_ia['porcentaje_maduracion'],
        "clima_recomendado":  analisis_ia['clima'],
        "consejos_cultivo":   analisis_ia['consejos_cultivo'],
        "tiempo_maduracion":  analisis_ia['tiempo_maduracion'],
        "almacenamiento":     analisis_ia['almacenamiento'],
        "mercado_local":      analisis_ia['mercado_local'],
        "estado_maduracion":  analisis_ia.get('estado_maduracion', '')
    }
