
import os
from flask import Flask
from ultralytics import YOLO

from config import BASE_DIR, MODEL_PATH
from base_datos import init_db
from rutas import register_routes
from funciones import consultar_perplexity_mejorado

app = Flask(__name__, template_folder='templates', static_folder='static')

init_db()

_model_path = MODEL_PATH
if not os.path.exists(_model_path):
    print(f"⚠️  Modelo no encontrado en {_model_path}. Usando yolov8n.pt por defecto.")
    _model_path = "yolov8n.pt"

model = YOLO(_model_path)
print(f"✅ Modelo cargado: {_model_path}")

register_routes(app, model)

def create_static_folders():
    for folder in ['static/sounds', 'static/css', 'static/js']:
        os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

    css_path = os.path.join(BASE_DIR, 'static/css', 'styles.css')
    if not os.path.exists(css_path):
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write('/* Estilos adicionales */')

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  CLASIFICADOR DE FRUTAS PERUANAS - SISTEMA INTELIGENTE v2.0")
    print("=" * 70)

    create_static_folders()

    print("\nProbando conexión con Perplexity API...")
    try:
        test_resp = consultar_perplexity_mejorado("Di 'Hola' en español para confirmar conexión.")
        if test_resp:
            print("✅ Conexión exitosa con Perplexity API")
        else:
            print("⚠️  Sin conexión con Perplexity. Usando respuestas por defecto.")
    except Exception as e:
        print(f"⚠️  Error probando Perplexity: {e}")

    from config import DB_PATH, UPLOAD_FOLDER, PROCESSED_FOLDER
    print(f"\n📊 Panel Admin:        http://127.0.0.1:5000")
    print(f"💾 Base de datos:      {DB_PATH}")
    print(f"📁 Carpeta subidas:    {UPLOAD_FOLDER}")
    print(f"🖼️  Carpeta procesadas: {PROCESSED_FOLDER}")
    print(f"🗺️  Regiones:           Costa, Sierra, Selva")
    print(f"🧠 Análisis IA:        Habilitado")
    print("=" * 70 + "\n")

    app.run(host='127.0.0.1', port=5000, debug=True)
