import sqlite3
import json
from config import DB_PATH


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS biblioteca (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp           TEXT,
        detecciones         TEXT,
        url                 TEXT,
        thumbnail_url       TEXT,
        has_detection       BOOLEAN,
        original_filename   TEXT,
        source              TEXT,
        confidence_average  REAL,
        detection_count     INTEGER,
        region_peru         TEXT,
        departamento        TEXT,
        descripcion_ia      TEXT,
        recomendaciones     TEXT,
        porcentaje_maduracion TEXT,
        clima_recomendado   TEXT,
        consejos_cultivo    TEXT,
        tiempo_maduracion   TEXT,
        almacenamiento      TEXT,
        mercado_local       TEXT,
        created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()
    print(f"Base de datos inicializada en: {DB_PATH}")


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
