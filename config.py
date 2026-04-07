import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER    = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
THUMBNAIL_FOLDER = os.path.join(BASE_DIR, 'thumbnails')
DB_PATH          = os.path.join(BASE_DIR, 'fruit_classifier2.db')

for _folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, THUMBNAIL_FOLDER]:
    os.makedirs(_folder, exist_ok=True)

ALLOWED_EXTENSIONS   = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
STATISTICS_MAX_RECORDS = 100

PERPLEXITY_API_KEY = "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"
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

MODEL_PATH = "modelo/best_yolov8s_fruits_v7.pt"

REGIONES_PERU = {
    'Costa': {
        'frutas': ['limón', 'mandarina', 'naranja', 'palta', 'plátano', 'mango',
                   'uva', 'maracuyá', 'granadilla', 'fresa', 'sandía', 'melón'],
        'clima': 'Cálido y seco',
        'departamentos': ['Lima', 'La Libertad', 'Lambayeque', 'Piura',
                          'Ica', 'Arequipa', 'Moquegua', 'Tacna']
    },
    'Sierra': {
        'frutas': ['manzana', 'pera', 'durazno', 'tuna', 'granadilla', 'aguaymanto',
                   'chirimoya', 'membrillo', 'capulí', 'lúcuma', 'pacae'],
        'clima': 'Templado y frío',
        'departamentos': ['Cusco', 'Puno', 'Junín', 'Huánuco', 'Ancash',
                          'Cajamarca', 'Ayacucho', 'Apurímac']
    },
    'Selva': {
        'frutas': ['piña', 'papaya', 'maracuyá', 'coco', 'guayaba', 'camu camu',
                   'aguaje', 'plátano', 'piñón', 'cacao', 'ají', 'castaña'],
        'clima': 'Cálido y húmedo',
        'departamentos': ['Loreto', 'Ucayali', 'San Martín', 'Madre de Dios', 'Amazonas']
    }
}
