from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()
# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Especifica dominios en producción (mejor práctica)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga el modelo YOLOv8
modelo = YOLO('best-armas.pt')

@app.get("/")
async def root():
    """
    Esta ruta redirige a la documentación de la API en Swagger.
    """
    return {"Domo de Hierro": "¡Bienvenido a la API! Accede a la documentación en /docs"}

@app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    """
    Endpoint para realizar predicciones sobre una imagen cargada.

    Se espera que se suba una imagen, y el modelo YOLO entrenado realizará la detección de objetos.
    """
    # Leer los bytes de la imagen
    image_bytes = await file.read()
    
    # Cargar la imagen desde los bytes
    img = Image.open(io.BytesIO(image_bytes))

    # Realizar la predicción
    results = modelo(img)

    # Procesar los resultados y devolverlos como un diccionario
    predictions = []
    for box in results[0].boxes.data.tolist():
        predictions.append({
            "índice_clase": int(box[5]),  # Índice de clase
            "confianza": float(box[4]),  # Confianza
            "coordenadas": box[:4]  # Coordenadas (x_min, y_min, x_max, y_max)
        })

    return {"predicciones": predictions}

