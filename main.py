from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar "*" por dominios específicos en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga el modelo YOLOv8
modelo = YOLO('best-armas.pt')

@app.get("/")
async def root():
    """
    Ruta raíz que da la bienvenida y redirige a la documentación.
    """
    return {"mensaje": "Bienvenido a la API. Documentación disponible en /docs"}

@app.post('/predict/')
async def predict(files: list[UploadFile] = File(...)):
    """
    Endpoint para realizar predicciones sobre imágenes cargadas.

    Recibe una o más imágenes, y el modelo YOLO entrenado realiza la detección de objetos.
    """
    predictions = []

    for file in files:
        try:
            # Validar el tipo de archivo
            if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
                raise HTTPException(status_code=400, detail=f"Formato no soportado: {file.content_type}")

            # Leer los bytes de la imagen
            image_bytes = await file.read()
            
            # Cargar la imagen desde los bytes
            img = Image.open(io.BytesIO(image_bytes))

            # Realizar la predicción
            results = modelo(img)

            # Procesar los resultados y almacenarlos
            for box in results[0].boxes.data.tolist():
                predictions.append({
                    "archivo": file.filename,
                    "índice_clase": int(box[5]),  # Índice de clase
                    "confianza": float(box[4]),  # Confianza
                    "coordenadas": box[:4]  # Coordenadas (x_min, y_min, x_max, y_max)
                })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al procesar {file.filename}: {str(e)}")

    return {"predicciones": predictions}


