@app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    """
    Endpoint para realizar predicciones sobre una sola imagen cargada.
    """
    try:
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")


