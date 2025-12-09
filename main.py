from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from model import predict_flower
import os

app = FastAPI(title="Flower Detection API")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Check uploaded file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Please upload an image")

        # Save uploaded image temporarily
        temp_path = "temp_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Run the prediction
        predicted_class, probs = predict_flower(temp_path)

        # Remove temporary image
        os.remove(temp_path)

        # Prepare readable probability output
        probability_dict = {
            "daisy": float(probs[0]),
            "dandelion": float(probs[1]),
            "rose": float(probs[2]),
            "sunflower": float(probs[3]),
            "tulip": float(probs[4])
        }

        # Return JSON response
        return JSONResponse({
            "predicted_class": predicted_class,
            "all_probabilities": probability_dict
        })

    except Exception as e:
        # Return the exact error message for debugging
        return JSONResponse({"error": str(e)})
        

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
