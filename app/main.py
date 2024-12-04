
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Load model
model = tf.keras.models.load_model("models/model.h5")

print(model.summary())

def load_image_into_numpy_array(data):
    """Convert an image to a numpy array."""
    return np.array(Image.open(BytesIO(data)))

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    image_data = await image.read()
    np_image = load_image_into_numpy_array(image_data)
    np_image = np.expand_dims(np_image, axis=0)  # Model expects batches of images

    # Ensure the image is in the correct format (e.g., size, channels)
    # This might include resizing, normalization, etc., depending on your model
    # np_image = tf.image.resize(np_image, (1, 28,28, 1))  # Example resize operation

    np_image = np_image.reshape((1,28,28,1)).astype("float32")/ 255.0
    # Prediction
    predictions = model.predict(np_image)
    predicted_class = np.argmax(predictions, axis=1)

    return JSONResponse(content={"predicted_class": int(predicted_class)})
