from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
from PIL import Image

app = FastAPI()
templates = Jinja2Templates(directory="templates")  # Assuming your templates are in a directory named "templates"

model = load_model('pneumonia_densenet201v2.h5')


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("template.html", {"request": request, "prediction": None})


@app.post("/uploadfile/")
async def create_upload_file(request: Request, file: UploadFile = File(...)):
    pil_image = read_image(await file.read())
    prediction = predictor(pil_image)

    return {"prediction": prediction}


def read_image(file) -> Image.Image:
    pil_image = Image.open(BytesIO(file))
    return pil_image


def predictor(file: Image.Image):
    img = file.resize((224, 224))
    x = image.img_to_array(img)
    if x.shape[-1] == 1:
        x = np.concatenate([x] * 3, axis=-1)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    images = np.vstack([x])

    predictions = model.predict(images)
    if predictions[0] > 0.5:
        predicted_class = 'X-ray image is classified Pneumonia'
    else:
        predicted_class = 'X-ray image is classified Normal'

    response = {
        "class": predicted_class,
        "score": f"{float(predictions[0]):.10f}"
    }
    return response
