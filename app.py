from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import joblib
import numpy as np


from numpy.linalg import norm
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Fashion Recommendation System")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
if os.path.isdir(IMAGES_DIR):
    app.mount("/images", StaticFiles(directory="images"), name="images")

templates = Jinja2Templates(directory="templates")

MODEL = None
FEATURES = None
FILE_PATHS = None
NN_INDEX = None


def build_model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    return tensorflow.keras.Sequential([base_model, GlobalMaxPooling2D()])


def feature_extraction(image_path: str):
    img = tensorflow.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tensorflow.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    embedding = MODEL.predict(img_preprocessed, verbose=0).flatten()
    normalized = embedding / norm(embedding)
    return normalized


@app.on_event("startup")
def load_assets():
    global MODEL, FEATURES, FILE_PATHS, NN_INDEX
    MODEL = build_model()

    try:
        FEATURES = np.array(joblib.load(os.path.join(BASE_DIR, "featuers.pkl")))
        FILE_PATHS = joblib.load(os.path.join(BASE_DIR, "images.pkl"))
    except Exception as exc:
        raise RuntimeError("Could not load feature assets. Make sure featuers.pkl and images.pkl exist.") from exc

    if FEATURES.size:
        NN_INDEX = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")
        NN_INDEX.fit(FEATURES)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename)
    if not filename:
        raise HTTPException(status_code=400, detail="Upload a valid image file.")

    extension = os.path.splitext(filename)[1].lower()
    if extension not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="Please upload a .jpg, .jpeg, or .png image.")

    saved_path = os.path.join(UPLOAD_DIR, filename)
    with open(saved_path, "wb") as buffer:
        buffer.write(await file.read())

    if MODEL is None or NN_INDEX is None:
        raise HTTPException(status_code=500, detail="Model or recommendation index is not available.")

    features = feature_extraction(saved_path)
    distances, indices = NN_INDEX.kneighbors(features.reshape(1, -1))

    recommendations = []
    for idx in indices[0]:
        image_src = FILE_PATHS[idx]
        image_name = os.path.basename(image_src)
        recommendations.append(f"/images/{image_name}")

    return JSONResponse({
        "uploaded": f"/uploads/{filename}",
        "recommendations": recommendations,
    })
