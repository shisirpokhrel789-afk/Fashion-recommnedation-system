# Fashion Recommendation System

A FastAPI-based fashion recommendation web app that lets users upload an outfit image and receive 5 visually similar style recommendations using Deeplearning concepts and alogrithms

## Features

- FastAPI backend with image upload support
- Pretrained ResNet50 feature extraction
- Nearest neighbor search for recommendation retrieval
- Clean responsive UI with upload preview and recommendation gallery
- Ready for GitHub deployment and local development

## Files

- `app.py` — FastAPI application entrypoint
- `templates/index.html` — UI template for the web interface
- `static/main.css` — styling for the app
- `featuers.pkl` — saved image feature vectors
- `images.pkl` — saved image file paths used for recommendation retrieval
- `images/` — folder containing the reference images used by the recommender
- `uploads/` — runtime folder for uploaded user images
- `requirements.txt` — Python dependencies

## Requirements

- Python 3.9+
- `pip` installed

## Install

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run Locally

```bash
uvicorn app:app --reload
```

Open the app in your browser at `http://127.0.0.1:8000`.

## GitHub Setup



```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/shisirpokhrel789-afk/fashion-recommendation-system.git
git push -u origin main
```

Dataset_link:https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

