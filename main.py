import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import pandas as pd
from io import BytesIO
from CNN import CNN  # Assuming CNN.py contains the CNN model
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Initialize FastAPI app
app = FastAPI()
static_folder_path = Path(r"C:/Users/muska/OneDrive/Desktop/mushu/Plant-Disease-Detection-main/Flask Deployed App")

# Mount the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")



# Set up templates (for HTML rendering)
templates = Jinja2Templates(directory="templates")  # Ensure the 'templates' folder contains your HTML files

# Detect device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CSV data
disease_info = pd.read_csv(r"C:\Users\muska\OneDrive\Desktop\mushu\Plant-Disease-Detection-main\Flask Deployed App\disease_info.csv", encoding='cp1252')
supplement_info = pd.read_csv(r"C:\Users\muska\OneDrive\Desktop\mushu\Plant-Disease-Detection-main\Flask Deployed App\supplement_info.csv", encoding='cp1252')

# Load the CNN Model
num_classes = 34  # Ensure consistency with training
model = CNN(num_classes)
model.load_state_dict(torch.load(r"C:\Users\muska\OneDrive\Desktop\mushu\Plant-Disease-Detection-main\Flask Deployed App\final_plant_disease_model.pt", map_location=device))
model.to(device)
model.eval()

# Reverse index mapping for disease names
transform_index_to_disease = dict(enumerate(disease_info["disease_name"]))

# Image processing and prediction function
def single_prediction(image_data):
    image = Image.open(BytesIO(image_data))
    image = image.resize((224, 224))

    # Convert to tensor and normalize
    input_data = TF.to_tensor(image)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    input_data = TF.normalize(input_data, mean=mean, std=std)
    input_data = input_data.view((-1, 3, 224, 224)).to(device)

    # Model prediction
    output = model(input_data)
    softmax_output = torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy()

    # Get top prediction
    index = np.argmax(softmax_output)
    confidence = softmax_output[0, index] * 100  # Convert to percentage
    predicted_disease = transform_index_to_disease.get(index, "Unknown Disease")

    return predicted_disease, confidence, index

# FastAPI route to render the homepage
@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# FastAPI route to render the AI engine page
@app.get("/index", response_class=HTMLResponse)
async def ai_engine_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# FastAPI route to handle file upload and prediction
@app.post("/submit", response_class=HTMLResponse)
async def submit(request: Request, image: UploadFile = File(...)):
    # Read the uploaded file content
    image_data = await image.read()

    # Run prediction
    predicted_disease, confidence, pred_index = single_prediction(image_data)

    # Check if the disease name exists in the disease_info DataFrame
    disease_row = disease_info[disease_info["disease_name"] == predicted_disease]

    if disease_row.empty:
        return templates.TemplateResponse("submit.html", {
            "request": request,
            "title": "Unknown Disease",
            "desc": "We couldn't identify the disease.",
            "prevent": "Please consult a plant specialist.",
            "image_url": "",
            "confidence": f"{confidence:.2f}%",
            "sname": "N/A",
            "simage": "",
            "buy_link": "",
            "pred": pred_index
        })

    # If disease found, proceed with fetching data
    disease_row = disease_row.iloc[0]  # Now safe to access the first row
    title = disease_row["disease_name"]
    description = disease_row["description"]
    prevent = disease_row["Possible Steps"]
    image_url = disease_row["image_url"]

    # Fetch supplement-related data
    supplement_row = supplement_info[supplement_info["supplement name"] == predicted_disease]

    if supplement_row.empty:
        supplement_name = "No supplement found"
        supplement_image_url = ""
        supplement_buy_link = ""
    else:
        supplement_row = supplement_row.iloc[0]  # Safe to access
        supplement_name = supplement_row["supplement name"]
        supplement_image_url = supplement_row["supplement image"]
        supplement_buy_link = supplement_row["buy link"]

    # Return the response with all required data
    return templates.TemplateResponse("submit.html", {
        "request": request,
        "title": title,
        "desc": description,
        "prevent": prevent,
        "image_url": image_url,
        "confidence": f"{confidence:.2f}%",
        "sname": supplement_name,
        "simage": supplement_image_url,
        "buy_link": supplement_buy_link,
        "pred": pred_index  # Make sure `pred` is passed here
    })
