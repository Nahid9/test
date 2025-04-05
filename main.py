from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomHeight, RandomWidth
from tensorflow.keras.utils import custom_object_scope
import numpy as np
from PIL import Image
import io
import base64
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import tensorflow as tf

# Initialize FastAPI
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="static")

# Model configurations
CROP_MODELS = {
    "corn": {
        "model_path": "mobilenet_corn.h5",
        "class_names": ["Corn___Common_Rust", "Corn___Gray_Leaf_Spot", "Corn___Healthy", "Corn___Leaf_Blight"],
        "display_name": "🌽 Corn Disease Classifier"
    },
    "potato": {
        "model_path": "mobilenet_potato.h5",
        "class_names": ["Potato___Early_Blight", "Potato___Healthy", "Potato___Late_Blight"],
        "display_name": "🥔 Potato Disease Classifier"
    },
    "rice": {
        "model_path": "mobilenet_rice.h5",
        "class_names": ["Rice___Brown_Spot", "Rice___Healthy", "Rice___Hispa", "Rice___Leaf_Blast"],
        "display_name": "🌿 Rice Disease Classifier"
    },
    "wheat": {
        "model_path": "mobilenet_wheat.h5",
        "class_names": ["Wheat___Brown_Rust", "Wheat___Healthy", "Wheat___Yellow_Rust"],
        "display_name": "🌾 Wheat Disease Classifier"
    }
}

# Common model settings
IMG_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Custom objects for model loading
custom_objects = {
    "RandomFlip": RandomFlip,
    "RandomRotation": RandomRotation,
    "RandomZoom": RandomZoom,
    "RandomHeight": RandomHeight,
    "RandomWidth": RandomWidth
}

# Load all models at startup
models = {}
for crop, config in CROP_MODELS.items():
    with custom_object_scope(custom_objects):
        models[crop] = load_model(config["model_path"])
    # Format class names
    CROP_MODELS[crop]["formatted_class_names"] = [
        name.replace("___", " ").replace("_", " ") for name in config["class_names"]
    ]

# Helper functions
def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(img, axis=0)

def denormalize_image(image):
    image = image * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(image, 0, 1)

def get_lime_explanation(image_array, model):
    def model_predict(images):
        processed_images = []
        for img in images:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
            processed_images.append(img)
        processed_images = np.array(processed_images)
        return model.predict(processed_images)
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_array[0],
        model_predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    
    temp = denormalize_image(temp)
    explained_img = mark_boundaries(temp, mask)
    return explained_img

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/classifier/{crop_type}", response_class=HTMLResponse)
async def crop_classifier(request: Request, crop_type: str):
    if crop_type not in CROP_MODELS:
        return RedirectResponse(url="/")
    return templates.TemplateResponse(
        f"{crop_type}.html",
        {
            "request": request,
            "crop_type": crop_type,
            "display_name": CROP_MODELS[crop_type]["display_name"]
        }
    )

@app.post("/predict/{crop_type}", response_class=HTMLResponse)
async def predict(request: Request, crop_type: str, file: UploadFile = File(...)):
    if crop_type not in CROP_MODELS:
        return RedirectResponse(url="/")
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        img = preprocess_image(io.BytesIO(image_bytes))
        
        # Get original image for display
        original_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_img = original_img.resize(IMG_SIZE)
        
        # Predict
        model = models[crop_type]
        preds = model.predict(img)
        class_idx = np.argmax(preds[0])
        confidence = float(np.max(preds[0]))
        confidence_percent = round(confidence * 100)
        predicted_class = CROP_MODELS[crop_type]["formatted_class_names"][class_idx]
        
        # Get LIME explanation
        lime_explanation = get_lime_explanation(img, model)
        
        # Create side-by-side plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original image
        ax1.imshow(original_img)
        ax1.set_title('Original Image', fontsize=38)
        ax1.axis('off')
        
        # LIME explanation
        ax2.imshow(lime_explanation)
        ax2.set_title('LIME Explanation', fontsize=38)
        ax2.axis('off')
        
        plt.tight_layout()
        explanation_plot = fig_to_base64(fig)
        plt.close(fig)
        
        # Return result
        return templates.TemplateResponse(
            f"{crop_type}.html",
            {
                "request": request,
                "crop_type": crop_type,
                "display_name": CROP_MODELS[crop_type]["display_name"],
                "prediction": predicted_class,
                "confidence": confidence_percent,
                "confidence_raw": confidence,
                "image_uploaded": True,
                "explanation_plot": explanation_plot
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            f"{crop_type}.html",
            {
                "request": request,
                "crop_type": crop_type,
                "display_name": CROP_MODELS[crop_type]["display_name"],
                "error": f"Error: {str(e)}"
            }
        )