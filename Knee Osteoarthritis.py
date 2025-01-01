from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import numpy as np
import torch
from torchvision import models
from io import BytesIO

app = FastAPI()

# Load the ResNet50 model
model = models.resnet50(weights=None)  # Initialize ResNet50 without pretrained weights

# Modify the `fc` layer to match your trained model's output classes (2 in this case)
num_classes = 2  # Update this to match your dataset's number of classes
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the checkpoint
checkpoint = torch.load("D:\\best_knee_model_resnet5.pth", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint)  # Load the state dictionary

model.eval()  # Set model to evaluation mode

# Preprocessing function
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # Resize the image
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    if image_array.ndim == 2:  # Handle grayscale images
        image_array = np.stack([image_array] * 3, axis=-1)
    image_array = np.transpose(image_array, (2, 0, 1))  # Convert to (C, H, W)
    mean = np.array([0.485, 0.456, 0.406])  # ImageNet mean
    std = np.array([0.229, 0.224, 0.225])  # ImageNet std
    image_array = (image_array - mean[:, None, None]) / std[:, None, None]
    image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Endpoint to receive image and return prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return JSONResponse({"error": "Invalid image file"})

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        with torch.no_grad():
            prediction = model(processed_image)

        # Convert prediction to a response format
        predicted_class = torch.argmax(prediction, dim=1).item()
        if predicted_class == 0:
            result = "Healthy knee"
        else:
            result = "There is knee Osteoarthritis"

        confidence = torch.softmax(prediction, dim=1)[0, predicted_class].item()

        return JSONResponse({
            "result": result,
            "confidence": float(confidence)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)})

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
