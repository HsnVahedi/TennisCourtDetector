# inference.py

import os
import io
import json
import logging
import base64

import torch
import mlflow.pytorch
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 1. model_fn: Load the model from the artifact directory (e.g., /opt/ml/model)
def model_fn(model_dir):
    """
    Deserialize and return the pre-trained model. SageMaker will call this
    function once when the hosting container starts up.
    """
    logger.info(f"Loading model from {model_dir}")
    
    # If you saved your model in MLflow format, you can use mlflow to load it:
    model = mlflow.pytorch.load_model(model_dir)
    model.eval()
    
    return model


# 2. input_fn: Deserialize incoming request into a PyTorch tensor
def input_fn(request_body, request_content_type):
    """
    Takes the request data and de-serializes it into an object for prediction.
    Assume the request is JSON with a base64-encoded image under the key 'image_data'.
    """
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        
        # For example, expect {"image_data": "<base64 string>"}
        if "image_data" not in data:
            raise ValueError("JSON request body must include 'image_data' key.")
        
        # Decode base64 string into bytes
        image_bytes = base64.b64decode(data["image_data"])
        
        # Convert bytes to a PIL image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Define the same image transforms you used in training
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])
        
        # Apply transforms and add batch dimension: shape (1, 3, 224, 224)
        input_tensor = transform(image).unsqueeze(0)
        
        return input_tensor
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


# 3. predict_fn: Perform prediction on the deserialized object, using your model
def predict_fn(input_data, model):
    """
    Apply the model to the incoming request data. 
    """
    # input_data is a torch.Tensor of shape (1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(input_data)
        # For binary classification, your final layer has 2 logits.
        # We can do something simple like picking the argmax for predicted class:
        _, predicted_class = torch.max(outputs, 1)
        
        # Optionally, you can also compute probabilities using softmax:
        probs = torch.softmax(outputs, dim=1)
        prob_of_positive = probs[0, 1].item()  # Probability that it's class "1"
        
    # Return a dictionary that has both the predicted label and probability:
    return {
        "predicted_label": int(predicted_class.item()),
        "prob_of_class_1": float(prob_of_positive)
    }


# 4. output_fn: Serialize the prediction result back to the client
def output_fn(prediction, accept):
    """
    Serializes the prediction output. By default, we’ll return JSON.
    """
    if accept == "application/json":
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
