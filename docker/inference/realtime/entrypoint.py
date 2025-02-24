from sagemaker_inference import model_server
# from handler_service import MyHandlerService
from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_inference.transformer import Transformer
from sagemaker_inference import content_types, decoder, encoder
import torch
import logging
import json
import base64
import io
from PIL import Image
import torchvision.transforms as T
import mlflow

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SimpleInferenceHandler:
    """Example inference handler that just does a dummy pass."""

    def default_model_fn(self, model_dir):
        logger.info(f"Loading model from {model_dir}")
        model = mlflow.pytorch.load_model(model_dir)
        model.eval()
        return model

    def default_input_fn(self, request_body, request_content_type):
        logger.info(f"Received request with content type: {request_content_type}")
        if request_content_type == 'application/json':
            data = json.loads(request_body)
            
            # Log the keys received in the request
            logger.info(f"Received data keys: {data.keys()}")
            
            if "image_data" not in data:
                raise ValueError("JSON request body must include 'image_data' key with base64 encoded image.")
            
            try:
                # Decode base64 string into bytes
                image_bytes = base64.b64decode(data["image_data"])
                
                # Convert bytes to a PIL image
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                # Define the same image transforms used in training
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor()
                ])
                
                # Apply transforms and add batch dimension
                input_tensor = transform(image).unsqueeze(0)
                logger.info("Successfully processed image input")
                return input_tensor
                
            except Exception as e:
                logger.error(f"Error processing input: {str(e)}")
                raise
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")

    def default_predict_fn(self, input_data, model):
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

    def default_output_fn(self, prediction, accept):
        if accept == "application/json":
            return json.dumps(prediction), accept
        else:
            raise ValueError(f"Unsupported accept type: {accept}")


class MyHandlerService(DefaultHandlerService):
    """
    Handler service that is executed by the model server.
    Uses a custom default inference handler above.
    """

    def __init__(self):
        transformer = Transformer(default_inference_handler=SimpleInferenceHandler())
        super(MyHandlerService, self).__init__(transformer=transformer) 

if __name__ == "__main__":
    # Start the model server, specifying our handler service
    model_server.start_model_server(handler_service=MyHandlerService()) 