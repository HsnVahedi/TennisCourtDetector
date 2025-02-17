from sagemaker_inference import model_server
from handler_service import MyHandlerService

if __name__ == "__main__":
    # Start the model server, specifying our handler service
    model_server.start_model_server(handler_service=MyHandlerService()) 