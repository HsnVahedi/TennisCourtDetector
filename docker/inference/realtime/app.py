from flask import Flask, request, jsonify
from handler_service import SimpleInferenceHandler
import os

app = Flask(__name__)
handler = SimpleInferenceHandler()

# Load the model at startup
model_dir = os.environ.get('SAGEMAKER_MODEL_DIR', '/opt/ml/model')
model = handler.default_model_fn(model_dir)

@app.route('/ping', methods=['GET'])
def ping():
    # Health check route
    return jsonify({'status': 'ok'}), 200

@app.route('/invocations', methods=['POST'])
def predict():
    if request.content_type != 'application/json':
        return jsonify({'error': 'Content type must be application/json'}), 415
    
    try:
        # Process the input using the handler
        input_data = handler.default_input_fn(request.get_data().decode('utf-8'), request.content_type)
        
        # Run prediction
        prediction = handler.default_predict_fn(input_data, model)
        
        # Format the output
        response, content_type = handler.default_output_fn(prediction, 'application/json')
        
        return response, 200, {'Content-Type': content_type}
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)
