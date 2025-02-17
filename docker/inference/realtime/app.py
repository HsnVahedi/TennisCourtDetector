from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    # Health check route
    return jsonify({'status': 'ok'}), 200

@app.route('/invocations', methods=['POST'])
def predict():
    input_data = request.json
    # Run inference here
    return jsonify({'prediction': 'something'})
