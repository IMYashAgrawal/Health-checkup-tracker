from flask import Flask, request, jsonify, send_from_directory
import pickle
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'D:\Codes\Projects\MentalHealthChatBot\model.pkl'

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
else:
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please train and save your model first.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.json.get('message', '')
        if not user_input:
            return jsonify({"error": "No message provided."}), 400

        prediction = model.predict([user_input])[0]
        return jsonify({"message": user_input, "prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/web', methods=['GET'])
def serve_web():
    return send_from_directory('.', 'index.html')

@app.route('/', methods=['GET'])
def home():
    return "Mental Health Chatbot API is running!"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
