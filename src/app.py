from flask import Flask, request, jsonify, render_template
import os
from rnn import CharRNN
import numpy as np

app = Flask(__name__, template_folder="../frontend")

MODEL_PATH = "../model/model_weights.npy"
if not os.path.exists("../model"):
    os.makedirs("../model")

# Initialize AI
ai = CharRNN()
h = None
if os.path.exists(MODEL_PATH):
    ai.load_model(MODEL_PATH)
    print("Model loaded.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global h
    user_input = request.json.get("message","")
    h, loss = ai.online_train(user_input+"\n", hprev=h)

    if len(ai.vocab) > 0:
        seed_idx = ai.char_to_ix.get(user_input[-1], np.random.randint(0, ai.vocab_size))
        sample_idx = ai.sample(seed_idx, 150, h)
        response = ''.join([ai.ix_to_char[i] for i in sample_idx])
        response = response.replace('\n',' ')
    else:
        response = "..."

    ai.save_model(MODEL_PATH)
    return jsonify({"response": response})
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
