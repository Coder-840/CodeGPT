from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from src.token_rnn import TokenRNN

app = Flask(__name__, template_folder="../frontend")

MODEL_PATH = "../model/model_weights.npy"
if not os.path.exists("../model"):
    os.makedirs("../model")

# Initialize token-RNN
ai = TokenRNN()
h = None
if os.path.exists(MODEL_PATH):
    ai.load_model(MODEL_PATH)
    print("Model loaded.")

# Optional: seed corpus
SEED_FILE = "../data/seed.txt"
if os.path.exists(SEED_FILE):
    with open(SEED_FILE,"r",encoding="utf-8") as f:
        lines = f.read().splitlines()
    for line in lines:
        h,_ = ai.online_train(line+"\n", hprev=h)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global h
    user_input = request.json.get("message","")
    h,loss = ai.online_train(user_input+"\n",hprev=h)

    if len(ai.vocab)>0:
        tokens = ai.tokenize(user_input)
        seed_token = tokens[-1] if tokens else np.random.choice(ai.vocab)
        seed_idx = ai.token_to_ix.get(seed_token, np.random.randint(0,ai.vocab_size))
        sample_idx = ai.sample(seed_idx,50,h)
        response = " ".join([ai.ix_to_token[i] for i in sample_idx])
    else:
        response = "..."

    ai.save_model(MODEL_PATH)
    return jsonify({"response": response})
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
