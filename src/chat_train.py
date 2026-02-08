# File: src/chat_train.py
import os
import numpy as np
from rnn import CharRNN

MODEL_PATH = "../model/model_weights.npy"
if not os.path.exists("../model"):
    os.makedirs("../model")

# Initialize AI
ai = CharRNN()
h = None

# Load model if exists
if os.path.exists(MODEL_PATH):
    ai.load_model(MODEL_PATH)
    print("Model loaded.")

print("Scratch Coding AI Chat. Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Train online with user input
    h, loss = ai.online_train(user_input+"\n", hprev=h)

    # Generate response
    if len(ai.vocab) > 0:
        seed_idx = ai.char_to_ix[user_input[-1]] if user_input[-1] in ai.char_to_ix else np.random.randint(0,ai.vocab_size)
        sample_idx = ai.sample(seed_idx, 150, h)
        response = ''.join([ai.ix_to_char[i] for i in sample_idx])
        print("AI:", response.replace('\n',' '))
    else:
        print("AI: ...")

    # Save model after each input
    ai.save_model(MODEL_PATH)
