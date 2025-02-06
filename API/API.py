import torch
import GPT
import GPTC
import GPTA
import torch.nn as nn
from flask_cors import CORS 
from flask import Flask, request, jsonify


# Initialize Flask app
app = Flask(__name__)
CORS(app)  

# Model cunfiguration
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # context 
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": True       # Query-key-value bias
}


# Tokenizer
tokenizer = GPT.create_tokenizer()

# Device
device = GPT.get_device()

# Assistant model 
classification_model = GPT.GPTModel(GPT_CONFIG_124M)
classification_model.out_head = torch.nn.Linear(in_features=GPT_CONFIG_124M["emb_dim"], out_features=2)
classification_model.load_state_dict(torch.load("../Models/classifier.pth", map_location=device, weights_only=True))
classification_model.to(device)
classification_model.eval()


# Classifier model 
assistant_model = GPT.GPTModel(GPT_CONFIG_124M)
assistant_model.load_state_dict(torch.load("../Models/Assistant.pth"))
assistant_model.to(device)




@app.route("/")
def home():
    return "Flask server is running!"





# Endpoint for Classification model
@app.route("/ClassificationMsg", methods=["POST"])
def classify():
	try:
		data = request.json
		if "input" not in data:
			return jsonify({"error": "Missing 'input' key in JSON payload"}), 400
		
		input_text = (data["input"])
		with torch.no_grad():
			output_model = GPTC.classify_review(input_text, classification_model, tokenizer, device, max_length=120)

		return jsonify({"response": output_model})
	except Exception as e:
		return jsonify({"error": str(e)}), 500






# Endpoint for Assistant model
@app.route("/AssistantMsg", methods=["POST"])
def predict():
	try:
		data = request.json
		if "instruction" not in data:
			return jsonify({"error": "Missing 'instruction' key in JSON payload"}), 400
		
		entry = { 
			"instruction": data["instruction"],
			"input": data["input"]
		} 

		with torch.no_grad():
			input_text = GPTA.format_input(entry)
			token_ids = GPT.text_generation(
				model=assistant_model,
				idx=GPT.text_to_token_ids(input_text, tokenizer).to(device),
				num_token_generation=256,
				context_size=GPT_CONFIG_124M["context_length"],
				eos_id=50256
			)
			generated_text = GPT.token_ids_to_text(token_ids, tokenizer)
			response_text = (
				generated_text[len(input_text):]
				.replace("### Response:", "")
				.strip()
			)
			output_model = response_text.strip()

		return jsonify({"response": output_model})
	except Exception as e:
		return jsonify({"error": str(e)}), 500




# Run the Flask app
if __name__ == "__main__":
	app.run(host="0.0.0.0", port=4000)
