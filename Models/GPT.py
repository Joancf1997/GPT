import os
import torch
import tiktoken
import numpy as np
import urllib.request
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib.ticker import MaxNLocator


class MultiHeadAttention(nn.Module):
	def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
		super().__init__() 
		assert (d_out % num_heads == 0),  "Out dimension must be divisible by the number of heads"

		self.d_out = d_out
		self.num_heads = num_heads
		self.head_dim = d_out // num_heads
		self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.out_proj = nn.Linear(d_out, d_out)
		self.dropout = nn.Dropout(dropout)
		self.register_buffer(
			"mask",
			torch.triu(torch.ones(context_length, context_length), diagonal=1)
		)


	def forward(self, x):
		b, num_tokens, _ = x.shape 	# Shape: (b, num_tokens, d_out)

		keys = self.W_key(x)
		queries = self.W_query(x)
		values = self.W_value(x)

		# Split the matrix on the heads 
		keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
		values = values.view(b, num_tokens, self.num_heads, self.head_dim)
		queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

		# Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
		keys = keys.transpose(1, 2)
		queries = queries.transpose(1, 2)
		values = values.transpose(1, 2)

		# Attention scores 
		attn_scores = queries @ keys.transpose(2, 3)

		# Mask 
		mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
		attn_scores.masked_fill_(mask_bool, -torch.inf)

		# Attention weights
		attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
		attn_weights = self.dropout(attn_weights)

		# Context
		context = (attn_weights @ values).transpose(1, 2) 

		# Combine all the heads 
		context = context.contiguous().view(b, num_tokens, self.d_out)
		context = self.out_proj(context)
		return context




class TransformerBlock(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		# Components of the transformer block 
		self.att = MultiHeadAttention(
			d_in=cfg["emb_dim"],
			d_out=cfg["emb_dim"],
			context_length=cfg["context_length"],
			num_heads=cfg["n_heads"], 
			dropout=cfg["drop_rate"],
			qkv_bias=cfg["qkv_bias"]
		)

		self.ff = FeedForward(cfg["emb_dim"])
		self.norm1 = LayerNorm(cfg["emb_dim"])
		self.norm2 = LayerNorm(cfg["emb_dim"])
		self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

	# Data flow inside the transformer block 
	def forward(self, x):
		shortcut = x
		x = self.norm1(x)
		x = self.att(x)
		x = self.drop_shortcut(x)
		x = x + shortcut 

		shortcut = x 
		x = self.norm2(x)
		x = self.ff(x)
		x = x + shortcut 
		return x




class GELU(nn.Module): 
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return 0.5 * x * (1 + torch.tanh(
			torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
			(x + 0.044715 * torch.pow(x, 3))
		))




class FeedForward(nn.Module):
	def __init__(self, emb_dim):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(emb_dim, 4 * emb_dim),
			GELU(),
			nn.Linear(4*emb_dim, emb_dim)
		)

	def forward(self, x):
		return self.layers(x)





class LayerNorm(nn.Module):
	def __init__(self, emb_dim):
		super().__init__()
		self.eps = 1e-5								# Done to avoid division by 0
		self.scale = nn.Parameter(torch.ones(emb_dim))
		self.shift = nn.Parameter(torch.ones(emb_dim))

	def forward(self, x):
		mean = x.mean(dim=-1, keepdim=True)
		var = x.var(dim=-1, keepdim=True, unbiased=False)
		norm_x = (x - mean) / torch.sqrt(var + self.eps)
		return self.scale * norm_x + self.shift




class GPTModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    # Word embedding
    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
    self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
    # Regularizaion
    self.drop_emb = nn.Dropout(cfg["drop_rate"])
    # Transformer
    self.trf_blocks = nn.Sequential(
      *[TransformerBlock(cfg) for I in range(cfg["n_layers"])]
    )
    # Layer normalization 
    self.final_norm = LayerNorm(cfg["emb_dim"])
    # UnEmbedding
    self.out_head = nn.Linear(
      cfg["emb_dim"], cfg["vocab_size"], bias=False
    )

  def forward(self, in_idx):
    batch_size, seq_len = in_idx.shape
    # Process the embeddings 
    tok_embeds = self.tok_emb(in_idx)     # Work embedding 
    # The positional, if the seq_len is smaller than the context_length, we use the seq_len.. 
    pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
    x = tok_embeds + pos_embeds
    # Regularization
    x = self.drop_emb(x)
    # Transformer blocks 
    x = self.trf_blocks(x)
    # MLP
    x = self.final_norm(x)
    # Logits for the next token prediction
    logits = self.out_head(x)
    return logits





class GPTDataset(Dataset):
	def __init__(self, txt, tokenizer, max_length, stride):
		self.input_ids = []
		self.target_ids = []
		# tokenize text 
		token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
		# Encode the chuncks giving jumps of stride size 
		for i in range(0, len(token_ids) - max_length, stride):
			input_chunk = token_ids[i: i+max_length]
			target_chunk = token_ids[i+1: i+max_length+1]
			self.input_ids.append(torch.tensor(input_chunk))
			self.target_ids.append(torch.tensor(target_chunk)) 

	# Number of training examples of the current text
	def __len__(self):
		return len(self.input_ids)

	# The requestes chunk from the text both inputs and targets
	def __getitem__(self, idx):
		return self.input_ids[idx], self.target_ids[idx]




"""
Downloads the data sample 'the-veredict', use as a small training dataset 
"""
def download_text_sample():
	file_path = "the-verdict.txt"
	url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

	if not os.path.exists(file_path):
		with urllib.request.urlopen(url) as response:
			text_data = response.read().decode('utf-8')
		with open(file_path, "w", encoding="utf-8") as file:
			file.write(text_data)
	else:
		with open(file_path, "r", encoding="utf-8") as file:
			text_data = file.read()
	return text_data



"""
  train_test_split
    Split the text into a trining and validation set.
"""
def train_test_split(text_data, train_ratio=0.9):
  split_idx = int(train_ratio * len(text_data))
  train_data = text_data[:split_idx]
  val_data = text_data[split_idx:]
  return train_data, val_data




def create_data_loader(txt, batch_size=6, max_length=256, stride=256, shuffle=True, drop_last=True, num_workers=0):
	# Tokenizer use on GPT2
  tokenizer = tiktoken.get_encoding("gpt2")
	# From text to dataloader
  dataset = GPTDataset(txt, tokenizer, max_length, stride)
  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers
  )
  return dataloader




def create_tokenizer():
  return tiktoken.get_encoding("gpt2")





def text_generation(model, idx, num_token_generation, context_size, temperature=0.0, top_k=None, eos_id=None):
	for _ in range(num_token_generation):
		idx_cond = idx[:, -context_size:]
		with torch.no_grad():
			# Generate the next tokens
			logits = model(idx_cond)
		logits = logits[:, -1, :]
		# Only take into account the top K words on the next word selection
		if top_k is not None:
			top_logits, _ = torch.topk(logits, top_k)
			min_val = top_logits[:, -1]
			logits = torch.where(
				logits < min_val,
				torch.tensor(float('-inf')).to(logits.device),
				logits
			)
		# Modify the final distribution with the temp
		if temperature > 0.0:
			logits = logits / temperature
			probs = torch.softmax(logits, dim=-1)
			idx_next = torch.multinomial(probs, num_samples=1)
		else: 
			idx_next = torch.argmax(logits, dim=-1, keepdim=True)
		if idx_next == eos_id:
			break
		idx = torch.cat((idx, idx_next), dim=1)
	return idx



#  ===== Text Manipulation =====
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor



def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())



# ================================================== Training ==================================================
def calc_loss_batch(input_batch, target_batch, model, device):
	input_batch, target_batch = input_batch.to(device), target_batch.to(device)
	logits = model(input_batch)
	loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
	return loss



def calc_loss_loader(data_loader, model, device, num_batches=None):
	total_loss = 0.
	if(len(data_loader) == 0):
		return float("nan")
	elif num_batches is None: 
		num_batches = len(data_loader)
	else:
		num_batches = min(num_batches, len(data_loader))
	for i, (input_batch, target_batch) in enumerate(data_loader):
		if i < num_batches:
			loss = calc_loss_batch(input_batch, target_batch, model, device)
			total_loss += loss.item()
		else:
			break
	return total_loss / num_batches



def evaluate_model(model, train_loader, val_loader, device, eval_iter):
	model.eval()
	with torch.no_grad():
		train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
		val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
	model.train()
	return train_loss, val_loss



def generate_and_print_sample(model, tokenizer, device, start_context):
	model.eval()
	context_size = model.pos_emb.weight.shape[0]
	encoded = text_to_token_ids(start_context, tokenizer).to(device)
	with torch.no_grad():
		token_ids = text_generation(
				model=model, 
				idx=encoded,
				num_token_generation=40, 
				context_size=context_size,
				top_k=40,
				temperature=1
		)
	decoded_text = token_ids_to_text(token_ids, tokenizer)
	print("Text Generation Sample")
	print(decoded_text.replace("\n", " "))  
	model.train()



def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
	train_losses, val_losses, track_tokens_seen = [], [], []
	tokens_seen, global_step = 0, -1

	# Main training loop
	for epoch in range(num_epochs):
		for input_batch, target_batch in train_loader:
			optimizer.zero_grad() # Reset loss gradients from previous batch iteration
			loss = calc_loss_batch(input_batch, target_batch, model, device)
			loss.backward() # Calculate loss gradients
			optimizer.step() # Update model weights using loss gradients
			tokens_seen += input_batch.numel()
			global_step += 1

			if global_step % eval_freq == 0:
				train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
				train_losses.append(train_loss)
				val_losses.append(val_loss)
				track_tokens_seen.append(tokens_seen)
				print(f"Ep {epoch+1} (Step {global_step:06d}): "
              f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
		
		# Generate a sample text for each epoch
		generate_and_print_sample(model, tokenizer, device, start_context)
		
	return train_losses, val_losses, track_tokens_seen





def get_device():
	# Apple silicon
	if torch.cuda.is_available():
		device = torch.device("cuda")
	elif torch.backends.mps.is_available():
		device = torch.device("mps")
	else:
		device = torch.device("cpu")
	return device




"""
  Load weights of a already pre-trained model into the current architecture 
  of this GPT library 
"""
def load_weights_into_gpt(gpt, params):          
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):   
        q_w, k_w, v_w = np.split(                           
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])



def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                          "Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))




def plot_values(epochs_seen, examples_seen, trainin_values, val_values, label="loss"):
	fig, ax1 = plt.subplots(figsize=(5, 3))
	
	# Plot training and validation loss against epochs
	ax1.plot(epochs_seen, trainin_values, label=f"Training {label}")
	ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
	ax1.set_xlabel("Epochs")
	ax1.set_ylabel(label.capitalize())
	ax1.legend()

	# Create a second x-axis for examples seen
	ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
	ax2.plot(examples_seen, trainin_values, alpha=0)  # Invisible plot for aligning ticks
	ax2.set_xlabel("Examples seen")

	fig.tight_layout()  # Adjust layout to make room
	plt.savefig(f"{label}-plot.pdf")
	plt.show()


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()

