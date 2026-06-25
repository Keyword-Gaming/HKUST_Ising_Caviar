import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2-large", cache_dir="./cache")
model = GPT2LMHeadModel.from_pretrained("gpt2-large", cache_dir="./cache")

tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def generate_response_smart_stop(prompt, max_new_tokens=100, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    generated_tokens = []
    past_key_values = None
    
    print("--- PROMPT ---")
    print(prompt)
    print("\n--- RESPONSE ---")
    
    for i in range(max_new_tokens):
        with torch.no_grad():
            if past_key_values is not None:
                outputs = model(
                    input_ids=input_ids[:, -1:], 
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            else:
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    use_cache=True
                )
        
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[0, -1, :]
        
        # Repetition Penalty
        for token_id in set(generated_tokens):
            if next_token_logits[token_id] > 0:
                next_token_logits[token_id] /= repetition_penalty
            else:
                next_token_logits[token_id] *= repetition_penalty
        
        # Temperature
        next_token_logits = next_token_logits / temperature
        
        # Top-P Filter
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        next_token_logits[indices_to_remove] = float('-inf')
        
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        
        # 1. Base Stop: End-of-Text token detected
        if next_token_id.item() == tokenizer.eos_token_id:
            break
            
        generated_tokens.append(next_token_id.item())
        
        # Print instantly
        next_token_string = tokenizer.decode(next_token_id)
        print(next_token_string, end="", flush=True)
        
        # 2. SMART STOP: Check if the model just finished a sentence
        # We enforce a minimum length (e.g., 20 tokens) so it doesn't stop immediately
        if len(generated_tokens) > 20 and next_token_string.strip() in [".", "!", "?", '"']:
            break
        
        # Update trackers
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=-1)
        
    print()

# Run the smart stopping version
prompt = "The future of artificial intelligence will be"
generate_response_smart_stop(prompt, max_new_tokens=100, temperature=0.7)
