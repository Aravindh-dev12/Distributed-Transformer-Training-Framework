import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from hf_converter import Llama

class LLMInferenceEngine:
    def __init__(self, model: Llama, tokenizer: AutoTokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        system_prompt: str = ""
    ):
        """
        Generates text token-by-token and yields the decoded tokens in real-time.
        """
        # Formulate Llama chat prompt if we want instruction structure
        formatted_prompt = prompt
        if system_prompt:
            formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        eos_token_id = self.tokenizer.eos_token_id
        
        # Keep track of generation metrics
        start_time = time.time()
        tokens_generated = 0
        first_token_time = None
        
        for _ in range(max_new_tokens):
            # Model forward pass
            # Note: custom Llama handles the entire sequence inside itself
            logits = self.model(input_ids=input_ids) # [batch, seq, vocab]
            next_token_logits = logits[0, -1, :].float() # [vocab]
            
            # Apply temperature
            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float("Inf")
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float("Inf")
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            next_token_id = next_token.item()
            
            if tokens_generated == 0:
                first_token_time = time.time() - start_time
                
            tokens_generated += 1
            
            # Check for EOS
            if next_token_id == eos_token_id:
                break
                
            # Append to inputs
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            # Decode this specific token
            token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            
            yield {
                "token": token_text,
                "metrics": {
                    "first_token_time": f"{first_token_time:.2f}s" if first_token_time else "0.00s",
                    "speed": f"{tokens_generated / (time.time() - start_time):.1f} tok/s",
                    "tokens_count": tokens_generated
                }
            }

    @torch.no_grad()
    def generate_full(self, prompt: str, **kwargs):
        """
        Helper that runs the generator fully and returns the entire generated string.
        """
        output = ""
        last_metrics = None
        for step in self.generate_stream(prompt, **kwargs):
            output += step["token"]
            last_metrics = step["metrics"]
        return output, last_metrics
