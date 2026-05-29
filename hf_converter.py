import os
import sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import create_repo, upload_folder

# Add project directories to path to ensure proper imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sys import path
path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "3d_parallel", "step1_modelling"))

from model import Llama

class HFWeightConverter:
    @staticmethod
    def get_key_mapping(num_layers):
        """
        Creates a mapping of parameter names from Hugging Face LlamaForCausalLM/SmolLMForCausalLM
        to our custom Llama implementation.
        """
        mapping = {
            "model.embed_tokens.weight": "embedding.weight",
            "model.norm.weight": "final_norm.weight",
            "lm_head.weight": "final_proj.weight"
        }
        for i in range(num_layers):
            prefix_hf = f"model.layers.{i}"
            prefix_custom = f"decoder_layers.{i}"
            
            mapping[f"{prefix_hf}.input_layernorm.weight"] = f"{prefix_custom}.input_layernorm.weight"
            mapping[f"{prefix_hf}.post_attention_layernorm.weight"] = f"{prefix_custom}.post_attention_layernorm.weight"
            
            mapping[f"{prefix_hf}.self_attn.q_proj.weight"] = f"{prefix_custom}.attention.q_proj.weight"
            mapping[f"{prefix_hf}.self_attn.k_proj.weight"] = f"{prefix_custom}.attention.k_proj.weight"
            mapping[f"{prefix_hf}.self_attn.v_proj.weight"] = f"{prefix_custom}.attention.v_proj.weight"
            mapping[f"{prefix_hf}.self_attn.o_proj.weight"] = f"{prefix_custom}.attention.out_proj.weight"
            
            mapping[f"{prefix_hf}.mlp.gate_proj.weight"] = f"{prefix_custom}.mlp.gate_proj.weight"
            mapping[f"{prefix_hf}.mlp.up_proj.weight"] = f"{prefix_custom}.mlp.up_proj.weight"
            mapping[f"{prefix_hf}.mlp.down_proj.weight"] = f"{prefix_custom}.mlp.down_proj.weight"
            
        return mapping

    @classmethod
    def load_and_convert_hf(cls, model_name: str, device="cpu") -> tuple[Llama, AutoConfig, AutoTokenizer]:
        """
        Downloads a Llama/SmolLM model from Hugging Face, maps its weights to our custom Llama model.
        """
        print(f"Fetching config and tokenizer for {model_name}...")
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Make sure mandatory properties are in config
        if not hasattr(config, "rope_theta"):
            config.rope_theta = 10000.0
            
        print(f"Downloading pre-trained weights from Hugging Face...")
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        hf_state_dict = hf_model.state_dict()
        
        print(f"Initializing custom Llama framework model...")
        custom_model = Llama(config)
        custom_state_dict = custom_model.state_dict()
        
        print("Mapping state dict weights...")
        key_map = cls.get_key_mapping(config.num_hidden_layers)
        
        mapped_state_dict = {}
        for hf_key, custom_key in key_map.items():
            if hf_key in hf_state_dict:
                mapped_state_dict[custom_key] = hf_state_dict[hf_key]
            else:
                print(f"Warning: Expected key {hf_key} not found in HF state dict!")
                
        # Fill any remaining keys from custom model defaults
        for k in custom_state_dict.keys():
            if k not in mapped_state_dict:
                mapped_state_dict[k] = custom_state_dict[k]
                
        custom_model.load_state_dict(mapped_state_dict)
        custom_model.to(device)
        print("Model weight mapping completed successfully!")
        
        # Clean up memory
        del hf_model
        del hf_state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return custom_model, config, tokenizer

    @classmethod
    def save_custom_to_hf_format(cls, custom_model: Llama, hf_config: AutoConfig, tokenizer: AutoTokenizer, save_directory: str):
        """
        Converts custom Llama model state_dict back to Hugging Face format and saves it.
        """
        os.makedirs(save_directory, exist_ok=True)
        print(f"Initializing a native Hugging Face model template...")
        
        # Suppress weight initialization warning since we will overwrite all of them
        hf_model = AutoModelForCausalLM.from_config(hf_config)
        
        custom_state_dict = custom_model.state_dict()
        key_map = cls.get_key_mapping(hf_config.num_hidden_layers)
        reverse_map = {v: k for k, v in key_map.items()}
        
        hf_state_dict = {}
        for custom_key, val in custom_state_dict.items():
            if custom_key in reverse_map:
                hf_key = reverse_map[custom_key]
                hf_state_dict[hf_key] = val
            else:
                # E.g., rotary embeddings, which are not saved in HF state dict
                pass
                
        hf_model.load_state_dict(hf_state_dict, strict=False)
        
        print(f"Saving standard Hugging Face model and tokenizer to {save_directory}...")
        hf_model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print("Save completed successfully!")

    @staticmethod
    def push_to_huggingface(local_dir: str, repo_id: str, token: str) -> str:
        """
        Pushes a saved Hugging Face model directory directly to Hugging Face Hub.
        """
        print(f"Connecting to Hugging Face to push model to {repo_id}...")
        
        # Create repo if not exist
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=False,
            exist_ok=True
        )
        
        # Generate custom Model Card README.md
        readme_path = os.path.join(local_dir, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(f"""---
language: en
license: mit
tags:
- llama
- text-generation
- custom-framework
---

# {repo_id.split('/')[-1]}

This model was trained or fine-tuned using the **Distributed-Transformer-Training-Framework**, an advanced and intelligent distributed training system.

## Model Description
- **Architecture**: LLaMA-style custom transformer
- **Base Model**: Loaded via pre-trained open-source parameters and mapped directly to custom modeling structures.
- **Framework capabilities**: Hand-rolled rotary positional embeddings, grouped query attention, data/tensor/pipeline parallelism.

## How to use
You can load this model directly using Hugging Face `transformers`:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

inputs = tokenizer("Hello, I am a custom LLaMA model", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Developed and uploaded using the **Intelligent Framework Command Center** of the Distributed-Transformer-Training-Framework.
""")

        print("Uploading directory contents to Hugging Face Hub...")
        upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            token=token,
            commit_message="Upload fine-tuned model from Distributed-Transformer-Training-Framework"
        )
        
        print(f"Model pushed successfully! Available at: {repo_url}")
        return repo_url
