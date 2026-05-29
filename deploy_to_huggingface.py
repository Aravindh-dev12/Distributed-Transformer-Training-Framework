import os
import sys
import shutil
import tempfile
from huggingface_hub import create_repo, HfApi

def print_banner():
    banner = """
======================================================================
  HUGGING FACE HUB DEPLOYMENT WIZARD | PROFILE: ARAVINDHAN11
======================================================================
"""
    print(banner)

def write_docker_assets(directory_path, space_title):
    """
    Writes a Dockerfile and README.md with spaces metadata for Hugging Face Spaces.
    """
    dockerfile_content = """FROM python:3.12-slim

# Install essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
RUN pip install --no-cache-dir torch transformers datasets rich huggingface_hub

# Copy all project files
COPY . .

# Expose port (HF Spaces defaults to 7860)
EXPOSE 7860
ENV PORT=7860
ENV HOME=/tmp

# Run command center server
CMD ["python", "server.py"]
"""

    readme_content = f"""---
title: {space_title}
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# {space_title}

An advanced and intelligent distributed transformer training and inference framework.

## Key Features
- **Hugging Face Hub Connection**: Convert, load, and push models.
- **Playground**: High-performance streaming chat.
- **Fine-Tuning**: Live loss chart.
- **Parallelism**: 3D Parallelism status.

Developed by **Aravindhan11**.
"""

    with open(os.path.join(directory_path, "Dockerfile"), "w", encoding="utf-8") as f:
        f.write(dockerfile_content)

    with open(os.path.join(directory_path, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)

def deploy_space(token, space_name):
    """
    Deploys the entire framework dashboard as a Docker Space on Hugging Face.
    """
    repo_id = f"Aravindhan11/{space_name}"
    print(f"\nInitializing Hugging Face Space repository: {repo_id}...")
    
    try:
        # Create Hugging Face Space repository
        create_repo(
            repo_id=repo_id,
            token=token,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True
        )
        print("Space repository initialized successfully.")
        
        # Create a temporary directory to bundle files, avoiding copying large .git folders
        print("Bundling project assets for container building...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy source files
            for item in os.listdir(current_dir):
                s = os.path.join(current_dir, item)
                d = os.path.join(temp_dir, item)
                
                # Exclude large cache directories, virtual environments, and checkpoints
                if item in [".git", ".venv", "__pycache__", "exported_checkpoints", "node_modules", ".gemini"]:
                    continue
                    
                if os.path.isdir(s):
                    shutil.copytree(s, d)
                else:
                    shutil.copy(s, d)
            
            # Write Docker assets into bundle
            write_docker_assets(temp_dir, "Distributed LLaMA Framework")
            
            print("Uploading bundled files to Hugging Face Spaces (this might take a minute)...")
            api = HfApi()
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=repo_id,
                repo_type="space",
                token=token,
                commit_message="Deploy Intelligent Distributed LLaMA Framework"
            )
            
            space_url = f"https://huggingface.co/spaces/{repo_id}"
            print("\n[SUCCESS] Your Intelligent Framework is now building and hosting on Hugging Face Spaces!")
            print(f"👉 Access your public dashboard here: {space_url}")
            print("Hugging Face will automatically build the Docker image and deploy it. Enjoy!")
            
    except Exception as e:
        print(f"\n[ERROR] Error during deployment: {e}")

def deploy_model(token, repo_name):
    """
    Converts and deploys a small LLaMA model configuration and pre-trained weights to Hugging Face Hub.
    """
    repo_id = f"Aravindhan11/{repo_name}"
    print(f"\nPreparing to upload pre-trained custom model weights to Model Hub: {repo_id}...")
    
    try:
        from hf_converter import HFWeightConverter
        
        # Download and map a small SmolLM model locally to convert it
        base_model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
        print(f"Downloading pre-trained base model ({base_model_name}) to convert keys...")
        
        model, config, tokenizer = HFWeightConverter.load_and_convert_hf(base_model_name)
        
        # Save custom checkpoints locally
        local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exported_checkpoints", repo_name)
        print(f"Reversing weight names and writing standard config files to {local_dir}...")
        
        HFWeightConverter.save_custom_to_hf_format(
            custom_model=model,
            hf_config=config,
            tokenizer=tokenizer,
            save_directory=local_dir
        )
        
        # Push folder to HF
        repo_url = HFWeightConverter.push_to_huggingface(
            local_dir=local_dir,
            repo_id=repo_id,
            token=token
        )
        
[SUCCESS] Your custom LLaMA model has been converted and hosted on Hugging Face Model Hub!
        print(f"👉 View your model repository here: {repo_url}")
        
    except Exception as e:
        print(f"\n❌ Error during model deployment: {e}")

def main():
    print_banner()
    
    # 1. Ask for credentials (use env var HF_TOKEN if set)
    token = os.getenv("HF_TOKEN")
    if not token:
        token = input("Enter your Hugging Face User Write Token (hf_...): ").strip()
    if not token:
        print("Error: Hugging Face Token is required.")
        sys.exit(1)
    
    # 2. Ask for deployment mode (use env var DEPLOY_MODE if set)
    print("\nWhat would you like to host on Hugging Face?")
    print("  [1] Entire Intelligent Web Dashboard (as an interactive Hugging Face Space)")
    print("  [2] Custom LLaMA Model Config & Weights (as a Hugging Face Model Repository)")
    
    choice = os.getenv("DEPLOY_MODE")
    if not choice:
        choice = input("\nSelect option (1 or 2): ").strip()
    
    if choice == "1":
        space_name = os.getenv("SPACE_NAME")
        if not space_name:
            space_name = input("Enter Space Name (default: Distributed-Transformer-Framework): ").strip()
        if not space_name:
            space_name = "Distributed-Transformer-Framework"
        deploy_space(token, space_name)
    elif choice == "2":
        repo_name = os.getenv("REPO_NAME")
        if not repo_name:
            repo_name = input("Enter Model Repository Name (default: Distributed-Llama-Model): ").strip()
        if not repo_name:
            repo_name = "Distributed-Llama-Model"
        deploy_model(token, repo_name)
    else:
        print("Invalid option selected. Exiting.")

if __name__ == "__main__":
    main()
