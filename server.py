import os
import sys
import json
import time
import threading
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch

from hf_converter import HFWeightConverter
from inference import LLMInferenceEngine
from finetune import LLMTrainer

# Ensure proper paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class GlobalState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.engine = None
        self.model_name = ""
        self.loading_status = "idle"  # idle, loading, success, error
        self.loading_progress = ""
        self.loading_error = ""
        
        # Training state
        self.trainer = None
        self.training_thread = None
        self.training_status = "idle"  # idle, training, finished, stopped, error
        self.training_metrics = []
        self.stop_training_flag = False
        self.train_dataset = ""
        
        # Lock for thread safety
        self.lock = threading.Lock()

global_state = GlobalState()

class IntelligentHubHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        # Allow Cross-Origin Requests for local developer servers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)

        # 1. Static file routing
        if path == "/" or path == "/index.html":
            self.serve_static("web_dashboard/index.html", "text/html")
            return
        elif path == "/style.css":
            self.serve_static("web_dashboard/style.css", "text/css")
            return
        elif path == "/app.js":
            self.serve_static("web_dashboard/app.js", "application/javascript")
            return

        # 2. REST API endpoints
        elif path == "/api/models/list":
            models = [
                {
                    "id": "HuggingFaceTB/SmolLM-135M-Instruct",
                    "name": "SmolLM 135M (Instruct)",
                    "description": "Ultra-lightweight and blazing fast. Ideal for local CPU/GPU testing.",
                    "size": "135M params",
                    "recommended": True
                },
                {
                    "id": "HuggingFaceTB/SmolLM-360M-Instruct",
                    "name": "SmolLM 360M (Instruct)",
                    "description": "Perfect balance between speed, memory footprint, and response quality.",
                    "size": "360M params",
                    "recommended": False
                },
                {
                    "id": "meta-llama/Llama-3.2-1B-Instruct",
                    "name": "LLaMA 3.2 1B (Instruct)",
                    "description": "Highly advanced small model. Requires Hugging Face user credentials/token.",
                    "size": "1.2B params",
                    "recommended": False
                }
            ]
            self.send_json(models)
            return

        elif path == "/api/models/status":
            with global_state.lock:
                status = {
                    "status": global_state.loading_status,
                    "progress": global_state.loading_progress,
                    "error": global_state.loading_error,
                    "loaded_model": global_state.model_name,
                    "specs": {
                        "vocab_size": global_state.config.vocab_size if global_state.config else None,
                        "hidden_size": global_state.config.hidden_size if global_state.config else None,
                        "layers": global_state.config.num_hidden_layers if global_state.config else None,
                        "heads": global_state.config.num_attention_heads if global_state.config else None
                    } if global_state.model else None
                }
            self.send_json(status)
            return

        elif path == "/api/chat":
            # Server-Sent Events (SSE) streaming chat endpoint
            prompt = query.get("prompt", [""])[0]
            temperature = float(query.get("temp", [0.7])[0])
            top_p = float(query.get("top_p", [0.9])[0])
            top_k = int(query.get("top_k", [50])[0])
            max_tokens = int(query.get("max_tokens", [128])[0])
            system_prompt = query.get("system", [""])[0]

            if not global_state.model or not global_state.engine:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No model has been loaded yet."}).encode("utf-8"))
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            print(f"Starting generation stream for prompt: '{prompt[:40]}...'")
            try:
                # Run text generation stream
                stream = global_state.engine.generate_stream(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    system_prompt=system_prompt
                )
                
                for output in stream:
                    data = json.dumps(output)
                    self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
                
                # Signal completion
                self.wfile.write("data: [DONE]\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception as e:
                err_data = json.dumps({"error": str(e)})
                self.wfile.write(f"data: {err_data}\n\n".encode("utf-8"))
                self.wfile.flush()
            return

        elif path == "/api/train/status":
            with global_state.lock:
                status = {
                    "status": global_state.training_status,
                    "dataset": global_state.train_dataset,
                    "metrics": global_state.training_metrics
                }
            self.send_json(status)
            return

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint not found")

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        try:
            body = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            body = {}

        if path == "/api/models/load":
            model_id = body.get("model_name", "")
            if not model_id:
                self.send_error_response("Missing 'model_name' in request body.")
                return

            with global_state.lock:
                if global_state.loading_status == "loading":
                    self.send_error_response("A model is already loading in the background.")
                    return
                global_state.loading_status = "loading"
                global_state.loading_progress = f"Initializing download for {model_id}..."
                global_state.loading_error = ""

            # Launch loading/conversion thread
            threading.Thread(target=self._bg_load_model, args=(model_id,), daemon=True).start()
            self.send_json({"message": "Loading started in background.", "model": model_id})
            return

        elif path == "/api/train/start":
            if not global_state.model:
                self.send_error_response("Please load a model before running fine-tuning.")
                return

            with global_state.lock:
                if global_state.training_status == "training":
                    self.send_error_response("Fine-tuning is already in progress.")
                    return
                global_state.training_status = "training"
                global_state.training_metrics = []
                global_state.stop_training_flag = False
                global_state.train_dataset = body.get("dataset", "custom")

            dataset = body.get("dataset", "")
            lr = float(body.get("lr", 3e-4))
            seq_len = int(body.get("seq_len", 32))
            batch_size = int(body.get("batch_size", 1))
            grad_acc = int(body.get("grad_acc", 1))
            max_steps = int(body.get("max_steps", 50))

            # Initialize trainer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            global_state.trainer = LLMTrainer(
                model=global_state.model,
                tokenizer=global_state.tokenizer,
                device=device,
                learning_rate=lr,
                seq_len=seq_len,
                batch_size=batch_size,
                gradient_accumulation_steps=grad_acc
            )

            # Start background thread for fine-tuning
            threading.Thread(
                target=self._bg_train_model, 
                args=(dataset, max_steps), 
                daemon=True
            ).start()
            
            self.send_json({"message": "Fine-tuning launched successfully!"})
            return

        elif path == "/api/train/stop":
            with global_state.lock:
                if global_state.training_status != "training":
                    self.send_error_response("Training is not currently running.")
                    return
                global_state.stop_training_flag = True
            
            self.send_json({"message": "Stop signal sent to trainer."})
            return

        elif path == "/api/export/huggingface":
            if not global_state.model:
                self.send_error_response("No loaded model to export.")
                return

            repo_id = body.get("repo_id", "")
            token = body.get("token", "")

            if not repo_id or not token:
                self.send_error_response("Hugging Face Repository ID and Write Token are required.")
                return

            # Run in a background thread to prevent UI lockup
            threading.Thread(
                target=self._bg_export_model,
                args=(repo_id, token),
                daemon=True
            ).start()

            self.send_json({"message": "Export initiated. Pushing files to Hugging Face..."})
            return

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Endpoint not found")

    # --- Background thread worker functions ---
    def _bg_load_model(self, model_id):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            custom_model, config, tokenizer = HFWeightConverter.load_and_convert_hf(
                model_name=model_id,
                device=device
            )
            engine = LLMInferenceEngine(custom_model, tokenizer, device=device)
            
            with global_state.lock:
                global_state.model = custom_model
                global_state.tokenizer = tokenizer
                global_state.config = config
                global_state.engine = engine
                global_state.model_name = model_id
                global_state.loading_status = "success"
                global_state.loading_progress = f"Successfully loaded {model_id} on {device.upper()}."
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            with global_state.lock:
                global_state.loading_status = "error"
                global_state.loading_error = str(e)
                global_state.loading_progress = "Model loading failed."

    def _bg_train_model(self, dataset, max_steps):
        try:
            trainer = global_state.trainer
            generator = trainer.fit_generator(dataset_source=dataset, max_steps=max_steps)
            
            for metrics in generator:
                # Check for stop flag
                with global_state.lock:
                    if global_state.stop_training_flag:
                        global_state.training_status = "stopped"
                        break
                    global_state.training_metrics.append(metrics)
                time.sleep(0.01) # Yield execution briefly
                
            with global_state.lock:
                if global_state.training_status == "training":
                    global_state.training_status = "finished"
                    
        except Exception as e:
            with global_state.lock:
                global_state.training_status = "error"
                global_state.training_metrics.append({"status": "error", "message": str(e)})

    def _bg_export_model(self, repo_id, token):
        try:
            with global_state.lock:
                model = global_state.model
                config = global_state.config
                tokenizer = global_state.tokenizer
                model_name = global_state.model_name
                
            local_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exported_checkpoints", repo_id.split("/")[-1])
            
            # 1. Convert our custom state_dict back to Hugging Face Llama structure and save locally
            HFWeightConverter.save_custom_to_hf_format(
                custom_model=model,
                hf_config=config,
                tokenizer=tokenizer,
                save_directory=local_save_dir
            )
            
            # 2. Push directory to user's profile on Hugging Face Hub
            HFWeightConverter.push_to_huggingface(
                local_dir=local_save_dir,
                repo_id=repo_id,
                token=token
            )
            
            # Success logging in terminal
            print(f"Export Completed! Model successfully hosted at Hugging Face under: {repo_id}")
            
        except Exception as e:
            print(f"Export Failed! Error: {e}")

    # --- Helper methods ---
    def serve_static(self, file_path, content_type):
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
        if not os.path.exists(full_path):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Static file not found")
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        
        # Read the file
        with open(full_path, "rb") as f:
            content = f.read()
            
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def send_error_response(self, message):
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        response_bytes = json.dumps({"error": message}).encode("utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, IntelligentHubHandler)
    print(f"=== Intelligent Framework Hub Server launched on http://localhost:{port} ===")
    print("Open this URL in your browser to access the beautiful control room dashboard!")
    httpd.serve_forever()

if __name__ == "__main__":
    port = 8000
    # Check for PORT env var (important for Hugging Face Spaces)
    if "PORT" in os.environ:
        try:
            port = int(os.environ["PORT"])
        except ValueError:
            pass
    elif len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
