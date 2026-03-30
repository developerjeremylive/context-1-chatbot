from flask import Flask, render_template, request, jsonify
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import threading
import time

app = Flask(__name__)

# Configuración
MODEL_NAME = os.environ.get("MODEL_NAME", "chromadb/context-1")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "2048"))

# Cargar modelo y tokenizer
print(f"Cargando modelo {MODEL_NAME} en {DEVICE}...")
tokenizer = None
model = None
model_loaded = False

def load_model():
    global tokenizer, model, model_loaded
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map=DEVICE,
            trust_remote_code=True
        )
        model_loaded = True
        print(f"Modelo {MODEL_NAME} cargado exitosamente!")
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        model_loaded = False

# Cargar en hilo separado para no bloquear el startup
load_thread = threading.Thread(target=load_model)
load_thread.start()

@app.route("/")
def index():
    return render_template("index.html", model_name=MODEL_NAME, loaded=model_loaded)

@app.route("/chat", methods=["POST"])
def chat():
    if not model_loaded:
        return jsonify({"error": "Modelo aún cargando..."}), 503
    
    data = request.json
    user_message = data.get("message", "")
    history = data.get("history", [])
    
    if not user_message:
        return jsonify({"error": "Mensaje vacío"}), 400
    
    # Construir prompt
    messages = [{"role": "user", "content": user_message}]
    
    # Generar
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        outputs = model.generate(
            inputs,
            generation_config=generation_config,
        )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        return jsonify({
            "response": response.strip(),
            "model": MODEL_NAME
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/status")
def status():
    return jsonify({
        "loaded": model_loaded,
        "model": MODEL_NAME,
        "device": DEVICE
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
