# Context-1 Chatbot

Web UI para el modelo chromadb/context-1 de HuggingFace.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python app.py
```

Luego abre http://localhost:5000 en tu navegador.

## Configuración

- `MODEL_NAME`: Nombre del modelo en HuggingFace (default: chromadb/context-1)
- `DEVICE`: cpu, cuda, o mps (default: auto)
- `MAX_LENGTH`: Longitud máxima de respuesta (default: 2048)
