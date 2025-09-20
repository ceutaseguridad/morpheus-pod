# worker_server.py (Versión 22.0 - Generación de Datasets)
import logging
import json
import os
import uuid
import threading
import torch
from transformers import pipeline, AutoTokenizer
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from urllib import request, parse
import websocket
import random
import sys
import shutil
import subprocess
import numpy as np

# --- Importaciones existentes ---
from morpheus_lib import management_handler

# --- [VERITAS UPGRADE] Importaciones para análisis visual y lógica facial mejorada ---
try:
    from transformers import BlipForConditionalGeneration, BlipProcessor
    from PIL import Image, ImageDraw
    VISUAL_ANALYSIS_AVAILABLE = True
except ImportError:
    VISUAL_ANALYSIS_AVAILABLE = False

try:
    import face_alignment
    from skimage import io
    FACE_ALIGNMENT_AVAILABLE = True
except ImportError:
    FACE_ALIGNMENT_AVAILABLE = False

try:
    import cv2
    VIDEO_ANALYSIS_AVAILABLE = True
except ImportError:
    VIDEO_ANALYSIS_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger("MorpheusPodServer")

# --- Modelos de IA ---
llm_pipeline = None
vlm_processor = None
vlm_model = None
fa = None # Para face-alignment

MORPHEUS_AI_MODEL_ID = "NousResearch/Hermes-2-Pro-Llama-3-8B"
VISUAL_MODEL_ID = "Salesforce/blip-image-captioning-base"

# --- Constantes y Base de Datos en Memoria ---
COMFYUI_URL = "http://127.0.0.1:8189"
SERVER_ADDRESS = COMFYUI_URL.split("//")[1]
MORPHEUS_LIB_DIR = "/workspace/morpheus_lib"
job_status_db: Dict[str, Dict[str, Any]] = {}

# --- Lógica de Prompts del Sistema ---
MORPHEUS_SYSTEM_PROMPT = ""
try:
    with open(os.path.join(os.path.dirname(__file__), 'morpheus_system_prompt.txt'), 'r', encoding='utf-8') as f:
        MORPHEUS_SYSTEM_PROMPT = f.read().strip()
except Exception as e:
    logger.critical(f"¡CRÍTICO! No se pudo leer el archivo de prompt: {e}.", exc_info=True)
    MORPHEUS_SYSTEM_PROMPT = "Eres un asistente de IA servicial."


app = FastAPI(title="Morpheus AI Pod (Veritas)", version="22.0")

@app.on_event("startup")
async def startup_event():
    # ... (sin cambios en esta sección)
    threading.Thread(target=initialize_llm_background, daemon=True).start()
    if VISUAL_ANALYSIS_AVAILABLE: threading.Thread(target=initialize_vlm_background, daemon=True).start()
    if FACE_ALIGNMENT_AVAILABLE: threading.Thread(target=initialize_face_alignment_background, daemon=True).start()

def initialize_llm_background():
    # ... (sin cambios en esta sección)
    global llm_pipeline
    try:
        if torch.cuda.is_available():
            tokenizer = AutoTokenizer.from_pretrained(MORPHEUS_AI_MODEL_ID)
            llm_pipeline = pipeline("text-generation", model=MORPHEUS_AI_MODEL_ID, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto")
            logger.info("Modelo de IA cargado.")
    except Exception as e:
        logger.critical(f"FALLO CRÍTICO al cargar el modelo de IA: {e}", exc_info=True)

def initialize_vlm_background():
    # ... (sin cambios en esta sección)
    global vlm_processor, vlm_model
    try:
        if torch.cuda.is_available():
            device = "cuda"
            vlm_processor = BlipProcessor.from_pretrained(VISUAL_MODEL_ID)
            vlm_model = BlipForConditionalGeneration.from_pretrained(VISUAL_MODEL_ID).to(device)
            logger.info("Modelo visual cargado.")
    except Exception as e:
        logger.critical(f"FALLO CRÍTICO al cargar el modelo visual: {e}", exc_info=True)

def initialize_face_alignment_background():
    # ... (sin cambios en esta sección)
    global fa
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
        logger.info("Modelo de alineación facial cargado.")
    except Exception as e:
        logger.critical(f"FALLO CRÍTICO al cargar el modelo de alineación facial: {e}", exc_info=True)

# --- Modelos Pydantic y funciones de API (sin cambios) ---
class ChatMessage(BaseModel): role: str; content: str
class ChatPayload(BaseModel): messages: List[ChatMessage]; context: Dict[str, Any]
class ActionData(BaseModel): # ... (sin cambios)
    details: Optional[str] = None; info_type: Optional[str] = None; file_type: Optional[str] = None
    job_id: Optional[str] = None; label: Optional[str] = None; options: Optional[List[str]] = None
    file_types: Optional[List[str]] = None; key: Optional[str] = None
    job_name: Optional[str] = None; job_type: Optional[str] = None; workflow: Optional[str] = None
    config_payload: Optional[Dict[str, Any]] = None
    name: Optional[str] = None; system_prompt: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None; tags: Optional[List[str]] = None
    execution_plan: Optional[List[Dict[str, Any]]] = None
class ChatResponse(BaseModel): response_text: str; action: str; action_data: ActionData; context: Optional[Dict[str, Any]]
class JobPayload(BaseModel): workflow: str; worker_job_id: Optional[str] = None; config_payload: Dict[str, Any] = {}
class StatusResponse(BaseModel): id: str; status: str; output: Optional[Dict[str, Any]] = None; error: Optional[str] = None; progress: int = 0; previews: Optional[List[str]] = None

# --- Funciones de análisis, chat y API (sin cambios) ---
def _analyze_and_tag_image(image_path: str, config_payload: Dict[str, Any]) -> Dict[str, Any]: # ... (sin cambios)
    return {}
def get_morpheus_response(messages: List[ChatMessage], context: Dict[str, Any]) -> Dict[str, Any]: # ... (sin cambios)
    return {}
@app.post("/chat", response_model=ChatResponse)
async def handle_chat(payload: ChatPayload): # ... (sin cambios)
    return ChatResponse(response_text="", action="", action_data={}, context={})
# ... otros endpoints de API sin cambios ...
@app.get("/")
def read_root(): return {"Morpheus Pod (Músculo y Conciencia - Veritas)": "Online"}
@app.get("/health")
def health_check(): return Response(status_code=200)
# ... endpoints de modelos (/models/...) sin cambios ...

# --- Funciones de ComfyUI (sin cambios) ---
def queue_prompt(prompt: Dict[str, Any], client_id: str): # ... (sin cambios)
    p = {"prompt": prompt, "client_id": client_id}; data = json.dumps(p).encode('utf-8'); req = request.Request(f"{COMFYUI_URL}/prompt", data=data); return json.loads(request.urlopen(req).read())
def get_image(filename, subfolder, folder_type): # ... (sin cambios)
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}; url_values = parse.urlencode(data); with request.urlopen(f"{COMFYUI_URL}/view?{url_values}") as response: return response.read()
def get_history(prompt_id): # ... (sin cambios)
    with request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as response: return json.loads(response.read())
def update_workflow_with_payload(workflow_data: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]: # ... (sin cambios)
    updated_workflow = json.loads(json.dumps(workflow_data)); # ... lógica de reemplazo
    return updated_workflow
def run_comfyui_generation(workflow_json: Dict, output_dir: str, client_id: str) -> List[str]: # ... (sin cambios)
    ws = websocket.WebSocket(); ws.connect(f"ws://{SERVER_ADDRESS}/ws?clientId={client_id}"); prompt_id = queue_prompt(workflow_json, client_id)['prompt_id']; output_files = []; # ... lógica de websocket
    return output_files


# --- Hilos de Ejecución de Trabajos ---

def run_job_thread(client_id: str, workflow_name: str, config_payload: Dict[str, Any]):
    # ... (sin cambios en esta función)
    job_dir = f"/workspace/job_data/{client_id}/output"; os.makedirs(job_dir, exist_ok=True); # ... lógica existente

# --- [NUEVA FUNCIÓN] Hilo para la generación de datasets ---
def run_dataset_generation_job_thread(client_id: str, config_payload: Dict[str, Any]):
    """
    Ejecuta el workflow 'generate_dataset' en un bucle para crear múltiples imágenes de dataset.
    """
    job_dir = f"/workspace/job_data/{client_id}/output"
    # El nombre del directorio de salida se toma del payload para coherencia
    output_folder_name = config_payload.get("output_folder_name", f"dataset_{client_id}")
    dataset_output_dir = os.path.join(job_dir, output_folder_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    try:
        job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 5, "output": None, "error": None}
        
        workflow_path = os.path.join(MORPHEUS_LIB_DIR, "workflows", "generate_dataset.json")
        with open(workflow_path, 'r') as f:
            workflow_data = json.load(f)

        dataset_size = config_payload.get("dataset_size", 5)
        all_generated_files = []

        # Bucle para generar el número deseado de imágenes
        for i in range(dataset_size):
            logger.info(f"[Job ID: {client_id}] Generando imagen de dataset {i+1}/{dataset_size}...")
            
            # Crear una copia del payload para esta iteración
            iter_payload = config_payload.copy()
            # Asignar una semilla aleatoria para cada imagen para asegurar la variedad
            iter_payload['seed'] = random.randint(0, 1_000_000_000)
            
            updated_workflow = update_workflow_with_payload(workflow_data, iter_payload)
            
            # Ejecutar la generación para una imagen
            # Pasamos un sub-cliente_id para evitar colisiones en ComfyUI
            iter_client_id = f"{client_id}_iter_{i}"
            output_files = run_comfyui_generation(updated_workflow, dataset_output_dir, iter_client_id)
            
            if output_files:
                all_generated_files.extend(output_files)

            # Actualizar el progreso global
            progress = 5 + int(90 * (i + 1) / dataset_size)
            job_status_db[client_id]["progress"] = progress

        if not all_generated_files:
            raise RuntimeError("La generación del dataset no produjo ninguna imagen.")

        # El resultado del trabajo es la RUTA al directorio que contiene todas las imágenes
        final_output = {"dataset_path": dataset_output_dir}
        job_status_db[client_id] = {"status": "COMPLETED", "output": final_output, "error": None, "progress": 100}

    except Exception as e:
        logger.error(f"Fallo en run_dataset_generation_job_thread [Job ID: {client_id}]: {e}", exc_info=True)
        job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e), "progress": 0}


# --- Resto de hilos de ejecución (sin cambios) ---
def run_live_animation_render_job_thread(client_id: str, config_payload: Dict[str, Any]): # ... (sin cambios)
    pass
def run_management_job_thread(client_id: str, workflow_type: str, config_payload: Dict[str, Any]): # ... (sin cambios)
    pass
def run_analysis_job_thread(client_id: str, workflow_type: str, config_payload: Dict[str, Any]): # ... (sin cambios)
    pass
def run_pid_creation_job_thread(client_id: str, config_payload: Dict[str, Any]): # ... (sin cambios)
    pass
def run_post_processing_job_thread(client_id: str, config_payload: Dict[str, Any]): # ... (sin cambios)
    pass
def run_finetuning_job_thread(client_id: str, workflow_type: str, config_payload: Dict[str, Any]): # ... (sin cambios)
    pass

@app.post("/job")
async def create_job(payload: JobPayload):
    client_id = payload.worker_job_id or str(uuid.uuid4())
    job_status_db[client_id] = {"status": "IN_QUEUE", "output": None, "error": None, "progress": 0}
    
    workflow_name = payload.workflow
    config = payload.config_payload
    
    # --- [ACTUALIZACIÓN] Añadimos el nuevo workflow a la lista de tipos especiales ---
    dataset_workflows = ["generate_dataset"]
    
    management_workflows = ["install_lora", "install_rvc"]
    finetuning_workflows = ["train_lora"]
    analysis_workflows = ["analyze_source_media"]
    pid_workflows = ["create_pid", "prepare_dataset"]
    post_proc_workflows = ["post_process_veritas"]
    live_anim_workflows = ["live_animation_render"]

    if workflow_name in management_workflows:
        thread = threading.Thread(target=run_management_job_thread, args=(client_id, workflow_name, config))
    elif workflow_name in dataset_workflows: # <-- Nueva condición
        thread = threading.Thread(target=run_dataset_generation_job_thread, args=(client_id, config))
    elif workflow_name in finetuning_workflows:
        thread = threading.Thread(target=run_finetuning_job_thread, args=(client_id, workflow_name, config))
    elif workflow_name in analysis_workflows:
        thread = threading.Thread(target=run_analysis_job_thread, args=(client_id, workflow_name, config))
    elif workflow_name in pid_workflows:
        thread = threading.Thread(target=run_pid_creation_job_thread, args=(client_id, config))
    elif workflow_name in post_proc_workflows:
        thread = threading.Thread(target=run_post_processing_job_thread, args=(client_id, config))
    elif workflow_name in live_anim_workflows:
        thread = threading.Thread(target=run_live_animation_render_job_thread, args=(client_id, config))
    else:
        thread = threading.Thread(target=run_job_thread, args=(client_id, workflow_name, config))
    
    thread.start()
    return {"message": "Trabajo recibido", "id": client_id, "status": "IN_QUEUE"}

@app.get("/status/{client_id}", response_model=StatusResponse)
async def get_job_status(client_id: str):
    # ... (sin cambios en esta función)
    if client_id not in job_status_db: raise HTTPException(status_code=404, detail="ID de trabajo no encontrado.")
    status_data = job_status_db[client_id]
    return StatusResponse(
        id=client_id, status=status_data.get("status", "UNKNOWN"),
        output=status_data.get("output"), error=status_data.get("error"),
        progress=status_data.get("progress", 0), previews=status_data.get("previews")
    )
