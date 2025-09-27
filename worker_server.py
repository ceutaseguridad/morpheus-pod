# Morphius/morpheus-pod/worker_server.py
# worker_server.py (v28.0 - Data-Driven Pipelines & Readiness State)
import logging
import json
import os
import uuid
import threading
import torch
import re
import time
import psutil
from pynvml import *
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

# --- Importaciones de la librería interna de Morpheus ---
from morpheus_lib import management_handler

# --- Importaciones para análisis visual y facial (sin cambios) ---
try:
    from transformers import BlipForConditionalGeneration, BlipProcessor
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
fa = None

MORPHEUS_AI_MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-Uncensored-4bit"
VISUAL_MODEL_ID = "Salesforce/blip-image-captioning-base"

# --- Constantes y Base de Datos en Memoria ---
COMFYUI_URL = "http://127.0.0.1:8189"
SERVER_ADDRESS = COMFYUI_URL.split("//")[1]
MORPHEUS_LIB_DIR = "/workspace/morpheus_lib"
job_status_db: Dict[str, Dict[str, Any]] = {}

# --- [CORRECCIÓN 1: ESTADO DE DISPONIBILIDAD DE MODELOS] ---
# Diccionario para rastrear el estado de carga de los modelos de IA.
MODEL_STATUS = {
    "llm": "unloaded",
    "vlm": "unloaded",
    "face_alignment": "unloaded",
}

# --- [CORRECCIÓN 2: CARGA DE PIPELINES DESDE CONFIGURACIÓN] ---
# Diccionario para almacenar las definiciones de meta-pipelines leídas desde JSON.
PIPELINES_CONFIG = {}

pod_specs = {
    "cpu_name": "N/A", "cpu_cores": 0, "total_ram_gb": 0,
    "gpu_name": "N/A", "gpu_vram_gb": 0, "benchmark_score_s_step": None,
    "status": "Initializing"
}

# --- Lógica de Prompts del Sistema (sin cambios) ---
MORPHEUS_SYSTEM_PROMPT = "Eres un asistente de IA servicial."
try:
    with open(os.path.join(os.path.dirname(__file__), 'morpheus_system_prompt.txt'), 'r', encoding='utf-8') as f:
        MORPHEUS_SYSTEM_PROMPT = f.read().strip()
except Exception as e:
    logger.critical(f"¡CRÍTICO! No se pudo leer el archivo de prompt: {e}.", exc_info=True)

app = FastAPI(title="Morpheus AI Pod (Libertatem)", version="28.0")

@app.on_event("startup")
async def startup_event():
    # --- [CORRECCIÓN 2.1: Cargar pipelines al inicio] ---
    load_pipelines_config()
    
    # Iniciar la carga de modelos en hilos de fondo
    threading.Thread(target=initialize_llm_background, daemon=True).start()
    if VISUAL_ANALYSIS_AVAILABLE: threading.Thread(target=initialize_vlm_background, daemon=True).start()
    if FACE_ALIGNMENT_AVAILABLE: threading.Thread(target=initialize_face_alignment_background, daemon=True).start()
    threading.Thread(target=initialize_specs_and_benchmark, daemon=True).start()

def load_pipelines_config():
    """Carga las definiciones de los meta-pipelines desde un archivo JSON."""
    global PIPELINES_CONFIG
    pipelines_file = os.path.join(os.path.dirname(__file__), 'pipelines.json')
    try:
        with open(pipelines_file, 'r') as f:
            PIPELINES_CONFIG = json.load(f)
        logger.info(f"Definiciones de meta-pipelines cargadas con éxito desde '{pipelines_file}'. Encontrados: {list(PIPELINES_CONFIG.keys())}")
    except FileNotFoundError:
        logger.warning(f"No se encontró el archivo 'pipelines.json'. Los meta-workflows no estarán disponibles.")
    except json.JSONDecodeError as e:
        logger.error(f"Error al parsear 'pipelines.json': {e}")

def initialize_specs_and_benchmark():
    # ... (sin cambios en esta función)
    global pod_specs
    try:
        pod_specs["cpu_cores"] = psutil.cpu_count(logical=True)
        pod_specs["total_ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        pod_specs["gpu_name"] = nvmlDeviceGetName(handle)
        gpu_memory = nvmlDeviceGetMemoryInfo(handle)
        pod_specs["gpu_vram_gb"] = round(gpu_memory.total / (1024**3), 2)
        nvmlShutdown()
        logger.info(f"Especificaciones del Pod obtenidas: {pod_specs}")
        time.sleep(45)
        pod_specs["status"] = "Ready"
    except Exception as e:
        pod_specs["status"] = f"Error: {e}"
        logger.error(f"Fallo al inicializar especificaciones: {e}", exc_info=True)

def initialize_llm_background():
    global llm_pipeline, MODEL_STATUS
    MODEL_STATUS["llm"] = "loading"
    try:
        if torch.cuda.is_available():
            tokenizer = AutoTokenizer.from_pretrained(MORPHEUS_AI_MODEL_ID)
            llm_pipeline = pipeline("text-generation", model=MORPHEUS_AI_MODEL_ID, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
            MODEL_STATUS["llm"] = "ready"
            logger.info(f"Modelo LLM '{MORPHEUS_AI_MODEL_ID}' cargado con éxito.")
    except Exception as e:
        MODEL_STATUS["llm"] = "error"
        logger.critical(f"FALLO CRÍTICO al cargar el modelo LLM: {e}", exc_info=True)

def initialize_vlm_background():
    global vlm_processor, vlm_model, MODEL_STATUS
    MODEL_STATUS["vlm"] = "loading"
    try:
        if torch.cuda.is_available():
            device = "cuda"
            vlm_processor = BlipProcessor.from_pretrained(VISUAL_MODEL_ID)
            vlm_model = BlipForConditionalGeneration.from_pretrained(VISUAL_MODEL_ID).to(device)
            MODEL_STATUS["vlm"] = "ready"
            logger.info("Modelo visual (VLM) cargado con éxito.")
    except Exception as e:
        MODEL_STATUS["vlm"] = "error"
        logger.critical(f"FALLO CRÍTICO al cargar el modelo VLM: {e}", exc_info=True)

def initialize_face_alignment_background():
    global fa, MODEL_STATUS
    MODEL_STATUS["face_alignment"] = "loading"
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
        MODEL_STATUS["face_alignment"] = "ready"
        logger.info("Modelo de alineación facial cargado con éxito.")
    except Exception as e:
        MODEL_STATUS["face_alignment"] = "error"
        logger.critical(f"FALLO CRÍTICO al cargar el modelo de alineación facial: {e}", exc_info=True)

# --- Definiciones de modelos Pydantic (sin cambios) ---
class ChatMessage(BaseModel): role: str; content: str
class ChatPayload(BaseModel): messages: List[ChatMessage]; context: Dict[str, Any]
class ActionData(BaseModel): details: Optional[str] = None
class ChatResponse(BaseModel): response_text: str; action: str; action_data: ActionData; context: Optional[Dict[str, Any]]
class JobPayload(BaseModel): workflow: str; worker_job_id: Optional[str] = None; config_payload: Dict[str, Any] = {}
class StatusResponse(BaseModel): id: str; status: str; output: Optional[Dict[str, Any]] = None; error: Optional[str] = None
class SpecsResponse(BaseModel): specs: Dict[str, Any]

# --- Funciones de ComfyUI (sin cambios) ---
def queue_prompt(prompt: Dict[str, Any], client_id: str):
    p = {"prompt": prompt, "client_id": client_id}; data = json.dumps(p).encode('utf-8'); req = request.Request(f"{COMFYUI_URL}/prompt", data=data); return json.loads(request.urlopen(req).read())
def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}; url_values = parse.urlencode(data); with request.urlopen(f"{COMFYUI_URL}/view?{url_values}") as response: return response.read()
def get_history(prompt_id):
    with request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as response: return json.loads(response.read())

# --- Lógica de ejecución de workflows (sin cambios) ---
def update_workflow_with_payload(workflow_data: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    updated_workflow = json.loads(json.dumps(workflow_data))
    model_package_name = payload.pop("model_package_name", None)
    if model_package_name:
        try:
            with open(os.path.join(os.path.dirname(__file__), 'model_packages.json'), 'r') as f: all_packages = json.load(f)
            workflow_type_key = payload.get('workflow_type', 'video_transfer') + "_packages"
            model_params = all_packages[workflow_type_key][model_package_name]
            payload = {**model_params, **payload}
        except Exception as e: raise ValueError(f"Paquete de modelos no válido: {model_package_name}")
    for node_id, node in updated_workflow.items():
        if 'inputs' in node:
            for key, value in list(node['inputs'].items()):
                if isinstance(value, str) and (value.startswith("__param:") or value.startswith("__file:") or value.startswith("__dir:")):
                    placeholder = value.split(":", 1)[1]
                    if placeholder in payload: node['inputs'][key] = payload[placeholder]
    return updated_workflow

def run_comfyui_generation(workflow_json: Dict, output_dir: str, client_id: str) -> List[str]:
    ws = websocket.WebSocket(); ws.connect(f"ws://{SERVER_ADDRESS}/ws?clientId={client_id}"); prompt_id = queue_prompt(workflow_json, client_id)['prompt_id']; output_files = []
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing' and message['data']['node'] is None and message['data']['prompt_id'] == prompt_id: break
    history = get_history(prompt_id)[prompt_id]
    for node_id, node_output in history['outputs'].items():
        for key in ['images', 'audio', 'text']:
            if key in node_output:
                for item in node_output[key]:
                    filename = item.get('filename', f"{node_id}_output.txt")
                    data = item if key == 'text' else get_image(filename, item['subfolder'], item['type'])
                    final_path = os.path.join(output_dir, filename)
                    mode = "w" if key == 'text' else "wb"
                    with open(final_path, mode, encoding='utf-8' if mode == 'w' else None) as f: f.write(data)
                    output_files.append(final_path)
    ws.close()
    return output_files

def run_chained_job_thread(client_id: str, task_chain: list, initial_payload: dict):
    job_dir = f"/workspace/job_data/{client_id}/output"; os.makedirs(job_dir, exist_ok=True)
    job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 5, "output": None, "error": None}
    pipeline_context = initial_payload.copy()
    total_tasks = len(task_chain)
    try:
        for i, task in enumerate(task_chain):
            workflow_name = task['workflow']
            config_payload = task.get('config_payload', {})
            
            # Chequeo de condición
            if "condition_flag" in task and not pipeline_context.get(task["condition_flag"], False):
                logger.info(f"[Chained Job ID: {client_id}] Omitiendo etapa {i+1}/{total_tasks}: '{workflow_name}' porque la bandera '{task['condition_flag']}' no es verdadera.")
                continue

            logger.info(f"[Chained Job ID: {client_id}] Etapa {i+1}/{total_tasks}: '{workflow_name}'")

            current_payload = {**pipeline_context, **config_payload}
            # Mapeo de configuraciones
            if "config_map" in task:
                for dest, src in task["config_map"].items():
                    if src in current_payload:
                        current_payload[dest] = current_payload[src]

            workflow_path = os.path.join(MORPHEUS_LIB_DIR, "workflows", f"{workflow_name}.json")
            with open(workflow_path, 'r') as f: workflow_data = json.load(f)
            
            updated_workflow = update_workflow_with_payload(workflow_data, current_payload)
            stage_client_id = f"{client_id}_stage_{i}"
            output_files = run_comfyui_generation(updated_workflow, job_dir, stage_client_id)
            
            if not output_files: raise RuntimeError(f"Etapa '{workflow_name}' sin salida.")
            
            output_file = output_files[0]
            if output_file.endswith(('.mp4', '.mov')): pipeline_context['video_pod_path'] = output_file
            elif output_file.endswith(('.wav', '.mp3')): pipeline_context['audio_pod_path'] = output_file
            pipeline_context['input_from_previous_step'] = output_file
            
            job_status_db[client_id]["progress"] = 5 + int(90 * (i + 1) / total_tasks)

        job_status_db[client_id] = {"status": "COMPLETED", "output": {"video_pod_path": pipeline_context.get('video_pod_path')}, "error": None, "progress": 100}
    except Exception as e:
        logger.error(f"Fallo en run_chained_job_thread [Job ID: {client_id}]: {e}", exc_info=True)
        job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e), "progress": 0}

def run_job_thread(client_id: str, workflow_name: str, config_payload: Dict[str, Any]):
    # ... (sin cambios en esta función)
    job_dir = f"/workspace/job_data/{client_id}/output"; os.makedirs(job_dir, exist_ok=True);
    try:
        job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 10, "output": None, "error": None}
        workflow_path = os.path.join(MORPHEUS_LIB_DIR, "workflows", f"{workflow_name}.json")
        with open(workflow_path, 'r') as f: workflow_data = json.load(f)
        config_payload["workflow_type"] = workflow_name
        updated_workflow = update_workflow_with_payload(workflow_data, config_payload)
        output_files = run_comfyui_generation(updated_workflow, job_dir, client_id)
        if not output_files: raise RuntimeError("La generación no produjo archivos.")
        output_key = "video_pod_path" if output_files[0].endswith(('.mp4', '.mov')) else "image_pod_path"
        job_status_db[client_id] = {"status": "COMPLETED", "output": {output_key: output_files[0]}, "error": None, "progress": 100}
    except Exception as e:
        logger.error(f"Fallo en run_job_thread [Job ID: {client_id}]: {e}", exc_info=True)
        job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e), "progress": 0}

# --- Endpoints FastAPI ---

@app.get("/specs", response_model=SpecsResponse)
async def get_pod_specs(): return SpecsResponse(specs=pod_specs)

@app.post("/job")
async def create_job(payload: JobPayload):
    client_id = payload.worker_job_id or str(uuid.uuid4())
    job_status_db[client_id] = {"status": "IN_QUEUE", "output": None, "error": None, "progress": 0}
    workflow_name = payload.workflow
    config = payload.config_payload
    
    # --- [CORRECCIÓN 2.2: LÓGICA DE ORQUESTACIÓN REFACTORIZADA] ---
    # Se unifica la lógica para manejar todos los tipos de pipelines.
    
    task_chain = []
    
    if workflow_name in PIPELINES_CONFIG:
        # Es un meta-workflow definido en pipelines.json (ej. "full_transform")
        logger.info(f"[Orquestador] Recibido meta-workflow '{workflow_name}' ID: {client_id}")
        task_chain = PIPELINES_CONFIG[workflow_name].get("steps", [])
        thread = threading.Thread(target=run_chained_job_thread, args=(client_id, task_chain, config))
        thread.start()
        return {"message": f"Meta-workflow '{workflow_name}' orquestado", "id": client_id, "status": "IN_QUEUE"}
        
    elif workflow_name == "dynamic_pipeline":
        # Es un pipeline dinámico enviado desde la UI
        task_chain = config.get("task_chain", [])
        if not task_chain:
            raise HTTPException(status_code=400, detail="El workflow 'dynamic_pipeline' requiere una 'task_chain' en el payload.")
        
        logger.info(f"[Orquestador] Recibido 'dynamic_pipeline' con {len(task_chain)} etapas. ID: {client_id}")
        thread = threading.Thread(target=run_chained_job_thread, args=(client_id, task_chain, config))
        thread.start()
        return {"message": f"Pipeline dinámico con {len(task_chain)} etapas orquestado", "id": client_id, "status": "IN_QUEUE"}
    
    else:
        # Es un trabajo simple, de una sola etapa
        logger.info(f"[Orquestador] Recibido trabajo simple '{workflow_name}'. ID: {client_id}")
        thread = threading.Thread(target=run_job_thread, args=(client_id, workflow_name, config))
        thread.start()
        return {"message": "Trabajo simple recibido", "id": client_id, "status": "IN_QUEUE"}


@app.get("/status/{client_id}", response_model=StatusResponse)
async def get_job_status(client_id: str):
    if client_id not in job_status_db: raise HTTPException(status_code=404, detail="ID de trabajo no encontrado.")
    return StatusResponse(id=client_id, **job_status_db[client_id])

@app.post("/cancel/{client_id}")
async def cancel_job(client_id: str):
    if client_id in job_status_db:
        job_status_db[client_id]['status'] = 'CANCELLED'; return Response(status_code=200)
    raise HTTPException(status_code=404, detail="ID de trabajo no encontrado.")

@app.post("/chat", response_model=ChatResponse)
async def handle_chat(payload: ChatPayload):
    # --- [CORRECCIÓN 1.1: COMPROBACIÓN DE DISPONIBILIDAD] ---
    if MODEL_STATUS["llm"] != "ready":
        raise HTTPException(status_code=503, detail=f"El modelo LLM no está disponible. Estado actual: {MODEL_STATUS['llm']}")
        
    system_prompt = payload.context.get('system_prompt', MORPHEUS_SYSTEM_PROMPT)
    conversation = [{"role": "system", "content": system_prompt}] + [msg.dict() for msg in payload.messages]
    try:
        terminators = [llm_pipeline.tokenizer.eos_token_id, llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = llm_pipeline(conversation, max_new_tokens=1024, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)
        generated_text = outputs[0]["generated_text"][-1]['content']
        json_match = re.search(r'\{[\s\S]*\}', generated_text)
        if json_match:
            try: return ChatResponse(**json.loads(json_match.group(0)))
            except (json.JSONDecodeError, Exception): pass
        return ChatResponse(response_text=generated_text, action="wait_for_user", action_data={})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
