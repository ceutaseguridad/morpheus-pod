# Morphius/morpheus-pod/worker_server.py
# worker_server.py (Versión 25.1 - Relighting Stage)
import logging
import json
import os
import uuid
import threading
import torch
import re
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

# --- Importaciones para análisis visual y facial ---
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


app = FastAPI(title="Morpheus AI Pod (Veritas)", version="25.1")

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=initialize_llm_background, daemon=True).start()
    if VISUAL_ANALYSIS_AVAILABLE: threading.Thread(target=initialize_vlm_background, daemon=True).start()
    if FACE_ALIGNMENT_AVAILABLE: threading.Thread(target=initialize_face_alignment_background, daemon=True).start()

def initialize_llm_background():
    global llm_pipeline
    try:
        if torch.cuda.is_available():
            tokenizer = AutoTokenizer.from_pretrained(MORPHEUS_AI_MODEL_ID)
            llm_pipeline = pipeline("text-generation", model=MORPHEUS_AI_MODEL_ID, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto")
            logger.info("Modelo de IA (LLM) cargado con éxito.")
    except Exception as e:
        logger.critical(f"FALLO CRÍTICO al cargar el modelo de IA (LLM): {e}", exc_info=True)

def initialize_vlm_background():
    global vlm_processor, vlm_model
    try:
        if torch.cuda.is_available():
            device = "cuda"
            vlm_processor = BlipProcessor.from_pretrained(VISUAL_MODEL_ID)
            vlm_model = BlipForConditionalGeneration.from_pretrained(VISUAL_MODEL_ID).to(device)
            logger.info("Modelo visual (VLM) cargado con éxito.")
    except Exception as e:
        logger.critical(f"FALLO CRÍTICO al cargar el modelo visual (VLM): {e}", exc_info=True)

def initialize_face_alignment_background():
    global fa
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
        logger.info("Modelo de alineación facial cargado con éxito.")
    except Exception as e:
        logger.critical(f"FALLO CRÍTICO al cargar el modelo de alineación facial: {e}", exc_info=True)

# --- Modelos Pydantic ---
class ChatMessage(BaseModel): role: str; content: str
class ChatPayload(BaseModel): messages: List[ChatMessage]; context: Dict[str, Any]
class ActionData(BaseModel):
    details: Optional[str] = None; info_type: Optional[str] = None; file_type: Optional[str] = None
    job_id: Optional[str] = None; label: Optional[str] = None; options: Optional[List[str]] = None
    file_types: Optional[List[str]] = None; key: Optional[str] = None
    job_name: Optional[str] = None; job_type: Optional[str] = None; workflow: Optional[str] = None
    config_payload: Optional[Dict[str, Any]] = None
    name: Optional[str] = None; system_prompt: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None; tags: Optional[List[str]] = None
    execution_plan: Optional[List[Dict[str, Any]]] = None
    media_id: Optional[int] = None; role: Optional[str] = None
    project_name: Optional[str] = None; prompt: Optional[str] = None; negative_prompt: Optional[str] = None

class ChatResponse(BaseModel): response_text: str; action: str; action_data: ActionData; context: Optional[Dict[str, Any]]
class JobPayload(BaseModel): workflow: str; worker_job_id: Optional[str] = None; config_payload: Dict[str, Any] = {}
class StatusResponse(BaseModel): id: str; status: str; output: Optional[Dict[str, Any]] = None; error: Optional[str] = None; progress: int = 0; previews: Optional[List[str]] = None

# --- Funciones de ComfyUI ---
def queue_prompt(prompt: Dict[str, Any], client_id: str):
    p = {"prompt": prompt, "client_id": client_id}; data = json.dumps(p).encode('utf-8'); req = request.Request(f"{COMFYUI_URL}/prompt", data=data); return json.loads(request.urlopen(req).read())
def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}; url_values = parse.urlencode(data); with request.urlopen(f"{COMFYUI_URL}/view?{url_values}") as response: return response.read()
def get_history(prompt_id):
    with request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as response: return json.loads(response.read())
def update_workflow_with_payload(workflow_data: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    updated_workflow = json.loads(json.dumps(workflow_data))
    for node in updated_workflow.values():
        if 'inputs' in node:
            for key, value in node['inputs'].items():
                if isinstance(value, str):
                    if value.startswith("__param:"):
                        param_name = value.split(":", 1)[1]
                        if param_name in payload: node['inputs'][key] = payload[param_name]
                    elif value.startswith("__file:"):
                        param_name = value.split(":", 1)[1]
                        if param_name in payload:
                            # Manejo especial para poder pasar rutas de vídeo o audio
                            if 'video' in node['inputs']:
                                node['inputs']['video'] = payload[param_name]
                            elif 'audio' in node['inputs']:
                                node['inputs']['audio'] = payload[param_name]
                            else:
                                node['inputs']['image'] = payload[param_name]
                    elif value.startswith("__dir:"):
                        param_name = value.split(":", 1)[1]
                        if param_name in payload: node['inputs']['directory'] = payload[param_name]
    return updated_workflow
def run_comfyui_generation(workflow_json: Dict, output_dir: str, client_id: str) -> List[str]:
    ws = websocket.WebSocket(); ws.connect(f"ws://{SERVER_ADDRESS}/ws?clientId={client_id}"); prompt_id = queue_prompt(workflow_json, client_id)['prompt_id']; output_files = [];
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing' and message['data']['node'] is None and message['data']['prompt_id'] == prompt_id: break
    history = get_history(prompt_id)[prompt_id]
    for node_id, node_output in history['outputs'].items():
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                final_path = os.path.join(output_dir, image['filename'])
                with open(final_path, "wb") as f: f.write(image_data)
                output_files.append(final_path)
        if 'audio' in node_output:
             for audio_file in node_output['audio']:
                audio_data = get_image(audio_file['filename'], audio_file['subfolder'], audio_file['type'])
                final_path = os.path.join(output_dir, audio_file['filename'])
                with open(final_path, "wb") as f: f.write(audio_data)
                output_files.append(final_path)
        if 'text' in node_output:
            for text_content in node_output['text']:
                final_path = os.path.join(output_dir, f"{node_id}_output.txt")
                with open(final_path, "w", encoding='utf-8') as f: f.write(text_content)
                output_files.append(final_path)
    ws.close()
    return output_files

# --- Hilos de Ejecución de Trabajos ---
def run_job_thread(client_id: str, workflow_name: str, config_payload: Dict[str, Any]):
    job_dir = f"/workspace/job_data/{client_id}/output"; os.makedirs(job_dir, exist_ok=True);
    try:
        job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 10, "output": None, "error": None}
        workflow_path = os.path.join(MORPHEUS_LIB_DIR, "workflows", f"{workflow_name}.json")
        with open(workflow_path, 'r') as f: workflow_data = json.load(f)
        updated_workflow = update_workflow_with_payload(workflow_data, config_payload)
        output_files = run_comfyui_generation(updated_workflow, job_dir, client_id)
        if not output_files: raise RuntimeError("La generación no produjo archivos.")
        output_key = "video_pod_path" if output_files[0].endswith(('.mp4', '.mov')) else "image_pod_path"
        job_status_db[client_id] = {"status": "COMPLETED", "output": {output_key: output_files[0]}, "error": None, "progress": 100}
    except Exception as e:
        logger.error(f"Fallo en run_job_thread [Job ID: {client_id}]: {e}", exc_info=True)
        job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e), "progress": 0}

def run_dataset_generation_job_thread(client_id: str, workflow_name: str, config_payload: Dict[str, Any]):
    job_dir = f"/workspace/job_data/{client_id}/output"
    output_folder_name = config_payload.get("output_folder_name", f"dataset_{client_id}")
    dataset_output_dir = os.path.join(job_dir, output_folder_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    try:
        job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 5, "output": None, "error": None}
        workflow_path = os.path.join(MORPHEUS_LIB_DIR, "workflows", f"{workflow_name}.json")
        with open(workflow_path, 'r') as f: workflow_data = json.load(f)
        dataset_size = config_payload.get("dataset_size", 5)
        all_generated_files = []
        for i in range(dataset_size):
            logger.info(f"[Job ID: {client_id}] Generando imagen de dataset {i+1}/{dataset_size}...")
            iter_payload = config_payload.copy(); iter_payload['seed'] = random.randint(0, 1_000_000_000)
            updated_workflow = update_workflow_with_payload(workflow_data, iter_payload)
            iter_client_id = f"{client_id}_iter_{i}"; output_files = run_comfyui_generation(updated_workflow, dataset_output_dir, iter_client_id)
            if output_files: all_generated_files.extend(output_files)
            progress = 5 + int(90 * (i + 1) / dataset_size); job_status_db[client_id]["progress"] = progress
        if not all_generated_files: raise RuntimeError("La generación del dataset no produjo ninguna imagen.")
        
        final_output = {"image_pod_path": all_generated_files[0]} if dataset_size == 1 else {"base_images_pod_paths": dataset_output_dir}
        job_status_db[client_id] = {"status": "COMPLETED", "output": final_output, "error": None, "progress": 100}
    except Exception as e:
        logger.error(f"Fallo en run_dataset_generation_job_thread [Job ID: {client_id}]: {e}", exc_info=True)
        job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e), "progress": 0}

def run_pid_creation_job_thread(client_id: str, config_payload: Dict[str, Any]):
    job_dir = f"/workspace/job_data/{client_id}/output"; os.makedirs(job_dir, exist_ok=True);
    try:
        job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 10, "output": None, "error": None}
        workflow_path = os.path.join(MORPHEUS_LIB_DIR, "workflows", "create_pid.json")
        with open(workflow_path, 'r') as f: workflow_data = json.load(f)
        updated_workflow = update_workflow_with_payload(workflow_data, config_payload)
        output_files = run_comfyui_generation(updated_workflow, job_dir, client_id)
        if not output_files: raise RuntimeError("La creación del PID no generó un archivo de salida.")
        final_output = {"pid_vector_path": output_files[0]}; job_status_db[client_id] = {"status": "COMPLETED", "output": final_output, "error": None, "progress": 100}
    except Exception as e:
        logger.error(f"Fallo en run_pid_creation_job_thread [Job ID: {client_id}]: {e}", exc_info=True)
        job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e), "progress": 0}

def run_management_job_thread(client_id: str, workflow_type: str, config_payload: Dict[str, Any]):
    try:
        job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 10, "output": None, "error": None}
        def update_status(progress, status_text=None): job_status_db[client_id]["progress"] = progress
        installed_path = management_handler.handle_install(workflow_type, config_payload['url'], config_payload['filename'], update_status)
        job_status_db[client_id] = {"status": "COMPLETED", "output": {"installed_path": installed_path}, "error": None, "progress": 100}
    except Exception as e:
        logger.error(f"Fallo en run_management_job_thread [Job ID: {client_id}]: {e}", exc_info=True)
        job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e), "progress": 0}

def run_rvc_training_thread(client_id: str, config_payload: Dict[str, Any]):
    job_dir = f"/workspace/job_data/{client_id}/output"; os.makedirs(job_dir, exist_ok=True);
    try:
        job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 10, "output": None, "error": None}
        workflow_path = os.path.join(MORPHEUS_LIB_DIR, "workflows", "train_rvc.json")
        with open(workflow_path, 'r') as f: workflow_data = json.load(f)
        
        config_payload['model_name'] = f"{config_payload['model_name']}.pth"

        updated_workflow = update_workflow_with_payload(workflow_data, config_payload)
        output_files = run_comfyui_generation(updated_workflow, job_dir, client_id)
        if not output_files: raise RuntimeError("El entrenamiento RVC no produjo un archivo de modelo.")
        
        final_output = {"rvc_model_path": output_files[0]}
        job_status_db[client_id] = {"status": "COMPLETED", "output": final_output, "error": None, "progress": 100}
    except Exception as e:
        logger.error(f"Fallo en run_rvc_training_thread [Job ID: {client_id}]: {e}", exc_info=True)
        job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e), "progress": 0}

def run_vlm_analysis_thread(client_id: str, workflow_name: str, config_payload: Dict[str, Any]):
    job_dir = f"/workspace/job_data/{client_id}/output"; os.makedirs(job_dir, exist_ok=True);
    try:
        job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 10, "output": None, "error": None}
        workflow_path = os.path.join(MORPHEUS_LIB_DIR, "workflows", f"{workflow_name}.json")
        with open(workflow_path, 'r') as f: workflow_data = json.load(f)
        updated_workflow = update_workflow_with_payload(workflow_data, config_payload)
        output_files = run_comfyui_generation(updated_workflow, job_dir, client_id)
        if not output_files: raise RuntimeError("El análisis VLM no produjo un archivo de texto.")
        
        final_output = {"text_file_pod_path": output_files[0]}
        job_status_db[client_id] = {"status": "COMPLETED", "output": final_output, "error": None, "progress": 100}
    except Exception as e:
        logger.error(f"Fallo en run_vlm_analysis_thread [Job ID: {client_id}]: {e}", exc_info=True)
        job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e), "progress": 0}

def run_audio_job_thread(client_id: str, workflow_name: str, config_payload: Dict[str, Any]):
    job_dir = f"/workspace/job_data/{client_id}/output"; os.makedirs(job_dir, exist_ok=True)
    try:
        job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 10, "output": None, "error": None}
        workflow_path = os.path.join(MORPHEUS_LIB_DIR, "workflows", f"{workflow_name}.json")
        with open(workflow_path, 'r') as f: workflow_data = json.load(f)
        
        updated_workflow = update_workflow_with_payload(workflow_data, config_payload)
        output_files = run_comfyui_generation(updated_workflow, job_dir, client_id)
        
        if not output_files: raise RuntimeError("El trabajo de audio no produjo un archivo de salida.")
        
        if workflow_name == 'transcribe_audio':
            final_output = {"text_file_pod_path": output_files[0]}
        else:
            final_output = {"audio_pod_path": output_files[0]}

        job_status_db[client_id] = {"status": "COMPLETED", "output": final_output, "error": None, "progress": 100}
    except Exception as e:
        logger.error(f"Fallo en run_audio_job_thread [Job ID: {client_id}, Workflow: {workflow_name}]: {e}", exc_info=True)
        job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e), "progress": 0}

def run_video_editing_job_thread(client_id: str, workflow_name: str, config_payload: Dict[str, Any]):
    job_dir = f"/workspace/job_data/{client_id}/output"; os.makedirs(job_dir, exist_ok=True)
    try:
        job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 10, "output": None, "error": None}
        workflow_path = os.path.join(MORPHEUS_LIB_DIR, "workflows", f"{workflow_name}.json")
        with open(workflow_path, 'r') as f: workflow_data = json.load(f)

        updated_workflow = update_workflow_with_payload(workflow_data, config_payload)
        output_files = run_comfyui_generation(updated_workflow, job_dir, client_id)

        if not output_files: raise RuntimeError("El trabajo de edición de vídeo no produjo un archivo de salida.")
        
        final_output = {"video_pod_path": output_files[0]}
        job_status_db[client_id] = {"status": "COMPLETED", "output": final_output, "error": None, "progress": 100}
    except Exception as e:
        logger.error(f"Fallo en run_video_editing_job_thread [Job ID: {client_id}, Workflow: {workflow_name}]: {e}", exc_info=True)
        job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e), "progress": 0}

def run_chained_job_thread(client_id: str, chained_tasks: list):
    job_dir = f"/workspace/job_data/{client_id}/output"; os.makedirs(job_dir, exist_ok=True)
    previous_outputs = {}
    total_tasks = len(chained_tasks)
    
    try:
        job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 5, "output": None, "error": None, "previews": []}
        
        for i, task in enumerate(chained_tasks):
            workflow_name = task['workflow']
            config_payload = task.get('config_payload', {})
            
            logger.info(f"[Chained Job ID: {client_id}] Iniciando etapa {i+1}/{total_tasks}: {workflow_name}")
            config_payload.update(previous_outputs)
            
            workflow_path = os.path.join(MORPHEUS_LIB_DIR, "workflows", f"{workflow_name}.json")
            with open(workflow_path, 'r') as f: workflow_data = json.load(f)
            
            updated_workflow = update_workflow_with_payload(workflow_data, config_payload)
            iter_client_id = f"{client_id}_stage_{i}"
            output_files = run_comfyui_generation(updated_workflow, job_dir, iter_client_id)
            
            if not output_files: raise RuntimeError(f"La etapa '{workflow_name}' no produjo archivos de salida.")
            
            output_file = output_files[0]
            if output_file.endswith(('.mp4', '.mov')):
                previous_outputs['video_pod_path'] = output_file
            elif output_file.endswith(('.wav', '.mp3')):
                previous_outputs['audio_pod_path'] = output_file
            
            progress = 5 + int(90 * (i + 1) / total_tasks)
            job_status_db[client_id]["progress"] = progress

        logger.info(f"[Chained Job ID: {client_id}] Pipeline completado. Salida final: {previous_outputs.get('video_pod_path')}")
        job_status_db[client_id] = {"status": "COMPLETED", "output": {"video_pod_path": previous_outputs.get('video_pod_path')}, "error": None, "progress": 100}
        
    except Exception as e:
        logger.error(f"Fallo en run_chained_job_thread [Job ID: {client_id}]: {e}", exc_info=True)
        job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e), "progress": 0}

@app.post("/job")
async def create_job(payload: JobPayload):
    client_id = payload.worker_job_id or str(uuid.uuid4())
    job_status_db[client_id] = {"status": "IN_QUEUE", "output": None, "error": None, "progress": 0}
    workflow_name = payload.workflow; config = payload.config_payload
    
    if workflow_name == "full_transform":
        logger.info(f"[Orquestador] Recibido trabajo 'full_transform' con ID: {client_id}")
        task_chain = []
        
        base_video_payload = {"target_video_pod_path": config["target_video_pod_path"]}
        task_chain.append({"workflow": "extract_audio", "config_payload": base_video_payload})
        if config.get("enable_voice_clean", True):
            task_chain.append({"workflow": "clean_audio", "config_payload": {}})
        task_chain.append({"workflow": "voice_transfer", "config_payload": {"actress_rvc": config["actress_rvc"], "pitch_shift": config["pitch_shift"], "index_ratio": config["index_ratio"]}})
        task_chain.append({"workflow": "video_transfer", "config_payload": {"target_video_pod_path": config["target_video_pod_path"], "pid_vector_pod_path": config["pid_vector_pod_path"], "denoise": config["denoise"], "identity_weight": config["identity_weight"], "seed": config["seed"]}})
        task_chain.append({"workflow": "mux_video_audio", "config_payload": {}})

        if config.get("enable_post_process", False) or config.get("lut_pod_path"):
            post_proc_payload = {"lens_distortion": config.get("lens_distortion", 0.0), "chromatic_aberration": config.get("chromatic_aberration", 0.0), "grain_amount": config.get("grain_amount", 0.0)}
            if config.get("lut_pod_path"): post_proc_payload["lut_pod_path"] = config["lut_pod_path"]
            task_chain.append({"workflow": "post_process_veritas", "config_payload": post_proc_payload})

        thread = threading.Thread(target=run_chained_job_thread, args=(client_id, task_chain))
        thread.start()
        return {"message": "Pipeline de transformación total recibido y orquestado", "id": client_id, "status": "IN_QUEUE"}

    workflow_map = {
        "train_rvc": (run_rvc_training_thread, (client_id, config)),
        "analyze_image": (run_vlm_analysis_thread, (client_id, workflow_name, config)),
        "generate_dataset_from_scratch": (run_dataset_generation_job_thread, (client_id, workflow_name, config)),
        "generate_dataset_from_reference": (run_dataset_generation_job_thread, (client_id, workflow_name, config)),
        "create_pid": (run_pid_creation_job_thread, (client_id, config)),
        "install_lora": (run_management_job_thread, (client_id, workflow_name, config)),
        "install_rvc": (run_management_job_thread, (client_id, workflow_name, config)),
        "text_to_speech": (run_audio_job_thread, (client_id, workflow_name, config)),
        "generate_sfx": (run_audio_job_thread, (client_id, workflow_name, config)),
        "clean_audio": (run_audio_job_thread, (client_id, workflow_name, config)),
        "extract_audio": (run_audio_job_thread, (client_id, workflow_name, config)),
        "transcribe_audio": (run_audio_job_thread, (client_id, workflow_name, config)),
        "voice_transfer": (run_audio_job_thread, (client_id, workflow_name, config)),
        "stitch_video": (run_video_editing_job_thread, (client_id, workflow_name, config)),
        "video_inpainting": (run_video_editing_job_thread, (client_id, workflow_name, config)),
        "mux_video_audio": (run_video_editing_job_thread, (client_id, workflow_name, config)),
        "relight_video": (run_video_editing_job_thread, (client_id, workflow_name, config)),
    }

    if workflow_name in workflow_map:
        target_func, args = workflow_map[workflow_name]
    else:
        target_func = run_job_thread
        args = (client_id, workflow_name, config)
        
    thread = threading.Thread(target=target_func, args=args)
    thread.start()
    return {"message": "Trabajo recibido", "id": client_id, "status": "IN_QUEUE"}


@app.get("/status/{client_id}", response_model=StatusResponse)
async def get_job_status(client_id: str):
    if client_id not in job_status_db: raise HTTPException(status_code=404, detail="ID de trabajo no encontrado.")
    status_data = job_status_db[client_id]; return StatusResponse(id=client_id, status=status_data.get("status", "UNKNOWN"), output=status_data.get("output"), error=status_data.get("error"), progress=status_data.get("progress", 0), previews=status_data.get("previews"))

@app.post("/cancel/{client_id}")
async def cancel_job(client_id: str):
    if client_id in job_status_db:
        if job_status_db[client_id]['status'] in ['IN_QUEUE', 'IN_PROGRESS']: job_status_db[client_id]['status'] = 'CANCELLED'; return Response(status_code=200, content=f"Trabajo {client_id} cancelado.")
        else: return Response(status_code=400, content=f"El trabajo {client_id} no se puede cancelar en su estado actual.")
    else: raise HTTPException(status_code=404, detail="ID de trabajo no encontrado.")

@app.post("/chat", response_model=ChatResponse)
async def handle_chat(payload: ChatPayload):
    if llm_pipeline is None:
        raise HTTPException(status_code=503, detail="El modelo de IA (LLM) no está listo. Inténtalo de nuevo en unos momentos.")
    
    system_prompt = MORPHEUS_SYSTEM_PROMPT
    if payload.context and payload.context.get('system_prompt'):
        system_prompt = payload.context['system_prompt']

    conversation_history = [{"role": "system", "content": system_prompt}]
    conversation_history.extend([msg.dict() for msg in payload.messages])

    try:
        terminators = [llm_pipeline.tokenizer.eos_token_id, llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = llm_pipeline(conversation_history, max_new_tokens=1024, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)
        generated_text = outputs[0]["generated_text"][-1]['content']
        
        json_match = re.search(r'\{[\s\S]*\}', generated_text)
        if json_match:
            try:
                json_response = json.loads(json_match.group(0))
                response_obj = ChatResponse(**json_response)
                return response_obj
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"Fallo al decodificar o validar el JSON de la IA: {e}")
                return ChatResponse(response_text=f"La IA devolvió una respuesta mal formada. Texto: {generated_text}", action="wait_for_user", action_data={})
        else:
            return ChatResponse(response_text=generated_text, action="wait_for_user", action_data={})

    except Exception as e:
        logger.critical(f"Error crítico durante la generación de texto de la IA: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
