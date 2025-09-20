# worker_server.py (Versión 20.1 - Veritas Fix)
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

MORPHEUS_AI_MODEL_ID = "ehartford/dolphin-2.2.1-mistral-7b"
VISUAL_MODEL_ID = "Salesforce/blip-image-captioning-base"

# --- Constantes y Base de Datos en Memoria ---
CONVERSATION_HISTORY_WINDOW = 10
COMFYUI_URL = "http://127.0.0.1:8189"
SERVER_ADDRESS = COMFYUI_URL.split("//")[1]
MORPHEUS_LIB_DIR = "/workspace/morpheus_lib"
job_status_db: Dict[str, Dict[str, Any]] = {}

# --- Lógica de Prompts del Sistema ---
MORPHEUS_SYSTEM_PROMPT = ""
FALLBACK_PROMPT = "Eres un asistente de IA servicial."
PROMPT_FILE_PATH = os.path.join(os.path.dirname(__file__), 'morpheus_system_prompt.txt')
try:
    with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as f:
        MORPHEUS_SYSTEM_PROMPT = f.read().strip()
    logger.info(f"System Prompt por defecto cargado con éxito desde '{PROMPT_FILE_PATH}'.")
except Exception as e:
    logger.critical(f"¡CRÍTICO! No se pudo leer el archivo de prompt: {e}. Se usará un prompt de respaldo.", exc_info=True)
    MORPHEUS_SYSTEM_PROMPT = FALLBACK_PROMPT


app = FastAPI(title="Morpheus AI Pod (Veritas)", version="20.1")

@app.on_event("startup")
async def startup_event():
    logger.info("Evento de arranque de FastAPI detectado. Iniciando carga de modelos en segundo plano.")
    threading.Thread(target=initialize_llm_background, daemon=True).start()
    if VISUAL_ANALYSIS_AVAILABLE:
        threading.Thread(target=initialize_vlm_background, daemon=True).start()
    else:
        logger.warning("Librerías para análisis visual no encontradas.")
    if FACE_ALIGNMENT_AVAILABLE:
        threading.Thread(target=initialize_face_alignment_background, daemon=True).start()
    else:
        logger.warning("Librería 'face-alignment' no encontrada.")

def initialize_llm_background():
    global llm_pipeline
    try:
        if torch.cuda.is_available():
            logger.info(f"HILO SECUNDARIO: Iniciando la carga del modelo de IA: {MORPHEUS_AI_MODEL_ID}...")
            tokenizer = AutoTokenizer.from_pretrained(MORPHEUS_AI_MODEL_ID)
            llm_pipeline = pipeline("text-generation", model=MORPHEUS_AI_MODEL_ID, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto")
            logger.info("HILO SECUNDARIO: Modelo de IA cargado con éxito.")
        else:
            logger.error("HILO SECUNDARIO: CUDA no está disponible.")
    except Exception as e:
        logger.critical(f"HILO SECUNDARIO: FALLO CRÍTICO al cargar el modelo de IA: {e}", exc_info=True)

def initialize_vlm_background():
    global vlm_processor, vlm_model
    try:
        if torch.cuda.is_available():
            logger.info(f"HILO SECUNDARIO: Iniciando la carga del modelo visual: {VISUAL_MODEL_ID}...")
            device = "cuda"
            vlm_processor = BlipProcessor.from_pretrained(VISUAL_MODEL_ID)
            vlm_model = BlipForConditionalGeneration.from_pretrained(VISUAL_MODEL_ID).to(device)
            logger.info("HILO SECUNDARIO: Modelo visual cargado con éxito.")
        else:
            logger.error("HILO SECUNDARIO: CUDA no está disponible para el VLM.")
    except Exception as e:
        logger.critical(f"HILO SECUNDARIO: FALLO CRÍTICO al cargar el modelo visual: {e}", exc_info=True)

def initialize_face_alignment_background():
    global fa
    try:
        logger.info("HILO SECUNDARIO: Cargando modelo de alineación facial...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
        logger.info("HILO SECUNDARIO: Modelo de alineación facial cargado con éxito.")
    except Exception as e:
        logger.critical(f"HILO SECUNDARIO: FALLO CRÍTICO al cargar el modelo de alineación facial: {e}", exc_info=True)
        fa = None

class ChatMessage(BaseModel):
    role: str; content: str

class ChatPayload(BaseModel):
    messages: List[ChatMessage]; context: Dict[str, Any]

class ActionData(BaseModel):
    details: Optional[str] = None; info_type: Optional[str] = None; file_type: Optional[str] = None
    job_id: Optional[str] = None; label: Optional[str] = None; options: Optional[List[str]] = None
    file_types: Optional[List[str]] = None; key: Optional[str] = None
    job_name: Optional[str] = None; job_type: Optional[str] = None; workflow: Optional[str] = None
    config_payload: Optional[Dict[str, Any]] = None
    name: Optional[str] = None; system_prompt: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None; tags: Optional[List[str]] = None
    execution_plan: Optional[List[Dict[str, Any]]] = None

class ChatResponse(BaseModel):
    response_text: str; action: str; action_data: ActionData; context: Optional[Dict[str, Any]]


def _analyze_and_tag_image(image_path: str, config_payload: Dict[str, Any]) -> Dict[str, Any]:
    # (Esta función permanece sin cambios)
    if not all([vlm_processor, vlm_model, llm_pipeline]):
        logger.warning("Análisis visual omitido: uno o más modelos de IA no están listos.")
        return {}
    try:
        logger.info(f"Iniciando análisis visual para: {image_path}")
        raw_image = Image.open(image_path).convert('RGB')
        inputs = vlm_processor(raw_image, return_tensors="pt").to(vlm_model.device)
        out = vlm_model.generate(**inputs, max_new_tokens=75)
        visual_description = vlm_processor.decode(out[0], skip_special_tokens=True)
        
        tagging_prompt_text = f"Extract a concise, comma-separated list of keywords or tags from the following description. Include subject, attributes, environment, and style. Description: {visual_description}"
        tagging_prompt = llm_pipeline.tokenizer.apply_chat_template([{"role": "user", "content": tagging_prompt_text}], tokenize=False, add_generation_prompt=True)
        outputs = llm_pipeline(tagging_prompt, max_new_tokens=64, do_sample=False)
        raw_tags = outputs[0]['generated_text'][len(tagging_prompt):].strip().replace("</s>", "")
        visual_tags = [tag.strip().lower() for tag in raw_tags.split(',') if tag.strip()]

        production_metadata = config_payload.get("production_metadata", {})
        
        final_tags = set(visual_tags)
        if production_metadata.get("actress_name"):
            final_tags.add(production_metadata["actress_name"].lower())

        final_metadata = {
            "project": production_metadata.get("project_name", "Sin Proyecto"),
            "character": production_metadata.get("actress_name"),
            "lora_model": config_payload.get("actress_lora"),
            "checkpoint_model": config_payload.get("checkpoint_name"),
            "workflow": production_metadata.get("workflow_type"),
            "visual_description": visual_description,
            "tags": sorted(list(final_tags))
        }
        
        logger.info(f"Análisis visual completado para {os.path.basename(image_path)}. Tags: {final_metadata['tags']}")
        return final_metadata
    except Exception as e:
        logger.error(f"Error durante el análisis visual de la imagen: {e}", exc_info=True)
        return {"error": "Visual analysis failed.", "details": str(e)}

def _get_video_duration(video_path: str) -> float:
    # (Esta función permanece sin cambios)
    if not VIDEO_ANALYSIS_AVAILABLE:
        logger.warning("Librería OpenCV no disponible, no se puede obtener la duración del vídeo.")
        return 0.0
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"No se pudo abrir el vídeo para análisis de duración: {video_path}")
            return 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps > 0:
            duration = frame_count / fps
            logger.info(f"Duración del vídeo '{os.path.basename(video_path)}' calculada: {duration:.2f} segundos.")
            return round(duration, 2)
        else:
            logger.warning(f"El FPS del vídeo es 0, no se puede calcular la duración para: {video_path}")
            return 0.0
    except Exception as e:
        logger.error(f"Error al obtener la duración del vídeo '{video_path}': {e}", exc_info=True)
        return 0.0

def get_morpheus_response(messages: List[ChatMessage], context: Dict[str, Any]) -> Dict[str, Any]:
    # (Esta función permanece sin cambios)
    if not llm_pipeline:
        return {"response_text": "El modelo de IA de la Conciencia aún está inicializándose...", "action": "wait_for_user", "action_data": {}, "context": context}
    new_context = context.copy()
    last_user_message = next((m.content for m in reversed(messages) if m.role == 'user'), "")
    is_confirming_plan = any(word in last_user_message.lower() for word in ["sí", "si", "procede", "confirmo", "hazlo", "adelante", "dale"])
    if is_confirming_plan and new_context.get("current_plan_list"):
        plan = new_context["current_plan_list"]
        if len(plan) == 1:
            action = "launch_workflow"
            action_data = plan[0] 
            response_text = "¡Perfecto! Lanzando el trabajo ahora. Puedes seguir su progreso en el 'Monitor de Tareas'."
        else:
            action = "launch_meta_workflow"
            action_data = {"execution_plan": plan} 
            response_text = "¡Excelente! Iniciando el plan de trabajo con múltiples pasos. Revisa el 'Monitor de Tareas' para ver cómo se crean."
        new_context.pop("current_plan_list", None)
        return {"response_text": response_text, "action": action, "action_data": action_data, "context": new_context}
    workbench = new_context.get("workbench", {})
    situation_briefing = ""
    if workbench:
        situation_briefing += "**Informe de Situación (Mesa de Trabajo):**\n"
        for item_id, item in workbench.items():
            base_info = f"- Asset ID: {item_id}, Rol: '{item['role']}', Fichero: '{item['filename']}'"
            try:
                if item.get("metadata_json"):
                    metadata = json.loads(item["metadata_json"])
                    meta_info = f", Info: Es '{metadata.get('character', 'N/A')}', LoRA: '{metadata.get('lora_model', 'N/A')}'"
                    base_info += meta_info
            except Exception: pass 
            situation_briefing += base_info + "\n"
        situation_briefing += "\n"
    situation_briefing += f"**Petición del Usuario:** \"{last_user_message}\"\n\n"
    situation_briefing += "**Tu Tarea:** Analiza la situación y la petición. Formula un plan de acción claro y responde en el formato JSON OBLIGATORIO que se te ha indicado en tus instrucciones."
    system_prompt_to_use = new_context.get("system_prompt", MORPHEUS_SYSTEM_PROMPT)
    tokenizer = llm_pipeline.tokenizer
    conversation_for_planning = [{"role": "system", "content": system_prompt_to_use}, {"role": "user", "content": situation_briefing}]
    prompt = tokenizer.apply_chat_template(conversation_for_planning, tokenize=False, add_generation_prompt=True)
    try:
        outputs = llm_pipeline(prompt, max_new_tokens=1024, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        raw_llm_output = outputs[0]['generated_text'][len(prompt):].strip().replace("</s>", "")
        json_start = raw_llm_output.find('{')
        json_end = raw_llm_output.rfind('}') + 1
        if json_start != -1:
            json_str = raw_llm_output[json_start:json_end]
            parsed_response = json.loads(json_str)
            response_text = parsed_response["plan_description"]
            action = "wait_for_user"
            action_data = {}
            new_context["current_plan_list"] = parsed_response["execution_plan"]
        else:
            response_text = "He tenido un problema formulando un plan. Esto es lo que pensé: " + raw_llm_output
            action = "wait_for_user"
            action_data = {}
    except Exception as e:
        logger.error(f"Error durante la inferencia o parseo del plan del LLM: {e}", exc_info=True)
        response_text = "He tenido un problema crítico procesando mi propio pensamiento..."
        action = "wait_for_user"
        action_data = {}
    final_response = {"response_text": response_text, "action": action, "action_data": action_data, "context": new_context}
    return final_response

@app.post("/chat", response_model=ChatResponse)
async def handle_chat(payload: ChatPayload):
    # (Esta función permanece sin cambios)
    response_data = get_morpheus_response(payload.messages, payload.context)
    action_data_obj = ActionData(**response_data.get("action_data", {}))
    return ChatResponse(response_text=response_data["response_text"], action=response_data["action"], action_data=action_data_obj, context=response_data["context"])

# --- Endpoints de gestión y de modelos (permanecen sin cambios) ---
@app.get("/")
def read_root(): return {"Morpheus Pod (Músculo y Conciencia - Veritas)": "Online"}
@app.get("/health")
def health_check(): return Response(status_code=200)
CHECKPOINTS_PATH = "/workspace/ComfyUI/models/checkpoints"
LORA_MODELS_PATH = "/workspace/ComfyUI/models/loras"
RVC_MODELS_PATH = "/workspace/ComfyUI/models/rvc"
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
os.makedirs(LORA_MODELS_PATH, exist_ok=True)
os.makedirs(RVC_MODELS_PATH, exist_ok=True)
def list_files_in_dir(directory: str, extensions: tuple, ignore_dirs: set = None) -> List[str]:
    if ignore_dirs is None: ignore_dirs = set()
    found_files = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for file in files:
            if file.endswith(extensions):
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                found_files.append(relative_path.replace("\\", "/"))
    return found_files
@app.get("/models/checkpoints", response_model=List[str])
async def list_checkpoints():
    try:
        ignore_dirs = {'vae', 'unet', 'text_encoder', 'text_encoder_2', 'scheduler', 'tokenizer'}
        return list_files_in_dir(CHECKPOINTS_PATH, ('.safetensors', '.ckpt', '.pt', '.pth'), ignore_dirs=ignore_dirs)
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
@app.get("/models/loras", response_model=List[str])
async def list_loras():
    try: return list_files_in_dir(LORA_MODELS_PATH, ('.safetensors', '.pt', '.pth'))
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
@app.delete("/models/loras/{lora_filename:path}")
async def delete_lora(lora_filename: str):
    try:
        file_path = os.path.join(LORA_MODELS_PATH, lora_filename)
        if os.path.exists(file_path): os.remove(file_path)
        else: raise HTTPException(status_code=404, detail="El archivo del modelo no fue encontrado.")
        return {"message": "Modelo eliminado con éxito"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
@app.get("/models/rvc", response_model=List[str])
async def list_rvc_models():
    try: return list_files_in_dir(RVC_MODELS_PATH, ('.pth', '.zip'))
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
@app.delete("/models/rvc/{rvc_filename:path}")
async def delete_rvc_model(rvc_filename: str):
    try:
        file_path = os.path.join(RVC_MODELS_PATH, rvc_filename)
        if os.path.exists(file_path): os.remove(file_path)
        else: raise HTTPException(status_code=404, detail="El archivo del modelo no fue encontrado.")
        return {"message": "Modelo eliminado con éxito"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))


class JobPayload(BaseModel):
    workflow: str
    worker_job_id: Optional[str] = None
    config_payload: Dict[str, Any] = {}

class StatusResponse(BaseModel):
    id: str
    status: str
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: int = 0
    previews: Optional[List[str]] = None

def queue_prompt(prompt: Dict[str, Any], client_id: str):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = request.Request(f"{COMFYUI_URL}/prompt", data=data)
    return json.loads(request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = parse.urlencode(data)
    with request.urlopen(f"{COMFYUI_URL}/view?{url_values}") as response: return response.read()

def get_history(prompt_id):
    with request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}") as response: return json.loads(response.read())

def update_workflow_with_payload(workflow_data: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Aplica los parámetros del payload al workflow JSON de ComfyUI."""
    PARAM_MAP = {
        "checkpoint_name": ("CheckpointLoaderSimple", "ckpt_name"),
        "prompt": ("CLIPTextEncode", "text"),
        "negative_prompt": ("CLIPTextEncode", "text"),
        "seed": ("KSampler", "seed"), "steps": ("KSampler", "steps"),
        "cfg_scale": ("KSampler", "cfg"), "sampler_name": ("KSampler", "sampler_name"),
        "width": ("EmptyLatentImage", "width"), "height": ("EmptyLatentImage", "height"),
        "actress_lora": ("LoraLoader", "lora_name"),
        "target_image_pod_path": ("LoadImage", "image"), "mask_image_pod_path": ("LoadImage", "image"),
        "target_video_pod_path": ("VHS_VideoLoader", "video"), "source_media_pod_path": ("LoadImage", "image"),
        "audio_pod_path": ("LoadAudio", "audio_file"),
        "pid_vector_path": ("LoadEmbeds", "embeds_path"), # Corregido para apuntar a LoadEmbeds
        "lens_distortion": ("ImageLensDistortion", "lens_distortion"),
        "chromatic_aberration": ("ImageLensDistortion", "chromatic_aberration"),
        "grain_amount": ("VHS_AddGrain", "amount")
    }

    for key, value in payload.items():
        if key in PARAM_MAP and value is not None:
            node_class, input_name = PARAM_MAP[key]
            for node_id, node in workflow_data.items():
                meta_title = node.get("_meta", {}).get("title", "").lower()
                if node.get("class_type") == node_class:
                    if key == 'prompt' and "negative" in meta_title: continue
                    if key == 'negative_prompt' and "negative" not in meta_title: continue
                    if key == 'target_image_pod_path' and "mask" in meta_title: continue
                    if key == 'mask_image_pod_path' and "mask" not in meta_title: continue
                    
                    if node_class in ["LoadImage", "VHS_VideoLoader", "LoadAudio", "LoadEmbeds"]:
                        node["inputs"][input_name] = os.path.basename(value)
                    else:
                        node["inputs"][input_name] = value
                    logger.info(f"Parámetro aplicado: Nodo '{node_id}' ({node_class}), Input '{input_name}' = '{value}'")
    return workflow_data

def run_job_thread(client_id: str, workflow_name: str, config_payload: Dict[str, Any]):
    job_dir = f"/workspace/job_data/{client_id}/output"
    os.makedirs(job_dir, exist_ok=True)
    try:
        job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 5, "output": None, "error": None}
        workflow_path = os.path.join(MORPHEUS_LIB_DIR, "workflows", f"{workflow_name}.json")
        with open(workflow_path, 'r') as f: workflow_data = json.load(f)
        updated_workflow = update_workflow_with_payload(workflow_data, config_payload)
        job_status_db[client_id]["progress"] = 20
        output_files = run_comfyui_generation(updated_workflow, job_dir, client_id)
        job_status_db[client_id]["progress"] = 90
        final_output = {}
        job_type = config_payload.get("job_type", "video") # Predeterminado a video para los nuevos flujos
        if job_type == "image" and output_files:
            final_output["image_pod_path"] = output_files[0]
            metadata = _analyze_and_tag_image(output_files[0], config_payload)
            if metadata: final_output["metadata"] = metadata
        elif job_type == "video" and output_files:
            final_output["video_pod_path"] = output_files[0]
            # ... (lógica de metadatos de vídeo)
        job_status_db[client_id] = {"status": "COMPLETED", "output": final_output, "error": None, "progress": 100}
    except Exception as e:
        logger.error(f"Fallo en run_job_thread [Job ID: {client_id}]: {e}", exc_info=True)
        job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e), "progress": 0}

# --- [VERITAS FIX] Nueva función de hilo para renderizado de Actuación Virtual ---
def run_live_animation_render_job_thread(client_id: str, config_payload: Dict[str, Any]):
    """Hilo para ejecutar el workflow de renderizado de Actuación Virtual."""
    job_dir = f"/workspace/job_data/{client_id}/output"
    os.makedirs(job_dir, exist_ok=True)
    try:
        job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 10, "output": None, "error": None}
        
        frames_input_dir = f"/workspace/job_data/{client_id}/input/captured_frames"
        os.makedirs(frames_input_dir, exist_ok=True)
        
        source_frames_dir = config_payload.get("captured_frames_dir_pod_path")
        if not source_frames_dir or not os.path.exists(source_frames_dir):
            raise ValueError("No se encontró el directorio de fotogramas capturados.")
        
        # Copiamos los fotogramas al directorio que espera el workflow
        for frame_file in os.listdir(source_frames_dir):
            shutil.copy(os.path.join(source_frames_dir, frame_file), frames_input_dir)
            
        job_status_db[client_id]["progress"] = 30
        
        workflow_path = os.path.join(MORPHEUS_LIB_DIR, "workflows", "live_animation_render.json")
        with open(workflow_path, 'r') as f: workflow_data = json.load(f)
        
        updated_workflow = update_workflow_with_payload(workflow_data, config_payload)
        output_files = run_comfyui_generation(updated_workflow, job_dir, client_id)
        
        if not output_files: raise RuntimeError("El workflow de Actuación Virtual no generó un vídeo de salida.")
            
        final_output = {"video_pod_path": output_files[0]}
        job_status_db[client_id] = {"status": "COMPLETED", "output": final_output, "error": None, "progress": 100}

    except Exception as e:
        logger.error(f"Fallo en run_live_animation_render_job_thread [Job ID: {client_id}]: {e}", exc_info=True)
        job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e), "progress": 0}


# --- Las demás funciones de hilo permanecen como estaban ---
def run_management_job_thread(client_id: str, workflow_type: str, config_payload: Dict[str, Any]):
    # (sin cambios)
    try:
        job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 10, "output": None, "error": None}
        url = config_payload.get('url'); filename = config_payload.get('filename')
        def update_status_callback(progress=None, status_text=None):
            if progress is not None: job_status_db[client_id]['progress'] = progress
        installed_path = management_handler.handle_install(workflow_type, url, filename, update_status_callback)
        job_status_db[client_id] = {"status": "COMPLETED", "output": {"installed_path": installed_path}, "error": None, "progress": 100}
    except Exception as e: job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e), "progress": 0}

def run_analysis_job_thread(client_id: str, workflow_type: str, config_payload: Dict[str, Any]):
    # (sin cambios)
    try:
        job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 10, "output": None, "error": None}
        source_pod_path = config_payload.get('source_media_pod_path')
        if not source_pod_path or not os.path.exists(source_pod_path): raise FileNotFoundError(f"Archivo no encontrado: {source_pod_path}")
        metadata = _analyze_and_tag_image(source_pod_path, {})
        job_status_db[client_id] = {"status": "COMPLETED", "output": {"metadata": metadata}, "error": None, "progress": 100}
    except Exception as e: job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e), "progress": 0}

def run_pid_creation_job_thread(client_id: str, config_payload: Dict[str, Any]):
    # (sin cambios)
    job_dir = f"/workspace/job_data/{client_id}/output"
    os.makedirs(job_dir, exist_ok=True)
    try:
        job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 10, "output": None, "error": None}
        dataset_input_dir = f"/workspace/job_data/{client_id}/input/dataset_images"
        os.makedirs(dataset_input_dir, exist_ok=True)
        source_image_paths = config_payload.get("base_images_pod_paths", [])
        if not source_image_paths: raise ValueError("No se proporcionaron imágenes.")
        for img_path in source_image_paths: shutil.copy(img_path, dataset_input_dir)
        job_status_db[client_id]["progress"] = 30
        workflow_path = os.path.join(MORPHEUS_LIB_DIR, "workflows", "create_pid.json")
        with open(workflow_path, 'r') as f: workflow_data = json.load(f)
        output_files = run_comfyui_generation(workflow_data, job_dir, client_id)
        if not output_files or not output_files[0].endswith('.pt'): raise RuntimeError("Workflow no generó archivo .pt")
        pid_file_path = output_files[0]
        final_output = {"pid_vector_path": pid_file_path, "influencer_db_id": config_payload.get("influencer_db_id")}
        job_status_db[client_id] = {"status": "COMPLETED", "output": final_output, "error": None, "progress": 100}
    except Exception as e: job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e), "progress": 0}

def run_post_processing_job_thread(client_id: str, config_payload: Dict[str, Any]):
    # (sin cambios)
    job_dir = f"/workspace/job_data/{client_id}/output"
    os.makedirs(job_dir, exist_ok=True)
    try:
        job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 10, "output": None, "error": None}
        video_input_path = config_payload.get("input_from_previous_step")
        if not video_input_path or not os.path.exists(video_input_path): raise FileNotFoundError(f"Vídeo no encontrado: {video_input_path}")
        input_dir = f"/workspace/job_data/{client_id}/input"
        os.makedirs(input_dir, exist_ok=True)
        shutil.copy(video_input_path, os.path.join(input_dir, "video_to_process.mp4")) # Renombrar para el workflow
        workflow_path = os.path.join(MORPHEUS_LIB_DIR, "workflows", "post_process_veritas.json")
        with open(workflow_path, 'r') as f: workflow_data = json.load(f)
        updated_workflow = update_workflow_with_payload(workflow_data, config_payload)
        output_files = run_comfyui_generation(updated_workflow, job_dir, client_id)
        if not output_files: raise RuntimeError("Post-procesado no generó salida.")
        final_output = {"video_pod_path": output_files[0]}
        job_status_db[client_id] = {"status": "COMPLETED", "output": final_output, "error": None, "progress": 100}
    except Exception as e: job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e), "progress": 0}

def run_comfyui_generation(workflow_json: Dict, output_dir: str, client_id: str) -> List[str]:
    # (sin cambios)
    ws = websocket.WebSocket()
    ws.connect(f"ws://{SERVER_ADDRESS}/ws?clientId={client_id}")
    prompt_id = queue_prompt(workflow_json, client_id)['prompt_id']
    output_files = []
    try:
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing' and message['data']['node'] is None and message['data']['prompt_id'] == prompt_id: break
        history = get_history(prompt_id)[prompt_id]
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            output_key = next((key for key in ['images', 'files', 'embeds'] if key in node_output), None)
            if output_key:
                for file_info in node_output[output_key]:
                    file_data = get_image(file_info['filename'], file_info.get('subfolder', ''), file_info.get('type', 'output'))
                    unique_filename = f"{uuid.uuid4().hex[:8]}_{file_info['filename']}"
                    output_path = os.path.join(output_dir, unique_filename)
                    with open(output_path, "wb") as f: f.write(file_data)
                    output_files.append(output_path)
    finally: ws.close()
    if not output_files: raise RuntimeError(f"Workflow no produjo salida (Prompt ID: {prompt_id}).")
    return output_files

def run_finetuning_job_thread(client_id: str, workflow_type: str, config_payload: Dict[str, Any]):
    # (sin cambios)
    try:
        job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 5, "output": None, "error": None, "previews": []}
        if workflow_type == "train_lora": # Simulación
            total_steps = config_payload.get("training_steps", 2000)
            for step in range(0, total_steps + 1, 100):
                time.sleep(1) 
                job_status_db[client_id]["progress"] = int((step / total_steps) * 100)
            final_lora_path = f"/workspace/ComfyUI/models/loras/{config_payload.get('output_lora_filename', 'trained_lora.safetensors')}"
            job_status_db[client_id]["status"] = "COMPLETED"
            job_status_db[client_id]["output"] = {"lora_path": final_lora_path}
    except Exception as e: job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e)}

@app.post("/job")
async def create_job(payload: JobPayload):
    client_id = payload.worker_job_id or str(uuid.uuid4())
    job_status_db[client_id] = {"status": "IN_QUEUE", "output": None, "error": None, "progress": 0}
    
    workflow_name = payload.workflow
    config = payload.config_payload
    
    management_workflows = ["install_lora", "install_rvc"]
    finetuning_workflows = ["train_lora"]
    analysis_workflows = ["analyze_source_media"]
    pid_workflows = ["create_pid", "prepare_dataset"]
    post_proc_workflows = ["post_process_veritas"]
    live_anim_workflows = ["live_animation_render"] # [VERITAS FIX] Nueva categoría

    if workflow_name in management_workflows:
        thread = threading.Thread(target=run_management_job_thread, args=(client_id, workflow_name, config))
    elif workflow_name in finetuning_workflows:
        thread = threading.Thread(target=run_finetuning_job_thread, args=(client_id, workflow_name, config))
    elif workflow_name in analysis_workflows:
        thread = threading.Thread(target=run_analysis_job_thread, args=(client_id, workflow_name, config))
    elif workflow_name in pid_workflows:
        thread = threading.Thread(target=run_pid_creation_job_thread, args=(client_id, config))
    elif workflow_name in post_proc_workflows:
        thread = threading.Thread(target=run_post_processing_job_thread, args=(client_id, config))
    elif workflow_name in live_anim_workflows: # [VERITAS FIX] Nuevo enrutamiento
        thread = threading.Thread(target=run_live_animation_render_job_thread, args=(client_id, config))
    else:
        thread = threading.Thread(target=run_job_thread, args=(client_id, workflow_name, config))
    
    thread.start()
    return {"message": "Trabajo recibido", "id": client_id, "status": "IN_QUEUE"}

@app.get("/status/{client_id}", response_model=StatusResponse)
async def get_job_status(client_id: str):
    # (sin cambios)
    if client_id not in job_status_db: raise HTTPException(status_code=404, detail="ID de trabajo no encontrado.")
    status_data = job_status_db[client_id]
    return StatusResponse(
        id=client_id, status=status_data.get("status", "UNKNOWN"),
        output=status_data.get("output"), error=status_data.get("error"),
        progress=status_data.get("progress", 0), previews=status_data.get("previews")
    )
