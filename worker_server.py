# worker_server.py (Versión 16.0 - Fine-Tuning REAL y Automatizado)
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
import asyncio
import random
import sys
import shutil
import subprocess
import numpy as np

# --- Importaciones existentes ---
from morpheus_lib import management_handler

# --- NUEVAS IMPORTACIONES PARA LÓGICA DE FINE-TUNING ---
try:
    import face_alignment
    from skimage import io
    from PIL import Image, ImageDraw
    FACE_ALIGNMENT_AVAILABLE = True
except ImportError:
    FACE_ALIGNMENT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger("MorpheusPodServer")

llm_pipeline = None
MORPHEUS_AI_MODEL_ID = "ehartford/dolphin-2.2.1-mistral-7b"
CONVERSATION_HISTORY_WINDOW = 10
COMFYUI_URL = "http://127.0.0.1:8189"
SERVER_ADDRESS = COMFYUI_URL.split("//")[1]
MORPHEUS_LIB_DIR = "/workspace/morpheus_lib"
job_status_db: Dict[str, Dict[str, Any]] = {}

KNOWN_FUNCTIONAL_LORAS = {
    "lcm": {
        "filename": "lcm_lora_sdxl.safetensors",
        "keywords": ["rápido", "acelerar", "velocidad", "lcm", "instantáneo"]
    },
    "detailer": {
        "filename": "detailer-xl.safetensors",
        "keywords": ["detalle", "detallado", "mejorar", "calidad", "realismo", "texturas"]
    }
}

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

app = FastAPI(title="Morpheus AI Pod", version="16.0")

# --- NUEVO: Inicializar el modelo de detección facial en el arranque ---
fa = None

@app.on_event("startup")
async def startup_event():
    global fa
    logger.info("Evento de arranque de FastAPI detectado. Iniciando carga de modelos en segundo plano.")
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, initialize_llm_background)
    
    if FACE_ALIGNMENT_AVAILABLE:
        try:
            logger.info("Cargando modelo de alineación facial (face-alignment)...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
            logger.info("Modelo de alineación facial cargado con éxito.")
        except Exception as e:
            logger.critical(f"FALLO CRÍTICO al cargar el modelo de alineación facial: {e}", exc_info=True)
            fa = None
    else:
        logger.warning("Librería 'face-alignment' no encontrada. El workflow de fine-tuning automático no estará disponible.")

def initialize_llm_background():
    global llm_pipeline
    try:
        if torch.cuda.is_available():
            logger.info(f"HILO SECUNDARIO: Iniciando la carga del modelo de IA: {MORPHEUS_AI_MODEL_ID}...")
            tokenizer = AutoTokenizer.from_pretrained(MORPHEUS_AI_MODEL_ID)
            llm_pipeline = pipeline("text-generation", model=MORPHEUS_AI_MODEL_ID, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto")
            logger.info("HILO SECUNDARIO: Modelo de IA cargado con éxito y listo para usarse.")
        else:
            logger.error("HILO SECUNDARIO: CUDA no está disponible.")
    except Exception as e:
        logger.critical(f"HILO SECUNDARIO: FALLO CRÍTICO al cargar el modelo de IA: {e}", exc_info=True)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    messages: List[ChatMessage]
    context: Dict[str, Any]

class ActionData(BaseModel):
    details: Optional[str] = None; info_type: Optional[str] = None; file_type: Optional[str] = None; job_id: Optional[str] = None; label: Optional[str] = None; options: Optional[List[str]] = None; file_types: Optional[List[str]] = None; key: Optional[str] = None; job_name: Optional[str] = None; job_type: Optional[str] = None; workflow: Optional[str] = None; config_payload: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response_text: str; action: str; action_data: ActionData; context: Optional[Dict[str, Any]]

def _handle_creation_flow(context: Dict[str, Any]) -> (str, Dict[str, Any], str):
    collected_data = context.get("collected_data", {})
    if "prompt" not in collected_data:
        return "wait_for_user", {}, "¡Entendido! Empecemos un flujo de creación. Por favor, describe la escena que quieres generar (este será tu prompt principal)."
    if "actress_lora" not in collected_data and "character_prompt" not in collected_data:
         return "request_local_info", {"info_type": "actress_list"}, "Gracias por el prompt. Ahora, ¿quieres usar un modelo de identidad (LoRA) preexistente o prefieres describir un personaje desde cero?"
    if "negative_prompt" not in collected_data:
        return "wait_for_user", {}, "Perfecto. Por último, ¿tienes un prompt negativo en mente para evitar ciertos elementos en la imagen?"
    return "launch_workflow", {}, "¡Genial! Tengo toda la información necesaria. Procedo a lanzar el trabajo de creación de imagen."

def _handle_deepfake_flow(context: Dict[str, Any]) -> (str, Dict[str, Any], str):
    collected_data = context.get("collected_data", {})
    if "target_image_local_path" not in collected_data and "target_video_local_path" not in collected_data:
        return "request_file_upload", {"label": "Por favor, sube la imagen o vídeo que quieres modificar.", "file_types": ["png", "jpg", "jpeg", "webp", "mp4", "mov", "avi"]}, "De acuerdo, empecemos un trabajo de deepfake. Necesito que subas el archivo (imagen o vídeo) al que le aplicaremos el cambio de rostro."
    if "actress_lora" not in collected_data:
        return "request_local_info", {"info_type": "actress_list"}, "Archivo recibido. Ahora, por favor, dime qué modelo de identidad (LoRA) quieres usar para el reemplazo facial."
    return "launch_workflow", {}, "¡Perfecto! Tengo el archivo y el modelo. Iniciando el trabajo de deepfake."

def get_morpheus_response(messages: List[ChatMessage], context: Dict[str, Any]) -> Dict[str, Any]:
    if not llm_pipeline:
        return {"response_text": "El modelo de IA de la Conciencia aún está inicializándose...", "action": "wait_for_user", "action_data": {}, "context": context}
    system_prompt_to_use = context.get("system_prompt", MORPHEUS_SYSTEM_PROMPT)
    is_test_chat_mode = "parameters" in context
    action = "wait_for_user"
    action_data = {}
    new_context = context.copy()
    llm_prompt_instruction = "La conversación está abierta. Responde de forma natural y directa según tu personalidad."
    if not is_test_chat_mode:
        new_context.setdefault("collected_data", {})
        last_user_message_obj = next((m for m in reversed(messages) if m.role == 'user'), None)
        last_system_message_obj = next((m for m in reversed(messages) if m.role == 'system'), None)
        if last_user_message_obj:
            last_user_message = last_user_message_obj.content
            last_user_message_lower = last_user_message.lower()
            if not new_context.get("current_workflow"):
                if "crear imagen" in last_user_message_lower or "generar vídeo" in last_user_message_lower: new_context["current_workflow"] = "creation"
                elif "deepfake" in last_user_message_lower: new_context["current_workflow"] = "deepfake"
                elif "lip-sync" in last_user_message_lower: new_context["current_workflow"] = "lipsync"
                elif "composición" in last_user_message_lower: new_context["current_workflow"] = "composition"
            
            if new_context.get("current_workflow") == "creation":
                new_context["collected_data"].setdefault("functional_loras", [])
                for lora_key, lora_info in KNOWN_FUNCTIONAL_LORAS.items():
                    if any(keyword in last_user_message_lower for keyword in lora_info["keywords"]):
                        if lora_info["filename"] not in new_context["collected_data"]["functional_loras"]:
                            new_context["collected_data"]["functional_loras"].append(lora_info["filename"])
                            logger.info(f"Detectado trigger para LoRA funcional: {lora_info['filename']}")

            if "he seleccionado:" in last_user_message_lower:
                selection = last_user_message.split('`')[1]
                new_context["collected_data"]["actress_lora"] = selection
            elif "he subido el archivo:" in last_user_message_lower:
                filename = last_user_message.split('`')[1]
                file_key = next((k for k, v in new_context.get("collected_data", {}).items() if isinstance(v, dict) and v.get("name") == filename), None)
                if file_key:
                    if new_context.get("current_workflow") == "deepfake":
                        new_context["collected_data"]["target_image_local_path"] = file_key
            elif new_context.get("current_workflow") == "creation" and "prompt" not in new_context["collected_data"]:
                new_context["collected_data"]["prompt"] = last_user_message

        if last_system_message_obj and "Respuesta a la consulta 'actress_list': []" in last_system_message_obj.content:
            llm_prompt_instruction = "El usuario no tiene modelos de identidad disponibles. Infórmale amablemente que necesita instalar uno en la sección 'Recursos' antes de continuar y finaliza la conversación."
            new_context["current_workflow"] = None
        else:
            current_workflow = new_context.get("current_workflow")
            if current_workflow:
                handler_map = {"creation": _handle_creation_flow, "deepfake": _handle_deepfake_flow}
                handler = handler_map.get(current_workflow)
                if handler:
                    action, action_data, llm_prompt_instruction = handler(new_context)
    if is_test_chat_mode:
        system_prompt_with_context = system_prompt_to_use
    else:
        system_prompt_with_context = f"""
{system_prompt_to_use}
---
**SITUACIÓN ACTUAL:** Estás operando como el asistente Morpheus.
**CONTEXTO DEL SISTEMA:**
{json.dumps(new_context, indent=2, default=str)}
**TU OBJETIVO INMEDIATO:**
{llm_prompt_instruction}
---
"""
    tokenizer = llm_pipeline.tokenizer
    limited_messages = messages[-CONVERSATION_HISTORY_WINDOW:]
    conversation = [{"role": "system", "content": system_prompt_with_context}]
    for msg in limited_messages:
        role = "user" if msg.role in ["user", "system"] else "assistant"
        conversation.append({"role": role, "content": msg.content})
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    raw_response_text = ""
    try:
        outputs = llm_pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        raw_response_text = outputs[0]['generated_text'][len(prompt):].strip().replace("</s>", "")
    except Exception as e:
        logger.error(f"Error durante la inferencia del LLM: {e}", exc_info=True)
        raw_response_text = "He tenido un problema crítico procesando mi propio pensamiento..."
    if action == "launch_workflow":
        workflow = new_context.get("current_workflow")
        job_type_map = {"creation": "image", "deepfake": "image", "lipsync": "video", "composition": "video"}
        action_data = {"job_name": f"Morpheus AI - {workflow}", "job_type": job_type_map.get(workflow, "unknown"), "workflow": workflow, "config_payload": new_context.get("collected_data", {})}
    final_response = {"response_text": raw_response_text, "action": action, "action_data": action_data, "context": new_context}
    return final_response

@app.post("/chat", response_model=ChatResponse)
async def handle_chat(payload: ChatPayload):
    loop = asyncio.get_running_loop()
    response_data = await loop.run_in_executor(None, get_morpheus_response, payload.messages, payload.context)
    return response_data

@app.get("/")
def read_root():
    return {"Morpheus Pod (Músculo y Conciencia)": "Online"}

@app.get("/health")
def health_check():
    return Response(status_code=200)

CHECKPOINTS_PATH = "/workspace/ComfyUI/models/checkpoints"
LORA_MODELS_PATH = "/workspace/ComfyUI/models/loras"
RVC_MODELS_PATH = "/workspace/ComfyUI/models/rvc"

os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
os.makedirs(LORA_MODELS_PATH, exist_ok=True)
os.makedirs(RVC_MODELS_PATH, exist_ok=True)

def list_files_in_dir(directory: str, extensions: tuple, ignore_dirs: set = None) -> List[str]:
    if ignore_dirs is None:
        ignore_dirs = set()
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/loras", response_model=List[str])
async def list_loras():
    try:
        return list_files_in_dir(LORA_MODELS_PATH, ('.safetensors', '.pt', '.pth'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.delete("/models/loras/{lora_filename:path}")
async def delete_lora(lora_filename: str):
    try:
        file_path = os.path.join(LORA_MODELS_PATH, lora_filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            raise HTTPException(status_code=404, detail="El archivo del modelo no fue encontrado.")
        return {"message": "Modelo eliminado con éxito"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/models/rvc", response_model=List[str])
async def list_rvc_models():
    try:
        return list_files_in_dir(RVC_MODELS_PATH, ('.pth', '.zip'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.delete("/models/rvc/{rvc_filename:path}")
async def delete_rvc_model(rvc_filename: str):
    try:
        file_path = os.path.join(RVC_MODELS_PATH, rvc_filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            raise HTTPException(status_code=404, detail="El archivo del modelo no fue encontrado.")
        return {"message": "Modelo eliminado con éxito"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class JobPayload(BaseModel):
    workflow: str
    worker_job_id: Optional[str] = None
    config_payload: Dict[str, Any] = {}

class StatusResponse(BaseModel):
    id: str; status: str; output: Optional[Dict[str, Any]] = None; error: Optional[str] = None; progress: int = 0

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
    # --- AÑADIMOS EL MAPEO PARA LA MÁSCARA DE INPAINTING ---
    PARAM_MAP = {
        "checkpoint_name": ("CheckpointLoaderSimple", "ckpt_name"),
        "prompt": ("CLIPTextEncode", "text"), 
        "negative_prompt": ("CLIPTextEncode", "text"), 
        "seed": ("KSampler", "seed"), 
        "steps": ("KSampler", "steps"), 
        "cfg_scale": ("KSampler", "cfg"), 
        "sampler_name": ("KSampler", "sampler_name"), 
        "width": ("EmptyLatentImage", "width"), 
        "height": ("EmptyLatentImage", "height"), 
        "actress_lora": ("LoraLoader", "lora_name"), 
        "target_image_pod_path": ("LoadImage", "image"), 
        "mask_image_pod_path": ("LoadImage", "image"), # Clave para la máscara
        "target_video_pod_path": ("VHS_VideoLoader", "video"), 
        "source_media_pod_path": ("LoadImage", "image"), 
        "audio_pod_path": ("LoadAudio", "audio_file")
    }

    for key, value in payload.items():
        if key in PARAM_MAP and value is not None:
            node_class, input_name = PARAM_MAP[key]
            for node_id, node in workflow_data.items():
                meta_title = node.get("_meta", {}).get("title", "").lower()
                is_correct_node = False
                if node.get("class_type") == node_class:
                    if key == 'prompt' and "negative" in meta_title: continue
                    if key == 'negative_prompt' and "negative" not in meta_title: continue
                    # --- LÓGICA DE DESAMBIGUACIÓN PARA LoadImage ---
                    if key == 'target_image_pod_path' and "mask" in meta_title: continue
                    if key == 'mask_image_pod_path' and "mask" not in meta_title: continue
                    is_correct_node = True
                if is_correct_node:
                    if node_class in ["LoadImage", "VHS_VideoLoader", "LoadAudio"]: 
                        node["inputs"][input_name] = os.path.basename(value)
                    else: 
                        node["inputs"][input_name] = value
                    logger.info(f"Parámetro aplicado: Nodo '{node_id}' ({node_class}), Input '{input_name}' = '{value}'")

    if "actress_lora" not in payload or not payload.get("actress_lora"):
        # ... (código de bypass de LoRA sin cambios) ...
        pass
    
    if payload.get("character_prompt"):
        # ... (código de fusión de prompt sin cambios) ...
        pass

    functional_loras = payload.get("functional_loras", [])
    if functional_loras:
        # ... (código de encadenamiento de LoRAs sin cambios) ...
        pass

    return workflow_data

def run_job_thread(client_id: str, workflow_name: str, config_payload: Dict[str, Any]):
    # ... (código sin cambios)
    pass

def run_management_job_thread(client_id: str, workflow_type: str, config_payload: Dict[str, Any]):
    # ... (código sin cambios)
    pass
    
# --- NUEVA FUNCIÓN UTILITARIA PARA EJECUTAR WORKFLOWS DE COMFYUI ---
def run_comfyui_generation(workflow_json: Dict, output_dir: str, client_id: str) -> List[str]:
    """Ejecuta un workflow en ComfyUI y devuelve las rutas de las imágenes de salida."""
    ws = websocket.WebSocket()
    ws.connect(f"ws://{SERVER_ADDRESS}/ws?clientId={client_id}")
    
    prompt_id = queue_prompt(workflow_json, client_id)['prompt_id']
    output_images = []

    try:
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing' and message['data']['node'] is None and message['data']['prompt_id'] == prompt_id:
                    break # Execution is done
        
        history = get_history(prompt_id)[prompt_id]
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                for image_info in node_output['images']:
                    image_data = get_image(image_info['filename'], image_info['subfolder'], image_info['type'])
                    # Usamos un nombre de archivo único para evitar colisiones en el inpainting
                    unique_filename = f"{uuid.uuid4()}_{image_info['filename']}"
                    output_path = os.path.join(output_dir, unique_filename)
                    with open(output_path, "wb") as f:
                        f.write(image_data)
                    output_images.append(output_path)
    finally:
        ws.close()

    if not output_images:
        raise RuntimeError(f"El workflow de ComfyUI (Prompt ID: {prompt_id}) no produjo ninguna imagen de salida.")
    return output_images

# --- NUEVAS FUNCIONES PARA CREACIÓN AUTOMÁTICA DE MÁSCARAS ---
def create_feature_mask(image_path: str, feature: str, output_path: str):
    """Crea una máscara poligonal para un rasgo facial específico."""
    if not fa:
        raise RuntimeError("El modelo de Face Alignment no está disponible.")
    
    input_image = io.imread(image_path)
    # Convertir a RGB si tiene canal alfa
    if input_image.shape[-1] == 4:
        input_image = input_image[..., :3]
        
    preds = fa.get_landmarks(input_image)
    if not preds:
        raise ValueError(f"No se detectaron caras en la imagen: {image_path}")
        
    landmarks = preds[0]
    
    # Mapeo de rasgos a índices de puntos de referencia (modelo 2D)
    feature_map = {
        'nose': list(range(27, 36)),
        'left_eye': list(range(36, 42)),
        'right_eye': list(range(42, 48)),
        'mouth': list(range(48, 68))
    }
    
    if feature not in feature_map:
        raise ValueError(f"Rasgo '{feature}' no soportado. Soportados: {list(feature_map.keys())}")
    
    points = [tuple(p) for p in landmarks[feature_map[feature]]]
    
    mask = Image.new('L', (input_image.shape[1], input_image.shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(points, outline=255, fill=255)
    mask.save(output_path)

# --- HILO DE FINE-TUNING ACTUALIZADO CON LÓGICA REAL ---
def run_finetuning_job_thread(client_id: str, workflow_type: str, config_payload: Dict[str, Any]):
    job_dir = f"/workspace/job_data/{client_id}"
    os.makedirs(job_dir, exist_ok=True)
    
    try:
        job_status_db[client_id] = {"status": "IN_PROGRESS", "progress": 5, "output": None, "error": None}
        
        if workflow_type == "generate_and_modify_dataset":
            # ETAPA 1.1: GENERACIÓN REAL DEL DATASET BASE
            logger.info(f"[Job ID: {client_id}] Etapa 1.1: Generando dataset base...")
            base_dataset_dir = os.path.join(job_dir, "base_dataset")
            os.makedirs(base_dataset_dir, exist_ok=True)

            with open(os.path.join(MORPHEUS_LIB_DIR, "workflows", "creation.json"), 'r') as f:
                creation_workflow_template = json.load(f)

            gen_payload = {
                "actress_lora": config_payload['base_lora_filename'],
                "prompt": f"RAW photo, portrait of a person, high detail, neutral expression",
                "negative_prompt": "cartoon, 3d, painting, ugly, deformed, blurry"
            }
            base_workflow = update_workflow_with_payload(creation_workflow_template, gen_payload)

            dataset_size = config_payload.get('dataset_size', 20)
            for i in range(dataset_size):
                base_workflow["3"]["inputs"]["seed"] = random.randint(0, sys.maxsize)
                run_comfyui_generation(base_workflow, base_dataset_dir, f"{client_id}_gen_{i}")
                job_status_db[client_id]["progress"] = 5 + int(45 * (i + 1) / dataset_size)
            
            # ETAPA 1.2: MODIFICACIÓN REAL CON INPAINTING AUTOMÁTICO
            logger.info(f"[Job ID: {client_id}] Etapa 1.2: Modificando dataset con Inpainting automático...")
            modified_dataset_dir = os.path.join(job_dir, config_payload["output_folder_name"])
            os.makedirs(modified_dataset_dir, exist_ok=True)
            
            mod_prompt = config_payload['modifications_prompt'].lower()
            features_to_modify = []
            if 'nariz' in mod_prompt or 'nose' in mod_prompt: features_to_modify.append('nose')
            if 'ojo' in mod_prompt or 'eye' in mod_prompt: 
                features_to_modify.append('left_eye'); features_to_modify.append('right_eye')
            if 'boca' in mod_prompt or 'mouth' in mod_prompt: features_to_modify.append('mouth')
            if not features_to_modify:
                raise ValueError("No se detectaron rasgos faciales modificables en el prompt (nariz, ojo, boca).")

            with open(os.path.join(MORPHEUS_LIB_DIR, "workflows", "inpainting.json"), 'r') as f:
                inpaint_workflow_template = json.load(f)

            base_images = sorted(os.listdir(base_dataset_dir))
            for i, img_name in enumerate(base_images):
                img_path = os.path.join(base_dataset_dir, img_name)
                current_img_to_inpaint = img_path
                
                for feature in features_to_modify:
                    mask_path = os.path.join(job_dir, f"mask_{i}_{feature}.png")
                    create_feature_mask(current_img_to_inpaint, feature, mask_path)
                    
                    inpaint_payload = {
                        "actress_lora": config_payload['base_lora_filename'],
                        "prompt": config_payload['modifications_prompt'],
                        "target_image_pod_path": current_img_to_inpaint,
                        "mask_image_pod_path": mask_path,
                        "seed": random.randint(0, sys.maxsize)
                    }
                    inpaint_workflow = update_workflow_with_payload(inpaint_workflow_template.copy(), inpaint_payload)
                    
                    # El resultado del inpainting se usa como entrada para el siguiente rasgo
                    output_files = run_comfyui_generation(inpaint_workflow, job_dir, f"{client_id}_inpaint_{i}_{feature}")
                    current_img_to_inpaint = output_files[0]
                
                # Mover la imagen final modificada al directorio del dataset
                shutil.move(current_img_to_inpaint, os.path.join(modified_dataset_dir, img_name))
                job_status_db[client_id]["progress"] = 50 + int(50 * (i + 1) / len(base_images))
            
            job_status_db[client_id] = {"status": "COMPLETED", "output": {"dataset_path": modified_dataset_dir}, "error": None, "progress": 100}

        elif workflow_type == "train_lora":
            logger.info(f"[Job ID: {client_id}] Etapa 2: Iniciando entrenamiento de LoRA...")
            job_status_db[client_id]["progress"] = 10
            
            dataset_path = config_payload.get("input_path")
            output_lora_filename = config_payload.get("output_lora_filename")
            training_steps = config_payload.get("training_steps", 1500)
            
            if not all([dataset_path, output_lora_filename]):
                raise ValueError("Faltan parámetros críticos para el entrenamiento: input_path, output_lora_filename.")
            
            output_lora_path = os.path.join(LORA_MODELS_PATH, output_lora_filename)
            training_output_dir = os.path.join(job_dir, "training_output")
            os.makedirs(training_output_dir, exist_ok=True)
            
            command = [
                "accelerate", "launch", "/workspace/kohya_ss/train_network.py",
                f"--pretrained_model_name_or_path={os.path.join(CHECKPOINTS_PATH, 'sd_xl_base_1.0.safetensors')}",
                f"--train_data_dir={dataset_path}",
                f"--output_dir={training_output_dir}",
                f"--output_name={output_lora_filename.replace('.safetensors', '')}",
                "--network_module=networks.lora",
                "--save_model_as=safetensors",
                f"--max_train_steps={training_steps}",
                "--learning_rate=1e-4", "optimizer_type=AdamW8bit", "--mixed_precision=fp16", "--xformers"
            ]
            
            logger.info(f"[Job ID: {client_id}] Ejecutando comando: {' '.join(command)}")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
            
            for line in process.stdout:
                logger.info(f"[Kohya_ss - Job {client_id}] {line.strip()}")

            process.wait()
            
            if process.returncode != 0:
                raise RuntimeError(f"El proceso de entrenamiento falló con código {process.returncode}.")

            final_lora_file = os.path.join(training_output_dir, output_lora_filename)
            if os.path.exists(final_lora_file):
                shutil.move(final_lora_file, output_lora_path)
                logger.info(f"[Job ID: {client_id}] LoRA entrenado movido a: {output_lora_path}")
            else:
                raise FileNotFoundError(f"El archivo LoRA entrenado no se encontró: {final_lora_file}")
            
            job_status_db[client_id] = {"status": "COMPLETED", "output": {"lora_path": output_lora_path}, "error": None, "progress": 100}

    except Exception as e:
        logger.error(f"Fallo en run_finetuning_job_thread [Job ID: {client_id}]: {e}", exc_info=True)
        job_status_db[client_id] = {"status": "FAILED", "output": None, "error": str(e), "progress": 0}

@app.post("/job")
async def create_job(payload: JobPayload):
    client_id = payload.worker_job_id or str(uuid.uuid4())
    job_status_db[client_id] = {"status": "IN_QUEUE", "output": None, "error": None, "progress": 0}
    
    workflow_name = payload.workflow
    management_workflows = ["install_lora", "install_rvc"]
    finetuning_workflows = ["generate_and_modify_dataset", "train_lora"]

    if workflow_name in management_workflows:
        logger.info(f"Enrutando trabajo [ID: {client_id}] al manejador de gestión para workflow: {workflow_name}")
        thread = threading.Thread(target=run_management_job_thread, args=(client_id, workflow_name, payload.config_payload))
    elif workflow_name in finetuning_workflows:
        logger.info(f"Enrutando trabajo [ID: {client_id}] al manejador de fine-tuning: {workflow_name}")
        thread = threading.Thread(target=run_finetuning_job_thread, args=(client_id, workflow_name, payload.config_payload))
    else:
        logger.info(f"Enrutando trabajo [ID: {client_id}] al manejador de ComfyUI para workflow: {workflow_name}")
        thread = threading.Thread(target=run_job_thread, args=(client_id, workflow_name, payload.config_payload))
    
    thread.start()
    return {"message": "Trabajo recibido", "id": client_id, "status": "IN_QUEUE"}

@app.get("/status/{client_id}", response_model=StatusResponse)
async def get_job_status(client_id: str):
    if client_id not in job_status_db:
        raise HTTPException(status_code=404, detail="ID de trabajo no encontrado.")
    return {"id": client_id, **job_status_db[client_id]}
