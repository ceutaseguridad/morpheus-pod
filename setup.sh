#!/bin/bash

# --- Script de Configuración y Arranque Autónomo para Morpheus Pod v8.1 (Lógica de Dependencias Saneada) ---
# ESTRATEGIA: Este script es la ÚNICA fuente de verdad para la instalación y el arranque.
# 1. Instala dependencias del sistema.
# 2. Clona ComfyUI y los nodos.
# 3. Sanea y unifica los archivos de requisitos para evitar conflictos con el entorno de RunPod.
# 4. Instala TODAS las dependencias de Python en un solo paso controlado.
# 5. Descarga los modelos de IA.
# 6. Lanza Supervisor.

# Salir inmediatamente si un comando falla
set -e

# --- FASE 1: INSTALACIÓN DE DEPENDENCIAS DEL SISTEMA ---
echo ">>> [FASE 1/5] Instalando dependencias del sistema (supervisor, ffmpeg, git)..."
apt-get update && apt-get install -y --no-install-recommends supervisor ffmpeg git

# --- FASE 2: CLONACIÓN DE COMFYUI Y NODOS PERSONALIZADOS ---
echo ">>> [FASE 2/5] Instalando ComfyUI y Nodos Personalizados..."
if [ ! -d "/workspace/ComfyUI" ]; then
    git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI
fi
CUSTOM_NODES_DIR="/workspace/ComfyUI/custom_nodes"
mkdir -p $CUSTOM_NODES_DIR
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-Manager" ]; then git clone https://github.com/ceutaseguridad/ComfyUI-Manager.git $CUSTOM_NODES_DIR/ComfyUI-Manager; fi
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-AnimateDiff-Evolved" ]; then git clone https://github.com/ceutaseguridad/ComfyUI-AnimateDiff-Evolved.git $CUSTOM_NODES_DIR/ComfyUI-AnimateDiff-Evolved; fi
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-VideoHelperSuite" ]; then git clone https://github.com/ceutaseguridad/ComfyUI-VideoHelperSuite.git $CUSTOM_NODES_DIR/ComfyUI-VideoHelperSuite; fi
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-wav2lip" ]; then git clone https://github.com/ceutaseguridad/ComfyUI-wav2lip.git $CUSTOM_NODES_DIR/ComfyUI-wav2lip; fi

# --- FASE 3: SANEAMIENTO Y UNIFICACIÓN DE DEPENDENCIAS DE PYTHON ---
echo ">>> [FASE 3/5] Saneando y unificando archivos de requisitos para evitar conflictos..."

# Definición de rutas a los archivos
COMFY_REQ_FILE="/workspace/ComfyUI/requirements.txt"
MORPHEUS_REQ_FILE="/workspace/requirements_pod.txt"
FINAL_REQ_FILE="/workspace/final_requirements.txt"

# PASO 3.1: Sanear el requirements.txt de ComfyUI.
# Usamos `grep -vE` para eliminar las líneas problemáticas en un solo comando.
# Esto evita que se intente degradar PyTorch y su versión de CUDA, ya que confiamos
# en la versión pre-optimizada que ya está en la imagen de RunPod.
# La expresión regular 'torch|pytorch-cuda|extra-index-url' busca y excluye cualquier línea que contenga esas palabras.
echo "Saneando el archivo de requisitos de ComfyUI..."
grep -vE 'torch|pytorch-cuda|extra-index-url' "$COMFY_REQ_FILE" > /tmp/saneado_comfy_reqs.txt

# PASO 3.2: Unificar todos los requisitos en un solo archivo final.
# Concatenamos el archivo saneado de ComfyUI con nuestros requisitos del pod.
echo "Unificando los requisitos de ComfyUI y Morpheus..."
cat /tmp/saneado_comfy_reqs.txt "$MORPHEUS_REQ_FILE" > "$FINAL_REQ_FILE"
rm /tmp/saneado_comfy_reqs.txt # Limpiamos el archivo temporal

echo "Archivo de requisitos final y unificado creado en $FINAL_REQ_FILE."

# --- FASE 4: INSTALACIÓN DE DEPENDENCIAS DE PYTHON ---
echo ">>> [FASE 4/5] Instalando todas las dependencias de Python en un solo paso controlado..."
# Usamos el intérprete de Python 3.11, que coincide con la nueva imagen base.
# El flag --upgrade asegura que los paquetes se actualicen a las versiones especificadas.
# pip resolverá el árbol de dependencias de este conjunto unificado una sola vez, evitando conflictos.
/usr/bin/python3.11 -m pip install --no-cache-dir --upgrade -r "$FINAL_REQ_FILE"
echo "Dependencias de Python instaladas con éxito."

# --- FASE 5: DESCARGA DE MODELOS DE IA ---
# (Esta sección es idéntica a la tuya, no requiere cambios)
echo ">>> [FASE 5/5] Descargando modelos de IA..."
wget -nc https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors -P /workspace/ComfyUI/models/checkpoints/
ANIMATE_DIFF_MODELS_DIR="/workspace/ComfyUI/models/animatediff_models"
mkdir -p $ANIMATE_DIFF_MODELS_DIR
wget -nc https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt -P $ANIMATE_DIFF_MODELS_DIR
WAV2LIP_MODELS_DIR="/workspace/ComfyUI/models/wav2lip"
mkdir -p $WAV2LIP_MODELS_DIR
wget -nc https://github.com/ceutaseguridad/Wav2Lip/releases/download/models/wav2lip.pth -P $WAV2LIP_MODELS_DIR
wget -nc https://github.com/ceutaseguridad/Wav2Lip/releases/download/models/wav2lip_gan.pth -P $WAV2LIP_MODELS_DIR

echo ""
echo "======================================================"
echo "==      [SETUP] Entorno del Pod configurado         =="
echo "======================================================"
echo ""

# --- FASE FINAL: EJECUCIÓN ---
echo ">>> Configuración completa. Iniciando los servicios con Supervisor..."

# Crear el directorio de logs antes de llamar a supervisord
mkdir -p /workspace/logs

# Ejecutar Supervisor como el proceso principal.
# Es crucial que supervisor.conf también use /usr/bin/python3.11 en sus comandos.
exec /usr/bin/supervisord -n -c /workspace/supervisor.conf
