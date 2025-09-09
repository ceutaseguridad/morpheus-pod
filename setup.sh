#!/bin/bash

# --- Script de Configuración y Arranque Autónomo para Morpheus Pod v8.2 (Solución a Conflicto distutils) ---
# ESTRATEGIA: Este script es la ÚNICA fuente de verdad para la instalación y el arranque.
# 1. Instala dependencias del sistema.
# 2. Clona ComfyUI y los nodos.
# 3. Sanea y unifica los archivos de requisitos para evitar conflictos con el entorno de RunPod.
# 4. Instala TODAS las dependencias de Python en un solo paso controlado, ignorando paquetes preinstalados conflictivos.
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
# Usamos `grep -vE` para eliminar las líneas problemáticas que intentan degradar PyTorch y su versión de CUDA.
# Confiamos en la versión pre-optimizada que ya está en la imagen de RunPod.
echo "Saneando el archivo de requisitos de ComfyUI..."
grep -vE 'torch|pytorch-cuda|extra-index-url' "$COMFY_REQ_FILE" > /tmp/saneado_comfy_reqs.txt

# PASO 3.2: Unificar todos los requisitos en un solo archivo final.
echo "Unificando los requisitos de ComfyUI y Morpheus..."
cat /tmp/saneado_comfy_reqs.txt "$MORPHEUS_REQ_FILE" > "$FINAL_REQ_FILE"
rm /tmp/saneado_comfy_reqs.txt # Limpiamos el archivo temporal

echo "Archivo de requisitos final y unificado creado en $FINAL_REQ_FILE."

# --- FASE 4: INSTALACIÓN DE DEPENDENCIAS DE PYTHON ---
echo ">>> [FASE 4/5] Instalando todas las dependencias de Python en un solo paso controlado..."
# Usamos el intérprete de Python 3.11, que coincide con la imagen base.
# Se añade el flag --ignore-installed:
# Este flag le dice a pip que no intente desinstalar la versión antigua de un paquete
# antes de instalar la nueva. En su lugar, simplemente sobrescribirá los archivos.
# Esto es CRÍTICO para evitar el error con paquetes "distutils" preinstalados
# en la imagen base, como 'blinker'.
# Los flags de timeout y reintentos añaden robustez a la red.
/usr/bin/python3.11 -m pip install \
    -v \
    --no-cache-dir \
    --upgrade \
    --ignore-installed \
    --timeout 60 \
    --retries 5 \
    -r "$FINAL_REQ_FILE"
echo "Dependencias de Python instaladas con éxito."

# --- FASE 5: DESCARGA DE MODELOS DE IA ---
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

# Ejecutar Supervisor como el proceso principal del contenedor.
# La directiva 'exec' reemplaza el proceso actual (este script) con supervisord,
# asegurando que el pod se mantenga vivo.
exec /usr/bin/supervisord -n -c /workspace/supervisor.conf
