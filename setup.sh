#!/bin/bash

# --- Script de Configuración y Arranque Autónomo para Morpheus Pod v7.1 (Corrección de Logs) ---
# ESTRATEGIA: Se clonan todos los nodos desde los forks de 'ceutaseguridad'.
# Los modelos de Wav2Lip se descargan desde el fork verificado.
# Se añade la creación del directorio de logs para Supervisor.

# Salir inmediatamente si un comando falla
set -e

# --- FASE 1: INSTALACIÓN DE DEPENDENCIAS DEL SISTEMA ---
echo ">>> [FASE 1/4] Instalando dependencias del sistema (supervisor, ffmpeg, git)..."
apt-get update && apt-get install -y supervisor ffmpeg git

# --- FASE 2: INSTALACIÓN DE NODOS Y DEPENDENCIAS DE PYTHON ---
echo ">>> [FASE 2/4] Instalando ComfyUI, Nodos Personalizados y librerías de Python..."
if [ ! -d "/workspace/ComfyUI" ]; then
    git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI
fi
CUSTOM_NODES_DIR="/workspace/ComfyUI/custom_nodes"
mkdir -p $CUSTOM_NODES_DIR
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-Manager" ]; then git clone https://github.com/ceutaseguridad/ComfyUI-Manager.git $CUSTOM_NODES_DIR/ComfyUI-Manager; fi
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-AnimateDiff-Evolved" ]; then git clone https://github.com/ceutaseguridad/ComfyUI-AnimateDiff-Evolved.git $CUSTOM_NODES_DIR/ComfyUI-AnimateDiff-Evolved; fi
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-VideoHelperSuite" ]; then git clone https://github.com/ceutaseguridad/ComfyUI-VideoHelperSuite.git $CUSTOM_NODES_DIR/ComfyUI-VideoHelperSuite; fi
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-wav2lip" ]; then git clone https://github.com/ceutaseguridad/ComfyUI-wav2lip.git $CUSTOM_NODES_DIR/ComfyUI-wav2lip; fi

echo "Instalando dependencias de ComfyUI (puede tardar)..."
/usr/bin/python3.11 -m pip install --no-cache-dir -r /workspace/ComfyUI/requirements.txt
echo "Instalando dependencias de Morpheus (unificadas)..."
/usr/bin/python3.11 -m pip install --no-cache-dir --ignore-installed -r /workspace/requirements.txt

# --- FASE 3: DESCARGA DE MODELOS DE IA ---
echo ">>> [FASE 3/4] Descargando modelos de IA..."
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

# --- FASE 4: EJECUCIÓN ---
echo ">>> [FASE 4/4] Configuración completa. Iniciando los servicios con Supervisor..."

# CORRECCIÓN FINAL: Crear el directorio de logs antes de llamar a supervisord
mkdir -p /workspace/logs

exec /usr/bin/supervisord -n -c /workspace/supervisor.conf
