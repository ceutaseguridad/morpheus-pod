#!/bin/bash

# --- Script de Configuración y Arranque Autónomo para Morpheus Pod v7.3 (Corrección de requirements.txt) ---
# ESTRATEGIA:
# 1. Se descarga la última versión de 'morpheus-pod' y se sobreescribe el contenido de /workspace.
# 2. Se clonan todos los nodos desde los forks de 'ceutaseguridad'.
# 3. Los modelos de Wav2Lip se descargan desde el fork verificado.
# 4. Se corrige el nombre del archivo de dependencias a 'requirements_pod.txt'.

# Salir inmediatamente si un comando falla
set -e

# --- FASE 0: AUTO-ACTUALIZACIÓN DEL ENTORNO DEL POD ---
echo ">>> [FASE 0/5] Descargando y sobreescribiendo la última versión del pod desde GitHub..."
REPO_URL="https://github.com/ceutaseguridad/morpheus-pod/archive/refs/heads/main.zip"
wget -O /workspace/morpheus-pod.zip "$REPO_URL"
apt-get update && apt-get install -y supervisor ffmpeg git unzip
unzip -o /workspace/morpheus-pod.zip -d /workspace
cp -rf /workspace/morpheus-pod-main/. /workspace/
rm -f /workspace/morpheus-pod.zip
rm -rf /workspace/morpheus-pod-main
echo ">>> Entorno del pod actualizado a la última versión."

# --- FASE 1: INSTALACIÓN DE DEPENDENCIAS DEL SISTEMA ---
echo ">>> [FASE 1/5] Dependencias del sistema (supervisor, ffmpeg, git, unzip) instaladas."

# --- FASE 2: INSTALACIÓN DE NODOS Y DEPENDENCIAS DE PYTHON ---
echo ">>> [FASE 2/5] Instalando ComfyUI, Nodos Personalizados y librerías de Python..."
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
# --- !! CORRECCIÓN APLICADA AQUÍ !! ---
# Apuntamos al archivo correcto 'requirements_pod.txt'
/usr/bin/python3.11 -m pip install --no-cache-dir --ignore-installed -r /workspace/requirements_pod.txt

# --- FASE 3: DESCARGA DE MODELOS DE IA ---
echo ">>> [FASE 3/5] Descargando modelos de IA..."
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
echo ">>> [FASE 4/5] Configuración completa. Iniciando los servicios con Supervisor..."
mkdir -p /workspace/logs
exec /usr/bin/supervisord -n -c /workspace/supervisor.conf
