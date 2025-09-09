#!/bin/bash

# --- Script de Configuración y Arranque Autónomo para Morpheus Pod v14.0 (Coherencia de Nombres) ---
# ESTRATEGIA: Se añade lógica para forzar un nombre de archivo estándar al descargar el LoRA de Civitai,
# garantizando la coherencia entre el archivo descargado y el nombre referenciado en los workflows.

# Salir inmediatamente si un comando falla
set -e

# --- FASE 1: INSTALACIÓN DE DEPENDENCIAS DEL SISTEMA ---
echo ">>> [FASE 1/6] Instalando dependencias del sistema (supervisor, ffmpeg, git)..."
apt-get update && apt-get install -y --no-install-recommends supervisor ffmpeg git

# --- FASE 2: CLONACIÓN DE COMFYUI Y NODOS (USANDO FORKS CONTROLADOS) ---
echo ">>> [FASE 2/6] Instalando ComfyUI y Nodos Personalizados desde tus forks..."
if [ ! -d "/workspace/ComfyUI" ]; then
    git clone https://github.com/ceutaseguridad/ComfyUI.git /workspace/ComfyUI
fi
CUSTOM_NODES_DIR="/workspace/ComfyUI/custom_nodes"
mkdir -p $CUSTOM_NODES_DIR

# Clonar desde tus repositorios en ceutaseguridad
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-Manager" ]; then git clone https://github.com/ceutaseguridad/ComfyUI-Manager.git $CUSTOM_NODES_DIR/ComfyUI-Manager; fi
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-AnimateDiff-Evolved" ]; then git clone https://github.com/ceutaseguridad/ComfyUI-AnimateDiff-Evolved.git $CUSTOM_NODES_DIR/ComfyUI-AnimateDiff-Evolved; fi
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-VideoHelperSuite" ]; then git clone https://github.com/ceutaseguridad/ComfyUI-VideoHelperSuite.git $CUSTOM_NODES_DIR/ComfyUI-VideoHelperSuite; fi
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-wav2lip" ]; then git clone https://github.com/ceutaseguridad/ComfyUI-wav2lip.git $CUSTOM_NODES_DIR/ComfyUI-wav2lip; fi
if [ ! -d "$CUSTOM_NODES_DIR/comfyui-workflow-templates" ]; then git clone https://github.com/ceutaseguridad/comfyui_workflow_templates.git $CUSTOM_NODES_DIR/comfyui-workflow-templates; fi

# --- FASE 3: SANEAMIENTO Y UNIFICACIÓN DE DEPENDENCIAS DE PYTHON ---
echo ">>> [FASE 3/6] Saneando y unificando archivos de requisitos..."
COMFY_REQ_FILE="/workspace/ComfyUI/requirements.txt"
MORPHEUS_REQ_FILE="/workspace/requirements_pod.txt"
FINAL_REQ_FILE="/workspace/final_requirements.txt"

grep -vE 'torch|pytorch-cuda|extra-index-url|comfyui-workflow-templates|comfyui-embedded-docs' "$COMFY_REQ_FILE" > /tmp/saneado_comfy_reqs.txt
cat /tmp/saneado_comfy_reqs.txt "$MORPHEUS_REQ_FILE" > "$FINAL_REQ_FILE"
rm /tmp/saneado_comfy_reqs.txt
echo "Archivo de requisitos final y unificado creado."

# --- FASE 4: INSTALACIÓN DE DEPENDENCIAS DE PYTHON ---
echo ">>> [FASE 4/6] Instalando todas las dependencias de Python..."
/usr/bin/python3.11 -m pip install \
    -v --no-cache-dir --upgrade --ignore-installed \
    --timeout 60 --retries 5 -r "$FINAL_REQ_FILE"
echo "Dependencias de Python instaladas."

# --- FASE 5: DESCARGA DE MODELOS DESDE EL MANIFIESTO ---
echo ">>> [FASE 5/6] Descargando modelos de IA desde el manifiesto 'modelos.txt'..."
MODELS_FILE="/workspace/modelos.txt"
if [ -f "$MODELS_FILE" ]; then
    IFS=','
    while read -r target_dir url || [ -n "$target_dir" ]; do
        [[ -z "$target_dir" || "$target_dir" == \#* ]] && continue
        
        target_dir=$(echo "$target_dir" | xargs)
        url=$(echo "$url" | xargs)
        DEST_PATH="/workspace/ComfyUI/models/$target_dir"
        mkdir -p "$DEST_PATH"
        
        echo "Procesando URL: $url..."
        
        # Lógica especial para el LoRA de Civitai para forzar el nombre del archivo y garantizar coherencia.
        if [[ "$url" == *"/api/download/models/417971"* ]]; then
            FILENAME="detail_tweaker_xl.safetensors"
            echo "URL de Civitai detectada. Forzando nombre de archivo a: $FILENAME"
            # Usamos -O para especificar el nombre de archivo de salida.
            wget -nc -O "$DEST_PATH/$FILENAME" "$url"
        else
            # Lógica estándar para Hugging Face y GitHub que usan nombres de archivo predecibles.
            wget -nc --content-disposition -P "$DEST_PATH" "$url"
        fi
    done < "$MODELS_FILE"
    echo "Descarga de modelos completada."
else
    echo "ADVERTENCIA: No se encontró el archivo '$MODELS_FILE'. No se descargarán modelos."
fi

# --- FASE 6: EJECUCIÓN ---
echo ">>> [FASE 6/6] Configuración completa. Iniciando los servicios con Supervisor..."
mkdir -p /workspace/logs
exec /usr/bin/supervisord -n -c /workspace/supervisor.conf
