#!/bin-bash

# --- Script de Configuración y Arranque Autónomo para Morpheus Pod v15.0 (Lógica Simplificada) ---
# ESTRATEGIA: Al usar un manifiesto 'modelos.txt' con URLs controladas y estables (Hugging Face),
# se elimina la necesidad de lógica condicional. Un único comando de descarga universal es suficiente.

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
        # Saltar líneas vacías o comentarios
        [[ -z "$target_dir" || "$target_dir" == \#* ]] && continue
        
        # Limpiar espacios en blanco
        target_dir=$(echo "$target_dir" | xargs)
        url=$(echo "$url" | xargs)
        
        DEST_PATH="/workspace/ComfyUI/models/$target_dir"
        mkdir -p "$DEST_PATH"
        
        echo "Procesando URL: $url..."
        
        # Lógica de descarga universal y simplificada.
        # Ya que todos los enlaces son de Hugging Face, todos respetan la cabecera 'content-disposition',
        # lo que permite a wget determinar el nombre de archivo correcto automáticamente.
        wget -nc --content-disposition -P "$DEST_PATH" "$url"
        
    done < "$MODELS_FILE"
    echo "Descarga de modelos completada."
else
    echo "ADVERTENCIA: No se encontró el archivo '$MODELS_FILE'. No se descargarán modelos."
fi

# --- FASE 6: EJECUCIÓN ---
echo ">>> [FASE 6/6] Configuración completa. Iniciando los servicios con Supervisor..."
mkdir -p /workspace/logs
exec /usr/bin/supervisord -n -c /workspace/supervisor.conf
