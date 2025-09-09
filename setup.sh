#!/bin/bash

# --- Script de Configuración y Arranque Autónomo para Morpheus Pod v17.0 (Versión Estable) ---
# ESTRATEGIA: Se implementa una recopilación inteligente de dependencias (FASE 3) y se corrige el
# nombre del directorio de clonación de 'comfyui_workflow_templates' para que coincida con su
# nombre de paquete Python, resolviendo el error de importación.

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
# [CORRECCIÓN CLAVE] El directorio de destino ahora usa guion bajo para coincidir con el nombre del paquete.
if [ ! -d "$CUSTOM_NODES_DIR/comfyui_workflow_templates" ]; then git clone https://github.com/ceutaseguridad/comfyui_workflow_templates.git $CUSTOM_NODES_DIR/comfyui_workflow_templates; fi

# --- FASE 3: RECOPILACIÓN INTELIGENTE DE DEPENDENCIAS DE PYTHON ---
echo ">>> [FASE 3/6] Recopilando y unificando TODOS los archivos de requisitos..."
COMFY_REQ_FILE="/workspace/ComfyUI/requirements.txt"
MORPHEUS_REQ_FILE="/workspace/requirements_pod.txt"
FINAL_REQ_FILE="/workspace/final_requirements.txt"
TEMP_REQ_FILE="/tmp/temp_requirements.txt"

# 1. Empieza con nuestro archivo base, que fija las versiones críticas.
cat "$MORPHEUS_REQ_FILE" > "$TEMP_REQ_FILE"

# 2. Añade las dependencias de ComfyUI, saneando SOLO la línea de torch, que ya hemos fijado.
grep -vE '^torch==' "$COMFY_REQ_FILE" >> "$TEMP_REQ_FILE"

# 3. Itera sobre CADA nodo personalizado, busca su requirements.txt y añádelo.
echo "Buscando archivos de requisitos en nodos personalizados..."
for dir in $CUSTOM_NODES_DIR/*/; do
    if [ -f "${dir}requirements.txt" ]; then
        echo "Añadiendo dependencias desde: ${dir}requirements.txt"
        cat "${dir}requirements.txt" >> "$TEMP_REQ_FILE"
    fi
done

# 4. Crea el archivo final eliminando duplicados y asegurando un formato limpio.
awk '!seen[$0]++' "$TEMP_REQ_FILE" > "$FINAL_REQ_FILE"
rm "$TEMP_REQ_FILE"
echo "Archivo de requisitos final y unificado creado en $FINAL_REQ_FILE."

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
