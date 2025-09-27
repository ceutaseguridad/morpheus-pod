#!/bin/bash

# --- Script de Configuración y Arranque Autónomo para Morpheus Pod v17.1 (con Logging Robusto) ---

# [CORRECCIÓN] Abortar el script inmediatamente si cualquier comando falla.
set -e

# --- [CORRECCIÓN 1: CONFIGURACIÓN DE LOGGING PERSISTENTE] ---
# Define un archivo de log central para todo el proceso de setup.
LOG_FILE="/workspace/logs/setup.log"
# Asegurarse de que el directorio de logs exista.
mkdir -p /workspace/logs
# Redirige toda la salida (stdout y stderr) a la consola Y al archivo de log.
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================================="
echo "==       INICIANDO SCRIPT DE ARRANQUE (setup.sh)        =="
echo "==      Log persistente disponible en ${LOG_FILE}       =="
echo "=========================================================="
echo "Fecha de inicio: $(date)"
echo ""

# --- FASE 1: INSTALACIÓN DE DEPENDENCIAS DEL SISTEMA ---
echo ">>> [FASE 1/6] Instalando dependencias del sistema (supervisor, ffmpeg, git)..."
apt-get update && apt-get install -y --no-install-recommends supervisor ffmpeg git
echo ">>> [FASE 1/6] Dependencias del sistema instaladas."
echo ""

# --- FASE 2: CLONACIÓN DE COMFYUI Y NODOS (USANDO FORKS CONTROLADOS) ---
echo ">>> [FASE 2/6] Instalando ComfyUI y Nodos Personalizados desde tus forks..."
if [ ! -d "/workspace/ComfyUI" ]; then
    echo "Directorio de ComfyUI no encontrado, clonando..."
    git clone https://github.com/ceutaseguridad/ComfyUI.git /workspace/ComfyUI
else
    echo "Directorio de ComfyUI ya existe."
fi
CUSTOM_NODES_DIR="/workspace/ComfyUI/custom_nodes"
mkdir -p $CUSTOM_NODES_DIR

# Clonar desde tus repositorios en ceutaseguridad
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-Manager" ]; then git clone https://github.com/ceutaseguridad/ComfyUI-Manager.git $CUSTOM_NODES_DIR/ComfyUI-Manager; fi
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-AnimateDiff-Evolved" ]; then git clone https://github.com/ceutaseguridad/ComfyUI-AnimateDiff-Evolved.git $CUSTOM_NODES_DIR/ComfyUI-AnimateDiff-Evolved; fi
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-VideoHelperSuite" ]; then git clone https://github.com/ceutaseguridad/ComfyUI-VideoHelperSuite.git $CUSTOM_NODES_DIR/ComfyUI-VideoHelperSuite; fi
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-wav2lip" ]; then git clone https://github.com/ceutaseguridad/ComfyUI-wav2lip.git $CUSTOM_NODES_DIR/ComfyUI-wav2lip; fi
echo ">>> [FASE 2/6] Clonación de nodos completada."
echo ""

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
echo ">>> [FASE 3/6] Recopilación de requisitos completada."
echo ""

# --- FASE 4: INSTALACIÓN DE DEPENDENCIAS DE PYTHON ---
echo ">>> [FASE 4/6] Instalando todas las dependencias de Python desde '$FINAL_REQ_FILE'..."
/usr/bin/python3.11 -m pip install \
    -v --no-cache-dir --upgrade --ignore-installed \
    --timeout 60 --retries 5 -r "$FINAL_REQ_FILE"
echo ">>> [FASE 4/6] Dependencias de Python instaladas con éxito."
echo ""

# --- FASE 5: DESCARGA DE MODELOS DESDE EL MANIFIESTO ---
echo ">>> [FASE 5/6] Descargando modelos de IA desde el manifiesto 'modelos.txt'..."
MODELS_FILE="/workspace/modelos.txt"
if [ -f "$MODELS_FILE" ]; then
    IFS=','
    while read -r target_dir url || [ -n "$target_dir" ]; do
        [[ -z "$target_dir" || "$target_dir" == \#* ]] && continue
        target_dir=$(echo "$target_dir" | xargs)
        url=$(echo "$url" | xargs)
        FILENAME=$(basename "$url")
        DEST_PATH="/workspace/ComfyUI/models/$target_dir"
        TARGET_FILE="${DEST_PATH}/${FILENAME}"

        mkdir -p "$DEST_PATH"
        echo "Procesando: ${FILENAME}"

        if [ -f "$TARGET_FILE" ]; then
            echo "  -> ESTADO: El archivo ya existe. Omitiendo descarga."
        else
            echo "  -> ESTADO: No encontrado. Descargando desde ${url}..."
            wget -nc --content-disposition -q --show-progress -P "$DEST_PATH" "$url"
            if [ $? -eq 0 ]; then
                echo "  -> DESCARGA COMPLETADA."
            else
                echo "  -> [ERROR FATAL] La descarga falló para ${url}."
                exit 1
            fi
        fi
    done < "$MODELS_FILE"
    echo ">>> [FASE 5/6] Descarga de modelos completada."
else
    echo ">>> [FASE 5/6] ADVERTENCIA: No se encontró el archivo '$MODELS_FILE'. No se descargarán modelos."
fi
echo ""

# --- FASE 6: EJECUCIÓN ---
echo ">>> [FASE 6/6] Configuración completa. Iniciando los servicios con Supervisor..."
# La creación del directorio de logs ya se hace al inicio del script, por lo que esta línea es redundante pero inofensiva.
mkdir -p /workspace/logs
/usr/bin/supervisord -c /workspace/supervisor.conf

# [CORRECCIÓN] Añadimos un mensaje final que solo se registrará si supervisord se lanza correctamente.
echo ""
echo "=========================================================="
echo "==     SCRIPT DE ARRANQUE FINALIZADO CON ÉXITO          =="
echo "=========================================================="
