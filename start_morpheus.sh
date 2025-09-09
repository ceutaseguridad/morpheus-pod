#!/bin/bash

# --- Entrypoint Script for Morpheus Pod ---
# Este script es el único punto de entrada del contenedor.
# Centraliza la lógica de sincronización de código y la ejecución del setup.

set -e # Salir inmediatamente si un comando falla

echo "--- [Morpheus Entrypoint] Iniciando secuencia de arranque... ---"

# Navegar al directorio de trabajo
cd /workspace

# --- Lógica de Sincronización de Código (ahora dentro de un script robusto) ---
echo "Sincronizando el repositorio de Morpheus..."
if [ ! -d "/workspace/.git" ]; then
    echo "Directorio .git no encontrado. Clonando el repositorio..."
    # Clonar el contenido del repo en el directorio actual
    git clone https://github.com/ceutaseguridad/morpheus-pod.git .
else
    echo "Repositorio existente. Forzando actualización a la última versión de 'main'..."
    # Limpiar cualquier estado inconsistente y forzar la actualización
    git fetch origin
    git reset --hard origin/main
    git clean -fdx # Elimina cualquier archivo no rastreado (logs, temporales) de arranques anteriores
fi
echo "Sincronización de código completada."

# --- Ejecución del Script de Setup Principal ---
# Ahora que el código está garantizado, ejecutamos el setup.
echo "Lanzando el script de configuración principal (setup.sh)..."
bash /workspace/setup.sh

# El script setup.sh termina con "exec supervisord...", por lo que este script
# cederá el control y el contenedor se mantendrá vivo.
echo "--- [Morpheus Entrypoint] Secuencia de arranque finalizada. ---"
