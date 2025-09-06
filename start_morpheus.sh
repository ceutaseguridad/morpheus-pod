#!/bin/bash

# --- Script de Arranque Inteligente para Morpheus Pod v1.1 (A Prueba de Fallos) ---
# Se añade 'set -e' para garantizar que el script se detenga si cualquier comando falla.

# Salir inmediatamente si un comando devuelve un estado de no-cero.
set -e

# Navega al directorio de trabajo
cd /workspace

echo ">>> [START] Iniciando secuencia de arranque de Morpheus..."

# Lógica robusta para clonar el repositorio si no existe, o actualizarlo si ya existe.
if [ -d ".git" ]; then
    echo ">>> [GIT] Repositorio existente encontrado. Actualizando con 'git pull'..."
    git reset --hard HEAD
    git pull
else
    echo ">>> [GIT] No se encontró repositorio. Clonando desde GitHub..."
    git clone https://github.com/ceutaseguridad/morpheus-pod.git .
fi

# Comprobación de que el setup.sh existe después de clonar/actualizar
if [ -f "setup.sh" ]; then
    echo ">>> [SETUP] 'setup.sh' encontrado. Transfiriendo control al script de configuración..."
    chmod +x /workspace/setup.sh
    bash /workspace/setup.sh
else
    echo ">>> [ERROR CRÍTICO] 'setup.sh' no se encontró. Abortando."
    exit 1
fi
