# morpheus_lib/management_handler.py
import requests
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Rutas base dentro del pod donde se almacenarán los modelos
# Estas rutas deben coincidir con la configuración de ComfyUI si se usa.
LORA_MODELS_PATH = Path("/workspace/ComfyUI/models/loras")
RVC_MODELS_PATH = Path("/workspace/ComfyUI/models/rvc") # Asumiendo una carpeta custom para RVC

def handle_install(workflow_type, url, filename, update_status_callback):
    """
    Descarga un archivo desde una URL al directorio apropiado.
    """
    if workflow_type == "install_lora":
        target_dir = LORA_MODELS_PATH
    elif workflow_type == "install_rvc":
        target_dir = RVC_MODELS_PATH
    else:
        raise ValueError(f"Tipo de instalación desconocido: {workflow_type}")

    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / filename

    if file_path.exists():
        logger.warning(f"El archivo '{filename}' ya existe. Omitiendo descarga.")
        return str(file_path)

    try:
        logger.info(f"Iniciando descarga de '{url}' a '{file_path}'...")
        update_status_callback(progress=20, status_text="Descargando modelo...")

        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            bytes_downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bytes_downloaded += len(chunk)
                    if total_size > 0:
                        progress = 20 + int(70 * bytes_downloaded / total_size)
                        update_status_callback(progress=progress)

        logger.info(f"Descarga de '{filename}' completada.")
        update_status_callback(progress=100, status_text="Instalación completada.")
        return str(file_path)

    except Exception as e:
        logger.error(f"Fallo al descargar/instalar '{filename}': {e}", exc_info=True)
        # Si falla, limpiar el archivo parcialmente descargado
        if file_path.exists():
            os.remove(file_path)
        raise e # Re-lanzar la excepción para que el worker_server la capture
