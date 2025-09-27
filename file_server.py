# file_server.py (Versión 2.1 - Specific File Download)
from flask import Flask, request, jsonify, send_from_directory
import os
import uuid
import shutil
import logging
from werkzeug.utils import secure_filename

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(module)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuración ---
app = Flask(__name__)
# Directorio base para almacenar todos los archivos de los trabajos.
BASE_DIR = "/workspace/job_data"

# Asegurarse de que el directorio base exista al iniciar la aplicación.
os.makedirs(BASE_DIR, exist_ok=True)

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint simple para verificar que el servidor está vivo."""
    logger.info("Endpoint /health llamado.")
    return jsonify({"status": "file_server is running"}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Maneja la subida de archivos. Asocia el archivo con un ID de trabajo.
    """
    logger.info("Endpoint /upload llamado.")
    if 'file' not in request.files:
        logger.warning("No se recibió ninguna parte de archivo en la petición.")
        return jsonify({"error": "No se encontró ninguna parte de archivo"}), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning("No se seleccionó ningún archivo.")
        return jsonify({"error": "No se seleccionó ningún archivo"}), 400

    worker_job_id = request.form.get('worker_job_id') or str(uuid.uuid4())
    sub_dir = request.form.get('sub_dir')
    
    # [SEGURIDAD] Usar secure_filename para evitar nombres de archivo maliciosos.
    safe_filename = secure_filename(file.filename)
    
    logger.info(f"Recibida solicitud de subida para job_id: {worker_job_id}, filename: {safe_filename}, sub_dir: {sub_dir}")

    job_input_dir = os.path.join(BASE_DIR, worker_job_id, "input")
    
    if sub_dir:
        # Sanear el nombre del subdirectorio para evitar ataques de path traversal
        safe_sub_dir = "".join(c for c in sub_dir if c.isalnum() or c in ('_', '-')).rstrip()
        final_dir = os.path.join(job_input_dir, safe_sub_dir)
    else:
        final_dir = job_input_dir

    os.makedirs(final_dir, exist_ok=True)

    save_path = os.path.join(final_dir, safe_filename)
    try:
        file.save(save_path)
        logger.info(f"Archivo '{safe_filename}' subido con éxito para el Job [ID: {worker_job_id}] a '{save_path}'")
        return jsonify({
            "message": "Archivo subido con éxito",
            "pod_path": save_path,
            "worker_job_id": worker_job_id
        }), 200
    except Exception as e:
        logger.error(f"Error al guardar el archivo '{safe_filename}' para el Job [ID: {worker_job_id}]: {e}", exc_info=True)
        return jsonify({"error": f"Error al guardar el archivo: {str(e)}"}), 500

# --- [CORRECCIÓN: ENDPOINT DE DESCARGA ACTUALIZADO] ---
# La ruta ahora acepta un nombre de archivo específico.
# Se usa <path:filename> para permitir nombres de archivo que contienen puntos.
@app.route('/download/<worker_job_id>/<path:filename>', methods=['GET'])
def download_file(worker_job_id, filename):
    """
    Descarga un archivo de resultado específico de un trabajo.
    """
    logger.info(f"Endpoint /download/{worker_job_id}/{filename} llamado.")
    
    # [SEGURIDAD] Sanear el ID del trabajo y el nombre del archivo para evitar path traversal.
    safe_worker_job_id = secure_filename(worker_job_id)
    safe_filename = secure_filename(filename)

    job_output_dir = os.path.join(BASE_DIR, safe_worker_job_id, "output")

    if not os.path.exists(job_output_dir):
        logger.warning(f"No se encontró el directorio de salida para el job ID: {safe_worker_job_id}")
        return jsonify({"error": "No se encontró el directorio de salida para este trabajo"}), 404
    
    # send_from_directory es una función segura de Flask que previene el path traversal
    # fuera del directorio especificado.
    try:
        logger.info(f"Descargando resultado '{safe_filename}' para el Job [ID: {safe_worker_job_id}]")
        return send_from_directory(job_output_dir, safe_filename, as_attachment=True)
    except FileNotFoundError:
        logger.warning(f"No se encontró el archivo '{safe_filename}' en el directorio de salida para el job ID: {safe_worker_job_id}")
        return jsonify({"error": f"No se encontró el archivo de resultado '{safe_filename}'"}), 404
    except Exception as e:
        logger.error(f"Error al descargar el archivo '{safe_filename}' para el Job [ID: {safe_worker_job_id}]: {e}", exc_info=True)
        return jsonify({"error": f"Error interno al descargar el archivo: {str(e)}"}), 500


@app.route('/cleanup/<worker_job_id>', methods=['POST'])
def cleanup_job_files(worker_job_id):
    """
    Elimina el directorio completo y todos los archivos de un trabajo.
    """
    logger.info(f"Endpoint /cleanup/{worker_job_id} llamado.")
    job_dir = os.path.join(BASE_DIR, worker_job_id)

    if os.path.exists(job_dir):
        try:
            shutil.rmtree(job_dir)
            logger.info(f"Archivos para el Job [ID: {worker_job_id}] eliminados con éxito.")
            return jsonify({"message": f"Limpieza completada para el trabajo {worker_job_id}"}), 200
        except Exception as e:
            logger.error(f"Error al limpiar los archivos para el Job [ID: {worker_job_id}]: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    else:
        logger.warning(f"Se solicitó limpieza para un Job [ID: {worker_job_id}] no existente.")
        return jsonify({"message": "El directorio del trabajo no existía, no se necesita limpieza"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
