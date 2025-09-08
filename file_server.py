# file_server.py
from flask import Flask, request, jsonify, send_from_directory
import os
import uuid
import shutil
import logging

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(module)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuración ---
app = Flask(__name__)
# Directorio base para almacenar todos los archivos de los trabajos.
# Es crucial que esté en /workspace para ser persistente.
BASE_DIR = "/workspace/job_data"

# Asegurarse de que el directorio base exista al iniciar la aplicación.
# Esto es vital para que Gunicorn funcione correctamente.
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
    Si no se proporciona un ID, crea uno nuevo.
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
    logger.info(f"Recibida solicitud de subida para job_id: {worker_job_id}, filename: {file.filename}")

    job_input_dir = os.path.join(BASE_DIR, worker_job_id, "input")
    os.makedirs(job_input_dir, exist_ok=True)

    save_path = os.path.join(job_input_dir, file.filename)
    try:
        file.save(save_path)
        logger.info(f"Archivo '{file.filename}' subido con éxito para el Job [ID: {worker_job_id}] a '{save_path}'")
        return jsonify({
            "message": "Archivo subido con éxito",
            "pod_path": save_path,
            "worker_job_id": worker_job_id
        }), 200
    except Exception as e:
        logger.error(f"Error al guardar el archivo '{file.filename}' para el Job [ID: {worker_job_id}]: {e}", exc_info=True)
        return jsonify({"error": f"Error al guardar el archivo: {str(e)}"}), 500

@app.route('/download/<worker_job_id>', methods=['GET'])
def download_file(worker_job_id):
    """
    Descarga el archivo de resultado de un trabajo.
    Asume que cada trabajo tiene un directorio con al menos un archivo.
    """
    logger.info(f"Endpoint /download/{worker_job_id} llamado.")
    job_output_dir = os.path.join(BASE_DIR, worker_job_id, "output")

    if not os.path.exists(job_output_dir):
        logger.warning(f"No se encontró el directorio de salida para el job ID: {worker_job_id}")
        return jsonify({"error": "No se encontró el directorio de salida para este trabajo"}), 404

    files = os.listdir(job_output_dir)
    if not files:
        logger.warning(f"No se encontraron archivos de resultado en el directorio de salida para el job ID: {worker_job_id}")
        return jsonify({"error": "No se encontraron archivos de resultado en el directorio de salida"}), 404

    result_file = files[0]
    logger.info(f"Descargando resultado '{result_file}' para el Job [ID: {worker_job_id}]")
    try:
        return send_from_directory(job_output_dir, result_file, as_attachment=True)
    except Exception as e:
        logger.error(f"Error al descargar el archivo '{result_file}' para el Job [ID: {worker_job_id}]: {e}", exc_info=True)
        return jsonify({"error": f"Error al descargar el archivo: {str(e)}"}), 500

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

# El bloque __main__ se mantiene para pruebas locales, pero no es usado por gunicorn
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
