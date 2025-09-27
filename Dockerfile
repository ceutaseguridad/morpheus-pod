# Usar una imagen base de RunPod optimizada para PyTorch y CUDA
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Instalar dependencias del sistema operativo (ffmpeg para vídeo) y Supervisor (gestor de procesos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# [CORRECCIÓN 1] Crear el directorio centralizado para todos los logs.
# Esto asegura que exista antes de que Supervisor o cualquier otro script intente usarlo.
RUN mkdir -p /workspace/logs

# Copiar solo el archivo de requerimientos para optimizar la caché de Docker
COPY requirements_pod.txt /

# [CORRECCIÓN 2] Instalar las dependencias de Python con logging mejorado.
# Los 'echo' añaden marcadores claros en los logs de build de RunPod.
RUN echo "--- INICIANDO INSTALACIÓN DE DEPENDENCIAS PIP ---" && \
    pip install --no-cache-dir -r /requirements_pod.txt && \
    echo "--- INSTALACIÓN DE DEPENDENCIAS PIP COMPLETADA CON ÉXITO ---"

# Supervisor necesita un directorio de logs para funcionar (diferente a los logs de los programas)
RUN mkdir -p /var/log/supervisor

# Copiar nuestro archivo de configuración para Supervisor
COPY supervisor.conf /etc/supervisor/conf.d/morpheus.conf

# El comando que se ejecutará al iniciar el contenedor.
# Lanza Supervisor en modo "no-daemon", que es lo que RunPod necesita para monitorizar el proceso.
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
