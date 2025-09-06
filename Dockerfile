# Usar una imagen base de RunPod optimizada para PyTorch y CUDA
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Instalar dependencias del sistema operativo (ffmpeg para vídeo) y Supervisor (gestor de procesos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copiar solo el archivo de requerimientos para optimizar la caché de Docker
COPY requirements_pod.txt /

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r /requirements_pod.txt

# Supervisor necesita un directorio de logs para funcionar
RUN mkdir -p /var/log/supervisor

# Copiar nuestro archivo de configuración para Supervisor
COPY supervisor.conf /etc/supervisor/conf.d/morpheus.conf

# El comando que se ejecutará al iniciar el contenedor.
# Lanza Supervisor en modo "no-daemon", que es lo que RunPod necesita para monitorizar el proceso.
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
