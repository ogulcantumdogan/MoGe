FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git python3-pip libgl1-mesa-glx libglib2.0-0 \
      libsm6 libxext6 libxrender-dev libassimp-dev openssh-server curl && \
    rm -rf /var/lib/apt/lists/*

# Install vast-cli for instance control
RUN pip install vast-ai-tools

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
# Install all requirements explicitly to ensure compatibility
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install fastapi uvicorn gunicorn gradio opencv-python

# Copy application code
COPY . /workspace/

# Create temp directory with appropriate permissions
RUN mkdir -p /tmp/moge && chmod 777 /tmp/moge

# Setup SSH for vast.ai access
RUN mkdir -p /var/run/sshd && \
    echo 'PasswordAuthentication no' >> /etc/ssh/sshd_config && \
    echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && \
    echo 'AuthorizedKeysFile .ssh/authorized_keys' >> /etc/ssh/sshd_config

# Expose API port, Gradio UI port, and SSH port
EXPOSE 8000 7860 22

# Create startup script with proper error handling
RUN echo '#!/bin/bash\n\
# Start SSH server\n\
/usr/sbin/sshd\n\
\n\
# Set up workspace directory permissions\n\
chmod -R 777 /workspace\n\
\n\
echo "Starting FastAPI server on port 8000..."\n\
gunicorn server:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 --workers 1 --timeout 120 &\n\
FASTAPI_PID=$!\n\
echo "Starting Gradio UI on port 7860..."\n\
python scripts/app.py --share=false &\n\
GRADIO_PID=$!\n\
# Monitor both processes\n\
wait $FASTAPI_PID $GRADIO_PID\n\
' > /workspace/start.sh && chmod +x /workspace/start.sh

# Make entrypoint.sh executable
COPY entrypoint.sh /workspace/
RUN chmod +x /workspace/entrypoint.sh

# Set the entrypoint to our wrapper script
ENTRYPOINT ["/workspace/entrypoint.sh"]

# Run both services
CMD ["/workspace/start.sh"]
