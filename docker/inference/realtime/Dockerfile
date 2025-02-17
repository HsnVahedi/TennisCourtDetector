FROM continuumio/miniconda3:latest

# Install Java and clean up
RUN apt-get update && \
    apt-get install -y openjdk-17-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Java environment variables (split into two ENV instructions)
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Set service ports
ENV PORT=8080 \
    SERVICE_PORT=8080

# Install pip and required packages
RUN conda install -y pip
RUN pip install torch torchvision pillow mlflow multi-model-server sagemaker-inference

# Copy your inference script
# COPY docker/inference/realtime/inference.py /opt/ml/code/inference.py
# COPY docker/inference/realtime/handler_service.py /opt/ml/code/handler_service.py
COPY docker/inference/realtime/entrypoint.py /opt/ml/code/entrypoint.py

# (Optional) environment variables so sagemaker-inference knows which script to run:
ENV SAGEMAKER_PROGRAM=inference.py \
    SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

EXPOSE 8080

ENTRYPOINT ["python", "/opt/ml/code/entrypoint.py"]
CMD ["serve"]
