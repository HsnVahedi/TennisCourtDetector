FROM continuumio/miniconda3:latest
RUN conda install -y pip
RUN pip install torch torchvision pillow mlflow sagemaker-inference

# Copy your inference script
COPY inference.py /opt/ml/code/inference.py

# (Optional) environment variables so sagemaker-inference
# knows which script to run:
ENV SAGEMAKER_PROGRAM inference.py
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

EXPOSE 8080

ENTRYPOINT ["python", "-m", "sagemaker_inference"]
