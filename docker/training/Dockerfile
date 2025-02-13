# Use a Miniconda base image (CPU-based). 
# If you need GPU support, consider an nvidia/cuda base image or pytorch/pytorch:latest-cuda.
FROM continuumio/miniconda3:latest

# Create a new conda environment.
RUN conda create --yes --name tennis-court python=3.9

# Switch your shell to the conda environment to install packages.
SHELL ["conda", "run", "-n", "tennis-court", "/bin/bash", "-c"]

# Install Python packages. Add or remove libraries as needed for your project.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      torch torchvision torchaudio \
      scikit-learn \
      pillow \
      boto3 \
      numpy \
      pandas \
      tqdm \
      requests \
      # add more libraries if your code needs them
      ipykernel

# Create a directory to hold your training script(s).
RUN mkdir -p /opt/ml/code

# Copy your training script into the container.
# In this example, we're calling it "train.py" and placing it in /opt/ml/code.
COPY docker/training/train.py /opt/ml/code/train.py

# Set the working directory.
WORKDIR /opt/ml/code

# The default SageMaker convention: SAGEMAKER_PROGRAM points to your training entry script
ENV SAGEMAKER_PROGRAM=train.py

# Expose conda environment to PATH so that the container can run your script with correct environment.
ENV PATH /opt/conda/envs/tennis-court/bin:$PATH

# Run your training script when the container starts.
ENTRYPOINT ["python", "train.py"]
