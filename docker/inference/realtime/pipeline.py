import os
import sagemaker
import mlflow
from sagemaker.pytorch import PyTorchModel
import boto3
import json
import tempfile
import shutil
import tarfile
from urllib.parse import urlparse

def main():
    # Retrieve environment variables/secrets
    role = os.getenv('SAGE_MAKER_EXECUTION_ROLE')
    region = os.getenv('AWS_REGION', 'us-east-2')
    bucket = os.getenv('S3_BUCKET')
    tracking_arn = os.getenv('MLFLOW_TRACKING_ARN')
    ecr_registry = os.getenv('ECR_REGISTRY')
    ecr_repository = os.getenv('ECR_REPOSITORY')
    github_sha = os.getenv('GITHUB_SHA', 'latest')
    source_branch = os.getenv('SOURCE_BRANCH')

    if not all([role, region, bucket, ecr_registry, ecr_repository]):
        raise ValueError(
            "Missing required environment variables. Please ensure all required "
            "environment variables are set."
        )

    # Set up the SageMaker session
    sm_session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    
    if tracking_arn:
        mlflow.set_tracking_uri(tracking_arn)

    # Use the inference image that was built and pushed by GitHub Actions
    inference_image_uri = f"{ecr_registry}/{ecr_repository}:inference-{github_sha}"


    # Get the most recent MLflow run for this experiment
    runs = mlflow.search_runs(
        filter_string="tags.mlflow.runName = 'tennis-court-training'",
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if len(runs) == 0:
        raise Exception("No MLflow run found")
        
    run_id = runs.iloc[0].run_id
    print(f"Retrieved MLflow run ID: {run_id}")

    # Get the model artifact location from MLflow
    client = mlflow.tracking.MlflowClient()
    artifact_uri = client.get_run(run_id).info.artifact_uri
    model_path = f"{artifact_uri}/model"
    print(f"Original model artifact path: {model_path}")

    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the model files from S3
        parsed_uri = urlparse(model_path)
        s3_bucket = parsed_uri.netloc
        s3_key = parsed_uri.path.lstrip('/')
        
        s3_client = boto3.client('s3')
        local_model_path = os.path.join(temp_dir, 'model')
        os.makedirs(local_model_path)
        
        # Download all files from the model directory
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_key):
            for obj in page.get('Contents', []):
                key = obj['Key']
                local_file = os.path.join(temp_dir, os.path.relpath(key, os.path.dirname(s3_key)))
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                s3_client.download_file(s3_bucket, key, local_file)

        # Create tar.gz archive
        archive_path = os.path.join(temp_dir, 'model.tar.gz')
        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(local_model_path, arcname='.')

        # Upload the tar.gz to S3
        model_artifact_key = f"{os.path.dirname(s3_key)}/model.tar.gz"
        s3_client.upload_file(archive_path, s3_bucket, model_artifact_key)
        
        model_artifact = f"s3://{s3_bucket}/{model_artifact_key}"
        print(f"Uploaded compressed model to: {model_artifact}")


    # Create a PyTorch Model with the compressed model artifact
    model = PyTorchModel(
        model_data=model_artifact,
        role=role,
        image_uri=inference_image_uri,
        sagemaker_session=sm_session,
        # framework_version='2.0.1',
    )

    # Create a preview endpoint configuration name
    preview_endpoint_name = f"tennis-court-preview-{github_sha}"

    print(f"Deploying real-time preview endpoint: {preview_endpoint_name}")
    
    # Deploy the model to a preview endpoint
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',
        endpoint_name=preview_endpoint_name,
        wait=True
    )
    
    print(f"Real-time preview endpoint deployed and available at: {predictor.endpoint_name}")

    # Create and run a batch transform job
    # print("Setting up batch transform job...")
    
    # transformer = model.transformer(
    #     instance_count=1,
    #     instance_type='ml.m5.large',
    #     output_path=f"s3://{bucket}/preview-tests/output/{github_sha}/",
    #     strategy='MultiRecord',
    #     max_concurrent_transforms=1,
    #     use_spot_instances=True,  # Enable spot instances for transform jobs
    #     max_run=7200,
    #     max_wait=9000,
    # )

    # print(f"Starting batch transform job for PR preview...")
    
    # Run batch transformation
    # transformer.transform(
    #     data=f"s3://{bucket}/preview-tests/input/",
    #     content_type='application/json',  # Adjust based on your input format
    #     split_type='Line'  # Adjust based on your input format
    # )
    
    # Wait for the transform job to complete
    # print("Waiting for batch transform job to complete...")
    # transformer.wait()
    
    print(f"Preview deployment complete!")
    print(f"Real-time endpoint available at: {predictor.endpoint_name}")
    # print(f"Batch transform results available at: s3://{bucket}/preview-tests/output/{github_sha}/")

    # Store endpoint info in a file that can be used by subsequent steps or PR comments
    endpoint_info = {
        "endpoint_name": predictor.endpoint_name,
        "batch_output": f"s3://{bucket}/preview-tests/output/{github_sha}/",
        "model_artifact": model_artifact
    }
    
    # with open('preview_endpoints.json', 'w') as f:
    #     json.dump(endpoint_info, f)

if __name__ == '__main__':
    main() 