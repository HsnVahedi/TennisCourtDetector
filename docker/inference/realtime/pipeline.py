import os
import sagemaker
import mlflow
from sagemaker.pytorch import PyTorchModel
import boto3
import json

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
    inference_image_uri = f"{ecr_registry}/{ecr_repository}:infer-{github_sha}"

    # Get the model artifact from the training job that ran on the source branch
    model_artifact = f"s3://{bucket}/models/{source_branch}/model.tar.gz"

    # Create a PyTorch Model
    model = PyTorchModel(
        model_data=model_artifact,
        role=role,
        image_uri=inference_image_uri,
        sagemaker_session=sm_session,
        framework_version='2.0.1',  # Adjust based on your PyTorch version
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