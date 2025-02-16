import os
import sagemaker
import mlflow
from sagemaker.pytorch import PyTorch
import boto3

def main():
    # Retrieve environment variables/secrets
    role = os.getenv('SAGE_MAKER_EXECUTION_ROLE')
    region = os.getenv('AWS_REGION', 'us-east-2')
    bucket = os.getenv('S3_BUCKET')
    tracking_arn = os.getenv('MLFLOW_TRACKING_ARN')
    ecr_registry = os.getenv('ECR_REGISTRY')
    ecr_repository = os.getenv('ECR_REPOSITORY')
    github_sha = os.getenv('GITHUB_SHA', 'latest')
    # run_id = os.getenv('MLFLOW_RUN_ID')

    if not all([role, region, bucket, ecr_registry, ecr_repository, tracking_arn]):
        raise ValueError(
            "Missing required environment variables. Please ensure all required "
            "environment variables are set: SAGE_MAKER_EXECUTION_ROLE, AWS_REGION, "
            "S3_BUCKET, ECR_REGISTRY, ECR_REPOSITORY, MLFLOW_TRACKING_ARN"
        )

    # if not run_id:
    #     raise ValueError("MLFLOW_RUN_ID environment variable is required")

    # Example: using environment variables for dataset version
    data_version = '2'
    
    # Set up the SageMaker session
    sm_session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    
    mlflow.set_tracking_uri(tracking_arn)

    # Use the training image that was built and pushed by GitHub Actions
    custom_image_uri = f"{ecr_registry}/{ecr_repository}:train-{github_sha}"

    # Create a PyTorch Estimator (adjust the framework version, instance_type, etc. if needed)
    estimator = PyTorch(
        entry_point='train.py',
        source_dir='.',
        role=role,
        sagemaker_session=sm_session,
        image_uri=custom_image_uri,
        instance_count=1,
        instance_type='ml.m5.large',
        use_spot_instances=True,
        max_run=7200,
        max_wait=9000,
        hyperparameters={
            'epochs': '5',
            'batch_size': '32'
        },
        environment={
            'MLFLOW_TRACKING_ARN': tracking_arn,
            # 'AWS_REGION': region,
            # 'S3_BUCKET': bucket,
        }
    )

    # Example channel mapping from your existing workflow:
    train_prefix = f's3://{bucket}/datasets/{data_version}/training/'
    val_prefix   = f's3://{bucket}/datasets/{data_version}/validation/'

    estimator.fit(
        inputs={
            'train': train_prefix,
            'validation': val_prefix
        }
    )

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

    # Now you can use the run_id for model deployment
    model_uri = f"runs:/{run_id}/model"
    
    # Deploy the model
    # model = mlflow.sagemaker.Model(
    #     model_uri=model_uri,
    #     role=role,
    #     image_uri=custom_image_uri,
    #     sagemaker_session=sm_session
    # )
    # predictor = model.deploy(
    #     initial_instance_count=1,
    #     instance_type='ml.m5.large',
    #     endpoint_name=preview_endpoint_name
    # )

if __name__ == '__main__':
    main() 