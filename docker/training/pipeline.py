import os
import sagemaker
import mlflow
from sagemaker.pytorch import PyTorch
import boto3

def main():
    # Retrieve environment variables/secrets
    role = os.getenv('SAGE_MAKER_EXECUTION_ROLE')
    region = os.getenv('AWS_REGION', 'us-east-2')
    print('--------------------------------')
    print('--------------------------------')
    print('--------------------------------')
    print('--------------------------------')
    print(f"Region: {region}")
    print('--------------------------------')
    print('--------------------------------')
    print('--------------------------------')
    print('--------------------------------')
    bucket = os.getenv('S3_BUCKET')
    tracking_arn = os.getenv('MLFLOW_TRACKING_ARN', '')

    # Example: using environment variables for dataset version
    data_version = os.getenv('DATA_VERSION', '1')
    
    # Set up the SageMaker session
    sm_session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    
    mlflow.set_tracking_uri(tracking_arn)

    # Create a PyTorch Estimator (adjust the framework version, instance_type, etc. if needed)
    estimator = PyTorch(
        entry_point='train.py',
        source_dir='docker/training',
        role=role,
        sagemaker_session=sm_session,
        framework_version='2.0.0',
        py_version='py310',
        instance_count=1,
        instance_type='ml.m5.xlarge',
        hyperparameters={
            'epochs': '5',
            'batch_size': '32'
        },
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

if __name__ == '__main__':
    main() 