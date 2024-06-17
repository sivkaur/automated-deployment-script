import os
import subprocess
import mlflow
from mlflow.deployments import get_deploy_client
import boto3
from datetime import datetime

# Set the parameters
region = 'us-west-2'
aws_id = '851725217119'
arn = 'arn:aws:iam::851725217119:role/ml_deployment'
app_name = 'deployed-model-application'
repository_name = 'mlflow-deployment-pyfunc'
model_name = f'{app_name}-model'
config_name = f'{app_name}-config'

# Model name in the registry
mlflow_model_name = "sample_model"
s3_bucket = 'sivleen-mlflow-artifacts'

# Initialize the ECR and SageMaker clients
ecr_client = boto3.client('ecr', region_name=region)
sagemaker_client = boto3.client('sagemaker', region_name=region)
s3_client = boto3.client('s3', region_name=region)

# Check if the repository exists; if not, create it
try:
    response = ecr_client.describe_repositories(repositoryNames=[repository_name])
    print(f"Repository '{repository_name}' already exists.")
except ecr_client.exceptions.RepositoryNotFoundException:
    response = ecr_client.create_repository(repositoryName=repository_name)
    print(f"Created repository '{repository_name}'.")

# Get the latest model version with the stage 'Production'
client = mlflow.tracking.MlflowClient()
versions = client.get_latest_versions(name=mlflow_model_name, stages=["Production"])

if not versions:
    raise ValueError("No model version found with the stage 'Production'.")

production_version = versions[0]
run_id = production_version.run_id

# Get the run to fetch the experiment ID
run = client.get_run(run_id)
experiment_id = run.info.experiment_id

# Define the S3 URI for the model artifacts
model_uri = f"{experiment_id}/{run_id}/artifacts/random-forest-model"

# Define the local path to download the artifact
artifact_path = f"/tmp/{model_uri}"

# Fetch the model artifacts from S3
def download_artifacts_from_s3(s3_bucket, model_uri, local_path):
    paginator = s3_client.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=s3_bucket, Prefix=model_uri):
        for content in result.get('Contents', []):
            key = content['Key']
            local_file_path = os.path.join(local_path, os.path.relpath(key, model_uri))
            if not os.path.exists(os.path.dirname(local_file_path)):
                os.makedirs(os.path.dirname(local_file_path))
            s3_client.download_file(s3_bucket, key, local_file_path)
            print(f"Downloaded {key} to {local_file_path}")

# Download the artifacts to the local path
download_artifacts_from_s3(s3_bucket, model_uri, artifact_path)

# Ensure Docker is running
try:
    subprocess.run(["docker", "info"], check=True)
except subprocess.CalledProcessError:
    raise EnvironmentError("Docker is not running. Please start Docker and try again.")

# Verify if the artifact path exists
if not os.path.exists(artifact_path):
    raise FileNotFoundError(f"Artifact path does not exist: {artifact_path}")

# Change to the artifact directory
os.chdir(artifact_path)
print(f"Changed directory to: {artifact_path}")

# Build the Docker container (without pushing)
build_result = subprocess.run(
    ["mlflow", "sagemaker", "build-and-push-container", "--no-push"],
    check=True,
    capture_output=True,
    text=True
)

# Print the build output for debugging
print("Build Output:\n", build_result.stdout)

# Construct a unique ECR image tag
unique_tag = datetime.now().strftime('%Y%m%d%H%M%S')
print("Unique tag: ", unique_tag)
image_tag = f"{aws_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}:{unique_tag}"

# Tag the Docker image
tag_result = subprocess.run(
    ["docker", "tag", "mlflow-pyfunc:latest", image_tag],
    check=True,
    capture_output=True,
    text=True
)

# Print the tag output for debugging
print("Tag Output:\n", tag_result.stdout)

# Push the Docker image to ECR
push_result = subprocess.run(
    ["docker", "push", image_tag],
    check=True,
    capture_output=True,
    text=True
)

# Print the push output for debugging
print("Push Output:\n", push_result.stdout)

# Return to the original directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Verify the image is in ECR
describe_images_result = subprocess.run([
    "aws", "ecr", "describe-images",
    "--repository-name", repository_name,
    "--image-ids", f"imageTag={unique_tag}",
    "--region", region
], check=True, capture_output=True, text=True)

# Print the describe images output for debugging
print("Describe Images Output:\n", describe_images_result.stdout)

# Function to delete SageMaker endpoint and related configurations
def delete_sagemaker_resources(endpoint_name, config_name, model_name):
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"Deleted endpoint: {endpoint_name}")
    except sagemaker_client.exceptions.ClientError as e:
        print(f"Endpoint {endpoint_name} does not exist or cannot be deleted: {e}")

    try:
        sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
        print(f"Deleted endpoint config: {config_name}")
    except sagemaker_client.exceptions.ClientError as e:
        print(f"Endpoint config {config_name} does not exist or cannot be deleted: {e}")

    try:
        sagemaker_client.delete_model(ModelName=model_name)
        print(f"Deleted model: {model_name}")
    except sagemaker_client.exceptions.ClientError as e:
        print(f"Model {model_name} does not exist or cannot be deleted: {e}")

# Delete existing SageMaker resources if they exist
delete_sagemaker_resources(app_name, config_name, model_name)

# Log the image being deployed
print(f"Deploying image to SageMaker: {image_tag}")

# Create the SageMaker deployment client
deploy_client = get_deploy_client(f"sagemaker://{region}")

# Deploy the model
try:
    deploy_client.create_deployment(
        name=app_name,
        model_uri=artifact_path,
        config={
            "image_url": image_tag,
            "execution_role_arn": arn,
            "region_name": region,
            "mode": "replace",
            "model_name": model_name,
            "endpoint_config_name": config_name,
        }
    )
    print(f"Model deployed successfully with run_id: {run_id}")
except Exception as e:
    print(f"Deployment failed: {e}")
