import time
import mlflow
from mlflow.tracking import MlflowClient
import boto3

# Set the MLflow tracking URI to the localhost server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Initialize MLflow client
client = MlflowClient()

# Initialize AWS Step Functions client
step_functions_client = boto3.client('stepfunctions')

# Step Function ARN for deployment
state_machine_arn = 'arn:aws:states:us-west-2:851725217119:stateMachine:DeploymentStateMachine'

# Function to check for new model version
def check_for_new_model_version(model_name, last_checked_timestamp):
    # Get the latest version in the "Production" stage
    versions = client.get_latest_versions(name=model_name, stages=["Production"])
    if versions:
        version = versions[0]  # Get the first version in the list
        if version.creation_timestamp > last_checked_timestamp:
            return version
    return None

def main():
    last_checked_timestamp = 0
    model_name = 'sample_model'

    while True:
        new_version = check_for_new_model_version(model_name, last_checked_timestamp)
        if new_version:
            print(f"New model version found: {new_version.version}")
            last_checked_timestamp = new_version.creation_timestamp

            # Trigger Deployment Step Function
            response = step_functions_client.start_execution(
                stateMachineArn=state_machine_arn,
                input='{}'
            )
            print(f"Deployment Step Function triggered: {response['executionArn']}")

        time.sleep(60)

if __name__ == "__main__":
    main()
