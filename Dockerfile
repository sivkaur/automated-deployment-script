FROM python:3.8

# Install necessary packages
RUN pip install mlflow boto3 pymysql

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set environment variables
ENV BACKEND_STORE_URI sqlite:///mlflow.db
ENV ARTIFACT_ROOT s3://sivleen-mlflow-artifacts

# Expose the default MLflow port
EXPOSE 5000

# Entrypoint
ENTRYPOINT ["/entrypoint.sh"]
