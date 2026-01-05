from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.sklearn.processing import SKLearnProcessor
import sagemaker
 
def create_inference_pipeline(role, bucket):
    # Define input and model path
    input_data = f"s3://{bucket}/data/test.csv"
    model_path = f"s3://{bucket}/models/your_trained_model.tar.gz"
 
    # SKLearnProcessor for batch inference
    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type="ml.m5.large",
        instance_count=1
    )
 
    inference_step = ProcessingStep(
        name="BatchInference",
        processor=processor,
        code="scripts/batch_inference.py",
        arguments=[
            "--input", input_data,
            "--model", model_path,
            "--output", "/opt/ml/processing/output"
        ],
        outputs=[
            sagemaker.processing.ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=f"s3://{bucket}/predictions"
            )
        ],
    )
 
    # Define the pipeline
    pipeline = Pipeline(
        name="mlops-inference-pipeline",
        steps=[inference_step]
    )
 
    return pipeline
 
if __name__ == "__main__":
    # Setup SageMaker session and role
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    bucket = session.default_bucket()
 
    # Create and run the pipeline
    pipeline = create_inference_pipeline(role, bucket)
    pipeline.upsert(role_arn=role)
    pipeline.start()