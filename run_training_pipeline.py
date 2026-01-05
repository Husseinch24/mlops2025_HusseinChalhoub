import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.parameters import ParameterString
from sagemaker.processing import ProcessingOutput
from sagemaker.inputs import TrainingInput
 
def create_training_pipeline(role, bucket):
    """
    Create a SageMaker Pipeline for ML training
    """
    # Define input dataset path
    input_data = ParameterString(
        name="InputData",
        default_value=f"s3://{bucket}/data/train.csv"
    )
 
    # Set up the SKLearnProcessor for preprocessing and feature engineering
    processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        sagemaker_session=sagemaker.Session()
    )
 
    # Preprocessing step
    preprocess_step = ProcessingStep(
        name="Preprocess",
        processor=processor,
        code="scripts/preprocess.py",
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=input_data,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="preprocessed_data",
                source="/opt/ml/processing/output",
                destination=f"s3://{bucket}/processed"
            )
        ],
        job_arguments=[
            "--input", "/opt/ml/processing/input",
            "--output", "/opt/ml/processing/output"
        ]
    )
 
    # Feature engineering step
    feature_step = ProcessingStep(
        name="FeatureEngineering",
        processor=processor,
        code="scripts/feature_engineering.py",
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=preprocess_step.properties.ProcessingOutputConfig.Outputs["preprocessed_data"].S3Output.S3Uri,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="feature_data",
                source="/opt/ml/processing/output",
                destination=f"s3://{bucket}/features"
            )
        ],
        job_arguments=[
            "--input", "/opt/ml/processing/input",
            "--output", "/opt/ml/processing/output"
        ]
    )
 
    # Train the model
    estimator = SKLearn(
        entry_point="scripts/train.py",
        framework_version="0.23-1",
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        py_version="py3",
        output_path=f"s3://{bucket}/models",
        sagemaker_session=sagemaker.Session()
    )
 
    train_step = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=feature_step.properties.ProcessingOutputConfig.Outputs["feature_data"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )
 
    # Define the pipeline
    pipeline = Pipeline(
        name="mlops-training-pipeline",
        parameters=[input_data],
        steps=[preprocess_step, feature_step, train_step],
        sagemaker_session=sagemaker.Session()
    )
 
    return pipeline
 
 
if __name__ == "__main__":
    session = sagemaker.Session()
   
    role = "arn:aws:iam::993952121120:role/SageMakerExecutionRole"
   
    bucket = session.default_bucket()
   
    print(f"Using bucket: {bucket}")
    print(f"Using role: {role}")
   
    try:
        pipeline = create_training_pipeline(role, bucket)
       
        pipeline.upsert(role_arn=role)
        print("Pipeline created/updated successfully!")
       
        execution = pipeline.start()
        print(f"Pipeline execution started: {execution.arn}")
       
       
    except Exception as e:
        print(f"Error created: {str(e)}")
        