from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.logging import logger

from textSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline

from textSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline

STAGE_NAME = 'Data Ingestion'

try:
    logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise(e)

STAGE_NAME = 'Data Validation'

try:
    logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise(e)


STAGE_NAME = 'Data Transformation'

try:
    logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
    data_tranformation = DataTransformationTrainingPipeline()
    data_tranformation.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise(e)