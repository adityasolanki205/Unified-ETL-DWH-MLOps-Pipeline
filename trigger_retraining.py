#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import base64
import json
from google.cloud import aiplatform
from google.cloud.aiplatform import PipelineJob

PROJECT_ID = "solar-dialect-264808"                     # <---CHANGE THIS
REGION = "asia-south1"                            # <---CHANGE THIS
PIPELINE_ROOT ="gs://demo_bucket_kfl/pipeline_root_demo" # <---CHANGE THIS

def subscribe(event, context):
    """Triggered from a message on a Cloud Pub/Sub topic.
    Args:
            event (dict): Event payload.
            context (google.cloud.functions.Context): Metadata for the event.
    """
    # decode the event payload string
    payload_message = base64.b64decode(event['data']).decode('utf-8')
    # parse payload string into JSON object
    payload_json = json.loads(payload_message)
    # trigger pipeline run with payload
    trigger_retraining(payload_json)

def trigger_retraining(payload_json):
    """Triggers a pipeline run
    Args:
            payload_json: expected in the following format:
            {
                "pipeline_spec_uri": "<path-to-your-compiled-pipeline>",
                "parameter_values": {
                "greet_name": "<any-greet-string>"
                }
            }
    """
    pipeline_spec_uri = "gs://demo_bucket_kfl/training_pipeline.json"
    
    # Create a PipelineJob using the compiled pipeline from pipeline_spec_uri
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
    )
    
    pipeline_job = PipelineJob(
        display_name="retraining_pipeline_job",
        template_path=pipeline_spec_uri,
        pipeline_root=PIPELINE_ROOT
        )
    pipeline_job.run()

