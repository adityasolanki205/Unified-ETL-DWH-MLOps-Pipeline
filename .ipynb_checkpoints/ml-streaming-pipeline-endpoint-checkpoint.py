#!/usr/bin/env python
# coding: utf-8


import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import argparse
from google.cloud import pubsub_v1
from google.cloud import storage
import numpy as np
import joblib
import pickle
import json
from google.cloud import aiplatform

SCHEMA='Existing_account:STRING,Duration_month:INTEGER,Credit_history:STRING,Purpose:STRING,Credit_amount:FLOAT,Saving:STRING,Employment_duration:STRING,Installment_rate:INTEGER,Personal_status:STRING,Debtors:STRING,Residential_Duration:INTEGER,Property:STRING,Age:INTEGER,Installment_plans:STRING,Housing:STRING,Number_of_credits:INTEGER,Job:STRING,Liable_People:INTEGER,Telephone:STRING,Foreign_worker:STRING,Classification:INTEGER'

class Split(beam.DoFn):
    #This Function Splits the Dataset into a dictionary
    def process(self, element):
        Existing_account,Duration_month,Credit_history,Purpose,Credit_amount,Saving,Employment_duration,Installment_rate,Personal_status,Debtors,Residential_Duration,Property,Age,Installment_plans,Housing,Number_of_credits,Job,Liable_People,Telephone,Foreign_worker= element.split(' ')
        return [{
           'Existing_account': str(Existing_account),
            'Duration_month': int(Duration_month),
            'Credit_history': str(Credit_history),
            'Purpose': str(Purpose),
            'Credit_amount': int(Credit_amount),
            'Saving': str(Saving),
            'Employment_duration':str(Employment_duration),
            'Installment_rate': int(Installment_rate),
            'Personal_status': str(Personal_status),
            'Debtors': str(Debtors),
            'Residential_Duration': int(Residential_Duration),
            'Property': str(Property),
            'Age': int(Age),
            'Installment_plans':str(Installment_plans),
            'Housing': str(Housing),
            'Number_of_credits': int(Number_of_credits),
            'Job': str(Job),
            'Liable_People': int(Liable_People),
            'Telephone': str(Telephone),
            'Foreign_worker': str(Foreign_worker)
        }]
def Filter_Data(data):
    #This will remove rows the with Null values in any one of the columns
    return data['Purpose'] !=  'NULL' and len(data['Purpose']) <= 3  and  data['Property'] !=  'NULL' and data['Personal_status'] != 'NULL' and data['Existing_account'] != 'NULL' and data['Credit_amount'] != 'NULL' and data['Installment_plans'] != 'NULL'

def Convert_Datatype(data):
    #This will convert the datatype of columns from String to integers or Float values
    data['Duration_month'] = int(data['Duration_month']) if 'Duration_month' in data else None
    data['Credit_amount'] = float(data['Credit_amount']) if 'Credit_amount' in data else None
    data['Installment_rate'] = int(data['Installment_rate']) if 'Installment_rate' in data else None
    data['Residential_Duration'] = int(data['Residential_Duration']) if 'Residential_Duration' in data else None
    data['Age'] = int(data['Age']) if 'Age' in data else None
    data['Number_of_credits'] = int(data['Number_of_credits']) if 'Number_of_credits' in data else None
    data['Liable_People'] = int(data['Liable_People']) if 'Liable_People' in data else None
    return data

def call_vertex_ai(data):
    aiplatform.init(project='827249641444', location='asia-south1')
    feature_order = ['Existing_account', 'Duration_month', 'Credit_history', 'Purpose',
                 'Credit_amount', 'Saving', 'Employment_duration', 'Installment_rate',
                 'Personal_status', 'Debtors', 'Residential_Duration', 'Property', 'Age',
                 'Installment_plans', 'Housing', 'Number_of_credits', 'Job', 
                 'Liable_People', 'Telephone', 'Foreign_worker']
    endpoint = aiplatform.Endpoint(endpoint_name=f"projects/827249641444/locations/asia-south1/endpoints/6457541741091225600")
    features = [data[feature] for feature in feature_order]
    response = endpoint.predict(
        instances=[features]
    )
    
    prediction = response.predictions[0]
    data['Classification'] = int(prediction)
    return data

# def call_vertex_ai(data):
#     aiplatform.init(project='827249641444', location='asia-south1')
#     endpoints = aiplatform.Endpoint.list()
#     # Display endpoint info
#     for endpoint in endpoints:
#         print(f"Resource Name: {endpoint.resource_name}")
#         endpoint_created = endpoint.resource_name
    
#     feature_order = ['Existing_account', 'Duration_month', 'Credit_history', 'Purpose',
#                  'Credit_amount', 'Saving', 'Employment_duration', 'Installment_rate',
#                  'Personal_status', 'Debtors', 'Residential_Duration', 'Property', 'Age',
#                  'Installment_plans', 'Housing', 'Number_of_credits', 'Job', 
#                  'Liable_People', 'Telephone', 'Foreign_worker']
#     #endpoint = aiplatform.Endpoint(endpoint_name=f"projects/827249641444/locations/asia-south1/endpoints/6457541741091225600")
#     endpoint = aiplatform.Endpoint(endpoint_name=endpoint_created)
#     features = [data[feature] for feature in feature_order]
#     response = endpoint.predict(
#         instances=[features]
#     )
    
#     prediction = response.predictions[0]
#     data['Classification'] = int(prediction)
#     return data
    
def run(argv=None, save_main_session=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--project',
        dest='project',
        help='Project used for this Pipeline')
    parser.add_argument(
        '--bucket_name', 
        required=True, 
        help='The name of the bucket')
    parser.add_argument(
        '--input_subscription',
        help=('Input PubSub subscription of the form '
              '"projects/<PROJECT>/subscriptions/<SUBSCRIPTION>."'))
    parser.add_argument(
        '--input_topic',
        help=('Input PubSub topic of the form '
              '"projects/<PROJECT>/topics/<TOPIC>".'))
    known_args, pipeline_args = parser.parse_known_args(argv)
    options = PipelineOptions(pipeline_args)
    PROJECT_ID = known_args.project
    TOPIC = known_args.input_topic
    SUBSCRIPTION = known_args.input_subscription
    with beam.Pipeline(options=PipelineOptions()) as p:
        Encoded_data   = (p 
                       | 'Read data' >> beam.io.ReadFromPubSub(topic=TOPIC).with_output_types(bytes) )
        Data           = ( Encoded_data
                      | 'Decode' >> beam.Map(lambda x: x.decode('utf-8') ) )
        Parsed_data    = (Data 
                       | 'Parsing Data' >> beam.ParDo(Split()))
        filtered_data = (Parsed_data
                     | 'Filtering Data' >> beam.Filter(Filter_Data))
        Converted_data = (filtered_data
                       | 'Convert Datatypes' >> beam.Map(Convert_Datatype))
        Prediction   = (Converted_data
                        |'Get Inference' >> beam.Map(call_vertex_ai))
        output         = ( Prediction      
                       | 'Writing to bigquery' >> beam.io.WriteToBigQuery(
                       '{0}:GermanCredit.GermanCreditTable-streaming'.format(PROJECT_ID),
                       schema=SCHEMA,
                       write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND)
                      )
        
if __name__ == '__main__':
    run()