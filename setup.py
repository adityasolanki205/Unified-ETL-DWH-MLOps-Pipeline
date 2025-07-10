#!/usr/bin/env python
# coding: utf-8

from setuptools import find_packages
from setuptools import setup

setup(
    name='ml-deployment',
    version='0.1',
    author='Aditya',
    author_email = 'Aditya',
    install_requires=[  "apache-beam[gcp]==2.64.0",
                        "numpy==1.24.4",
                        "scikit-learn==1.2.2",
                        "Cython==0.29.36",
                        "pandas==1.5.3",
                        "joblib==1.2.0",
                        "google-cloud-storage>=2.18.2,<3",
                        "google-cloud-pubsub",
                        "google-cloud-aiplatform",
                        "google-cloud-bigquery",
                        "google-cloud-bigquery-storage",
                        "googleapis-common-protos==1.70.0"
                     ],
    packages=find_packages(exclude=['data']),
    include_package_data=True,
    description='Dataflow sklearn Streaming',
    python_requires='>=3.7',
    zip_safe=False,
    url=''
)