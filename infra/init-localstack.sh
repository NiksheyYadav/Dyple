#!/bin/bash
# Initialize LocalStack S3 bucket

echo "Creating S3 bucket..."
awslocal s3 mb s3://dyple-storage
echo "S3 bucket 'dyple-storage' created successfully"
