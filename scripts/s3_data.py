# this file connect s3 aws and read file from s3 bucket
import os
import io
import boto3
import logging
import docx
import pypdf
from dotenv import load_dotenv

# define logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()

# example s3 env
s3_bucket = os.getenv("S3_BUCKET", "raw-rag-data")
s3_key = os.getenv("S3_KEY", "raw/Thẻ tín dụng VNBank StepUp Mastercard.docx")
s3_region = os.getenv("S3_REGION", "ap-south-1")

# connect and read file from s3 bucket
def read_file_from_s3(bucket_name, key, region):
    try:
        # load environment variables from .env
        load_dotenv()
        
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        client_kwargs = {"region_name": region}
        if aws_access_key_id and aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key
            
        # create s3 client
        s3 = boto3.client("s3", **client_kwargs)
        # get object from s3 bucket
        response = s3.get_object(Bucket=bucket_name, Key=key)
        
        # read raw binary bytes
        file_bytes = response["Body"].read()
        
        # extract text based on file type
        key_lower = key.lower()
        if key_lower.endswith(".docx"):
            doc = docx.Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs)
        elif key_lower.endswith(".pdf"):
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages_text = [page.extract_text() for page in reader.pages if page.extract_text()]
            return "\n".join(pages_text)
        else:
            # plain text / markdown / json / csv
            return file_bytes.decode("utf-8", errors="ignore")
            
    except Exception as e:
        logger.error(f"Error reading file from S3: {e}")
        return None
    
file_content = read_file_from_s3(s3_bucket, s3_key, s3_region)

import sys

if file_content:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
    print(file_content)