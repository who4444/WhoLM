import uuid
import os
import boto3
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

S3_BUCKET = ""
AWS_REGION = ""

s3_client = boto3.client('s3', region_name = AWS_REGION)

class UploadRequest(BaseModel):
    filename: str
    content_type: str

@app.post("/api/get-upload-url")
async def generate_upload_url(request: UploadRequest):
    try:
        ext = os.path.splitext(request.file_name)[1].lower()
        if not ext:
            raise HTTPException(status_code=400, detail="File extension missing")
            
        object_name = f"uploads/{uuid.uuid4()}{ext}"

        presigned_url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': S3_BUCKET,
                'Key': object_name,
                'ContentType': request.content_type
            },
            ExpiresIn=300
        )

        return {
            "upload_url": presigned_url,
            "s3_key": object_name
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))