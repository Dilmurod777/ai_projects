import time

import requests
import sys

from api_secrets import API_KEY_ASSEMBLYAI
from helpers import upload, transcribe, poll, save

audio_filename = sys.argv[1] if len(sys.argv) >= 2 else './files/output.m4a'
text_filename = sys.argv[2] if len(sys.argv) >= 3 else './files/output.txt'
upload_endpoint = 'https://api.assemblyai.com/v2/upload'
transcript_endpoint = 'https://api.assemblyai.com/v2/transcript'

headers = {'authorization': API_KEY_ASSEMBLYAI}

audio_url = upload(
    filename=audio_filename,
    upload_endpoint=upload_endpoint,
    headers=headers)
job_id = transcribe(
    audio_url=audio_url,
    transcript_endpoint=transcript_endpoint,
    headers=headers)
data, error = poll(
    job_id=job_id,
    polling_endpoint=transcript_endpoint + '/' + job_id,
    headers=headers)
save(data=data,
     error=error,
     filename=text_filename)
