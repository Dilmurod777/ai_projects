import time

import requests


def read_file(fn, chunk_size=5242880):
    with open(fn, 'rb') as _file:
        while True:
            chunk = _file.read(chunk_size)
            if not chunk:
                break
            yield chunk


def upload(filename, upload_endpoint, headers):
    response = requests.post(upload_endpoint,
                             headers=headers,
                             data=read_file(filename))

    print("Uploaded")
    return response.json()['upload_url']


def transcribe(audio_url, transcript_endpoint, headers):
    json = {"audio_url": audio_url}
    response = requests.post(transcript_endpoint, json=json, headers=headers)
    return response.json()['id']


def poll(job_id, polling_endpoint, headers, interval=30):
    while True:
        print("Transcribing...")
        response = requests.get(polling_endpoint, headers=headers)
        response_json = response.json()

        if response_json['status'] == 'completed':
            print("Transcribed")
            return response_json, None
        elif response_json['status'] == 'error':
            return response_json, response_json['error']

        time.sleep(interval)


def save(data, error, filename):
    if error:
        print("Something went wrong! Error:", error)
    else:
        with open(filename, 'w') as _file:
            _file.write(data['text'])

        print("Saved")
