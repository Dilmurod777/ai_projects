import time
import requests
import json

from api_secrets import API_KEY_ASSEMBLYAI
from youtube_extractor import get_video_infos, get_audio_url


def transcribe(audio_url, transcript_endpoint, headers, sentiment_analysis=False):
    data = {
        "audio_url": audio_url,
        'sentiment_analysis': sentiment_analysis
    }
    response = requests.post(transcript_endpoint, json=data, headers=headers)
    return response.json()['id']


def poll(polling_endpoint, headers, interval=30):
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


def save(data, error, filename, sentiment_analysis=False):
    if error:
        print("Something went wrong! Error:", error)
    else:
        if sentiment_analysis:
            print(filename)
            with open(filename, 'w') as _file:
                sentiments = data['sentiment_analysis_results']
                json.dump(sentiments, _file, indent=4)
        else:
            with open(filename, 'w') as _file:
                _file.write(data['text'])

        print("Saved")


def save_video_sentiments(url, transcript_endpoint, headers):
    video_info = get_video_infos(url)
    audio_url = get_audio_url(video_info)

    job_id = transcribe(
        audio_url=audio_url,
        transcript_endpoint=transcript_endpoint,
        sentiment_analysis=True,
        headers=headers)
    data, error = poll(
        polling_endpoint=transcript_endpoint + '/' + job_id,
        headers=headers)
    save(
        data=data,
        error=error,
        filename='data/sentiment_results.json',
        sentiment_analysis=True)

    return 'data/sentiment_results.json'
