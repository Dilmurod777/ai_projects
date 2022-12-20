import os
import time
import requests
import json
import pprint

from api_secrets import API_KEY_ASSEMBLYAI, API_KEY_LISTENNOTES

FILES_FOLDER_NAME = 'files'

ASSEMBLYAI_HEADERS = {'authorization': API_KEY_ASSEMBLYAI}
TRANSCRIPT_ENDPOINT = 'https://api.assemblyai.com/v2/transcript'

LISTENNOTES_EPISODE_ENDPOINT = "https://listen-api.listennotes.com/api/v2/episodes"
LISTENNOTES_HEADERS = {'X-ListenAPI-Key': API_KEY_LISTENNOTES}


def get_episode_audio_url(episode_id):
    url = LISTENNOTES_EPISODE_ENDPOINT + '/' + episode_id
    response = requests.request('GET', url, headers=LISTENNOTES_HEADERS)
    data = response.json()

    audio_url = data['audio']
    episode_thumbnail = data['thumbnail']
    podcast_title = data['podcast']['title']
    episode_title = data['title']

    return audio_url, episode_thumbnail, podcast_title, episode_title


def transcribe(audio_url, auto_chapters=False):
    data = {
        "audio_url": audio_url,
        'auto_chapters': auto_chapters
    }
    response = requests.post(TRANSCRIPT_ENDPOINT, json=data, headers=ASSEMBLYAI_HEADERS)
    return response.json()['id']


def poll(job_id, interval=60):
    while True:
        print("Transcribing...")
        response = requests.get(TRANSCRIPT_ENDPOINT + '/' + job_id, headers=ASSEMBLYAI_HEADERS)
        response_json = response.json()

        if response_json['status'] == 'completed':
            print("Transcribed")
            return response_json, None
        elif response_json['status'] == 'error':
            return response_json, response_json['error']

        time.sleep(interval)


def save(episode_id):
    audio_url, episode_thumbnail, podcast_title, episode_title = get_episode_audio_url(episode_id)
    job_id = transcribe(audio_url, auto_chapters=True)
    data, error = poll(job_id)

    if FILES_FOLDER_NAME not in os.listdir('.'):
        os.mkdir(FILES_FOLDER_NAME)

    if error:
        print("Something went wrong! Error:", error)
    else:
        filename = FILES_FOLDER_NAME + '/' + episode_id + '.txt'
        with open(filename, 'w') as _file:
            _file.write(data['text'])

        chapters_filename = FILES_FOLDER_NAME + '/' + episode_id + '_chapters.json'
        with open(chapters_filename, 'w') as _file:
            chapters = data['chapters']
            episode_data = {
                'chapters': chapters,
                'episode_thumbnail': episode_thumbnail,
                'episode_title': episode_title,
                'podcast_title': podcast_title
            }

            json.dump(episode_data, _file, indent=4)
            print("Saved")


def get_clean_time(t):
    seconds = int((t / 1000) % 60)
    minutes = int((t / (1000 * 60)) % 60)
    hours = int((t / (1000 * 60 * 60)) % 24)
    if hours > 0:
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    else:
        return f'{minutes:02d}:{seconds:02d}'
