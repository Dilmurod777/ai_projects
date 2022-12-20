import json

import requests

from api_secrets import API_KEY_ASSEMBLYAI
from helpers import save_video_sentiments

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=e-kSGNzu0hM"
    headers = {'authorization': API_KEY_ASSEMBLYAI}
    transcript_endpoint = 'https://api.assemblyai.com/v2/transcript'

    sentiments_filename = save_video_sentiments(
        url=video_url,
        transcript_endpoint=transcript_endpoint,
        headers=headers)

    with open(sentiments_filename, 'r') as _file:
        data = json.load(_file)

    positives = []
    negatives = []
    neutrals = []

    for item in data:
        text = item['text']
        if item["sentiment"] == "POSITIVE":
            positives.append(text)
        elif item['sentiment'] == 'NEGATIVE':
            negatives.append(text)
        else:
            neutrals.append(text)

    n_positive = len(positives)
    n_negative = len(negatives)
    n_neutral = len(neutrals)
    print(f"Number of positives: {n_positive}")
    print(f"Number of negatives: {n_negative}")
    print(f"Number of neutrals: {n_neutral}")

    pos_ratio = n_positive / (n_positive + n_negative)
    print(f"Positive ratio: {pos_ratio:.3f}")
