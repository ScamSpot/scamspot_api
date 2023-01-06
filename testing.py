import json
import requests


API_TOKEN = "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"
MODEL = "svalabs/twitter-xlm-roberta-crypto-spam"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/" + MODEL
print(API_URL)
payload = {"inputs": "something good here"}
data = json.dumps(payload)
response = requests.request("POST", API_URL, headers=headers, data=data)
data = json.loads(response.content.decode("utf-8"))

print(data)

positive_score = None
negative_score = None
for d in data[0]:
    if d['label'] == 'HAM':
        positive_score = d['score']
    elif d['label'] == 'SPAM':
        negative_score = d['score']

# Print the scores
print("Positive score: ", positive_score)
print("Negative score: ", negative_score)

score = 100 - (negative_score * 100)
# round score to 0 decimal places down
score = int(score)

print(score)
