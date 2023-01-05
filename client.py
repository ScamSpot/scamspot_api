# post request to http://127.0.0.1:5000/scam/ with comment_id and comment_text as parameters

import requests

url = "http://127.0.0.1:5000/scam/"

payload = {'comment_id': '123456', 'comment_text': 'This is a test comment'}
headers = {'Content-Type': 'application/json'}

response = requests.post(url, json=payload, headers=headers)

print(response.status_code)
print(response.text)
