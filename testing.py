import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Set the device to CPU
device = 'cpu'

# Set the maximum length of the messages
max_len = 50

# Load the saved model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_attentions=False, output_hidden_states=False)
model.load_state_dict(torch.load('model/model.pt', map_location=torch.device('cpu')))
model.eval()

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Define a function for evaluating a single comment text
def evaluate_comment(comment):
    encoded_comment = tokenizer.encode_plus(comment, add_special_tokens=True, max_length=max_len, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
    input_ids = encoded_comment['input_ids'].to(device)
    attention_mask = encoded_comment['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = torch.nn.functional.softmax(outputs[0], dim=1).cpu().numpy()
    pred = torch.argmax(outputs[0], dim=1).cpu().numpy()[0]
    prediction = 'Spam' if pred == 1 else 'Not Spam'
    probability = probs[0][1]

    return prediction, probability

# Test the function with an example comment
comment = "Thats awsomeee . Sue @bitboy_crypto"
prediction, probability = evaluate_comment(comment)
print(f"Comment: {comment}")
print(f"Prediction: {prediction}")
print(f"Probability: {probability:.2f}")

exit(0)

######################################################

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
