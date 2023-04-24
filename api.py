import random
from flask import Flask, make_response
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
import json
import requests
import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader


app = Flask(__name__)
CORS(app)

api = Api(app)
parser = reqparse.RequestParser()

# Set the device to CPU
device = 'cpu'

# Set the maximum length of the messages
max_len = 512

# define pretrained model name
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=True)

class ScamClassifier(nn.Module):
    def __init__(self, n_classes):
        super(ScamClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[1]
        output = self.drop(pooled_output)
        return self.out(output)


# Load the saved model
model = ScamClassifier(n_classes=2)
#model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_attentions=False, output_hidden_states=False)
model.load_state_dict(torch.load('model/best_model_state.bin', map_location=torch.device('cpu'))) #, strict=False
model.eval()





@app.after_request
def set_csp_header(response):
    response.headers['Content-Security-Policy'] = "connect-src http://127.0.0.1:5000 *.facebook.com facebook.com *.fbcdn.net *.facebook.net wss://*.facebook.com:* ws://localhost:* blob: *.instagram.com *.cdninstagram.com wss://*.instagram.com:* 'self' *.teststagram.com wss://edge-chat.instagram.com connect.facebook.net"
    return response


@app.route('/')
def hello_world():
    return 'Hello, World! Scam Checker API'



class ScamChecker(Resource):
    def get(self):
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = 'https://www.instagram.com'

        return {"message": "Hello, World!"}, 200

    def post(self):
        parser.add_argument("comment_id", type=str, required=True, help="comment_id is required")
        parser.add_argument("comment_text", type=str, required=True, help="comment_text is required")
        args = parser.parse_args()

        comment_id = args["comment_id"]
        comment_text = args["comment_text"]

        # Test the function with an example comment
        #comment_text = "SANTOS coin time follow today"
        prediction, probability = evaluate_comment(comment_text)
        print(f"Comment: {comment_text}")
        print(f"Prediction: {prediction}")
        print(f"Probability: {probability:.2f}")
        score = int(round(probability*100))

        # Access-Control-Allow-Origin
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = 'https://www.instagram.com'

        #print(score, comment_text)

        return {"comment_id": comment_id, "score": score}, 201


api.add_resource(ScamChecker, '/scam/')


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



if __name__ == "__main__":
  app.run(debug=True, host='0.0.0.0')



