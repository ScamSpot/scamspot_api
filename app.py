import random
from flask import Flask, make_response
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
import torch
import torch.nn as nn

import os.path
import urllib.request
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader

from model.helper import ScamClassifier, predict_single_comment


app = Flask(__name__)
CORS(app)

api = Api(app)
parser = reqparse.RequestParser()

local_filename = 'model/best_model_state.bin'
if os.path.isfile(local_filename):
    pass
else:
    url = 'https://ig-scam-model.fra1.cdn.digitaloceanspaces.com/best_model_state.bin'

    with urllib.request.urlopen(url) as response, open(local_filename, 'wb') as out_file:
        print(f"Downloading {url} ...")
        
        # read and write data in 1MB chunks
        data_chunk = response.read(1024*1024)
        while data_chunk:
            out_file.write(data_chunk)
            data_chunk = response.read(1024*1024)
        print(f"Downloaded and saved {local_filename}")




PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = ScamClassifier(n_classes=2)
model = model.to(device)
model.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, output_attentions=True, output_hidden_states=True)

model.load_state_dict(torch.load("model/best_model_state.bin", map_location=device))
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model.to(device) # Add this line to move the model to the specified device




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

        return {"message": "API active!"}, 200

    def post(self):
        parser.add_argument("comment_id", type=str, required=True, help="comment_id is required")
        parser.add_argument("comment_text", type=str, required=True, help="comment_text is required")
        args = parser.parse_args()

        comment_id = args["comment_id"]
        comment_text = args["comment_text"]

        # Test the function with an example comment
        #comment = "Good reviews ofdaily from her techniques Almost €30,000 within the week payout on week days feels better @staci.elmafx"
        comment = comment_text
        
        testing_mode = False
        
        if testing_mode:
            rating = 0.5
            predicted_class = "scam"
        else:
            predicted_class, rating = predict_single_comment(model, tokenizer, comment)
            #print(f"Comment: {comment}")
            #print(f"Predicted class: {predicted_class}")
            #print(f"Confidence level: {confidence}")
        
        #score = int(round(rating*100)) #d

        # Access-Control-Allow-Origin
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = 'https://www.instagram.com'

        #print(score, comment_text)

        #return {"comment_id": comment_id, "score": score}, 201
        return {"comment_id": comment_id, "class": predicted_class}, 201


api.add_resource(ScamChecker, '/scam/')


if __name__ == "__main__":
  app.run(debug=True, host='0.0.0.0')



