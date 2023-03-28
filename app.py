from flask import Flask, request
from flask_restful import Api, Resource
import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel

app = Flask(__name__)
api = Api(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 512

class SpamClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SpamClassifier, self).__init__()
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

class SpamDetector(Resource):
    @staticmethod
    def post():
        data = request.get_json(force=True)
        msg = data['message']

        prediction = predict_spam(msg)
        response = {'prediction': prediction}

        return response, 200

def predict_spam(msg):
    model.eval()
    msg = [msg]
    encoding = tokenizer.encode_plus(
        msg,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    _, prediction = torch.max(output, dim=1)

    return prediction.item()
api.add_resource(SpamDetector, '/detect-spam')

if __name__ == '__main__':
    # Load the trained model
    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    model = SpamClassifier(n_classes=2)
    model.load_state_dict(torch.load('model/best_model_state.bin', map_location=torch.device('cpu')))
    model.eval()

    # Run the Flask app
    app.run(debug=True, port=5000)
