import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import warnings
warnings.filterwarnings('ignore')

import math

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
MAX_LEN=512
BATCH_SIZE=16

class ScamCollectionDataset(Dataset):
    def __init__(self, scam, msgs, tokenizer, max_len):
        self.msgs = msgs
        self.scam = scam
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.msgs)

    def __getitem__(self, i):
        msg = str(self.msgs[i])
        scam = self.scam[i]

        encoding = self.tokenizer.encode_plus(
            msg, 
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'msg': msg,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'scam': torch.tensor(scam, dtype=torch.long)
        }
        

def get_predictions(model, data_loader):
    print("1 Get predictions...")
    model = model.eval()
    print("2 Get predictions...")

    msgs = []
    predictions = []
    predictions_probs = []
    real_values = []

    print("3 Get predictions...")
    with torch.no_grad():
        print("30 Get predictions...")
        print("len of data_loader: ", len(data_loader))
        for d in data_loader:
            print("31 Get predictions...")
            msg = d['msg']
            input_ids = d['input_ids'].to(device)
            attention_masks = d['attention_mask'].to(device)
            scam = d['scam'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks
            )
            print("32 Get predictions...")

            _, preds = torch.max(outputs, dim=1)
            print("33 Get predictions...")

            probs = torch.nn.functional.softmax(outputs, dim=1)
            print("34 Get predictions...")

            msgs.extend(msg)
            predictions.extend(preds)
            predictions_probs.extend(probs)
            real_values.extend(scam)

    print("4 Get predictions...")
    predictions = torch.stack(predictions).cpu()
    print("5 Get predictions...")
    predictions_probs = torch.stack(predictions_probs).cpu()
    print("6 Get predictions...")
    real_values = torch.stack(real_values).cpu()
    print("7 Get predictions...")
    return msgs, predictions, predictions_probs, real_values

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = ScamCollectionDataset(
        scam=df['scam'].to_numpy(),
        msgs=df['msg'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

class ScamClassifier(nn.Module):
    def __init__(self, n_classes):
        super(ScamClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes) # self.out
    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[1]
        output = self.drop(pooled_output)
        return self.classifier(output) # out



if __name__ == '__main__':    

    device = torch.device('cpu')
    model_loaded = ScamClassifier(n_classes=2)
    model_loaded.load_state_dict(torch.load("model/best_model_state.bin", map_location=device))

    
    exit()
    device = torch.device('cpu')
    model_loaded = ScamClassifier(n_classes=2)
    model_loaded = model_loaded.to(device)
    model_loaded.load_state_dict(torch.load("model/best_model_state.bin", map_location=device))
    tokenizer_loaded = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    data_example_loaded = [
        ["Hi this is a test", 0],
        ["Santos coin follow", 1],
        ["Santos coin follow", 1],
        ["Santos coin follow", 1],
        ["Resources Elon! Mars or Bust?", 1],
        ["This seems dumb. So let’s say I’m an individual who wants to go get the Bitcoins, I have to pay to be taken to the moon, then I can’t just go off on my own to find them, I have to pay to be taken there with a suit, a crew, vehicles and all that. Then they would’ve taken multiple others who have also payed to be taken there at the same time. I don’t get it. Seems weird 😂", 0],
        ["This seems dumb. So let’s say I’m an individual who wants to go get the Bitcoins, I have to pay to be taken to the moon, then I can’t just go off on my own to find them, I have to pay to be taken there with a suit, a crew, vehicles and all that. Then they would’ve taken multiple others who have also payed to be taken there at the same time. I don’t get it. Seems weird 😂", 1],
        ["What's a king to a god, what's a god to a non beliver, who don't belive in..", 1],
        ["What's a king to a god, what's a god to a non beliver, who don't belive in..", 0],
        ["HI EVERYONE 𝒊𝒏 𝒍𝒂𝒔𝒕 1 𝒘𝒆𝒆𝒌 𝑰 𝒉𝒂𝒗𝒆 𝒎𝒂𝒅𝒆 𝒐𝒗𝒆𝒓 $29,000 𝒘𝒊𝒕𝒉 𝒋𝒖𝒔𝒕 𝒂𝒏 𝒊𝒏𝒗𝒆𝒔𝒕𝒎𝒆𝒏𝒕 𝒐𝒇 $1,300, 𝒊𝒕'𝒔 𝒂𝒄𝒕𝒖𝒂𝒍𝒍𝒚 𝒎𝒚 𝒇𝒊𝒓𝒔𝒕 𝒕𝒊𝒎𝒆 𝒐𝒇 𝒐𝒏𝒍𝒊𝒏𝒆 𝒕𝒓𝒂𝒅𝒆 𝒊𝒏𝒗𝒆𝒔𝒕𝒎𝒆𝒏𝒕 𝒂𝒏𝒅 𝒂𝒎 𝒈𝒍𝒂𝒅 𝑰 𝒆𝒂𝒓𝒏𝒆𝒅 𝒂𝒍𝒍 𝒕𝒉𝒂𝒏𝒌𝒔 𝒕𝒐 @invest_with_danielle_d entrepreneur 𝒂𝒏𝒅 her 𝒕𝒓𝒂𝒅𝒊𝒏𝒈 𝒕𝒆𝒂𝒎", 1],
        ["Waking up every day to see my balance increased is the best ever happended to me. Thank you so much @harry87", 1]
    ]
    print("6Loading model...")

    df_example_loaded = pd.DataFrame(data_example_loaded, columns=['msg', 'scam'])
    print("7Loading model...")
    example_data_loader_loaded = create_data_loader(df_example_loaded, tokenizer_loaded, MAX_LEN, BATCH_SIZE)
    print("8Loading model...")

    y_ex_msgs, y_ex_pred, y_ex_pred_probs, y_ex_test = get_predictions(
    model_loaded,
    example_data_loader_loaded
    )
    print("9Loading model...")

    for x in range(len(y_ex_msgs)): 
        print(round(y_ex_pred_probs[x][1].item(), 4), y_ex_msgs[x])



exit()


import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel, BertForSequenceClassification

# Set the device to CPU
device = 'cpu'

# Set the maximum length of the messages
max_len = 512

# define pretrained model name
PRE_TRAINED_MODEL_NAME = 'bert-base-uncas'

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
model.load_state_dict(torch.load('model/best_model_state.bin', map_location=torch.device('cpu')), strict=False)
#model.eval()

data_example_loaded = [
    ["Hi this is a test", 0],
    ["Santos coin follow", 1],
    ["Santos coin follow", 1],
    ["Santos coin follow", 1],
    ["Resources Elon! Mars or Bust?", 1],
    ["This seems dumb. So let’s say I’m an individual who wants to go get the Bitcoins, I have to pay to be taken to the moon, then I can’t just go off on my own to find them, I have to pay to be taken there with a suit, a crew, vehicles and all that. Then they would’ve taken multiple others who have also payed to be taken there at the same time. I don’t get it. Seems weird 😂", 0],
    ["This seems dumb. So let’s say I’m an individual who wants to go get the Bitcoins, I have to pay to be taken to the moon, then I can’t just go off on my own to find them, I have to pay to be taken there with a suit, a crew, vehicles and all that. Then they would’ve taken multiple others who have also payed to be taken there at the same time. I don’t get it. Seems weird 😂", 1],
    ["What's a king to a god, what's a god to a non beliver, who don't belive in..", 1],
    ["What's a king to a god, what's a god to a non beliver, who don't belive in..", 0],
    ["HI EVERYONE 𝒊𝒏 𝒍𝒂𝒔𝒕 1 𝒘𝒆𝒆𝒌 𝑰 𝒉𝒂𝒗𝒆 𝒎𝒂𝒅𝒆 𝒐𝒗𝒆𝒓 $29,000 𝒘𝒊𝒕𝒉 𝒋𝒖𝒔𝒕 𝒂𝒏 𝒊𝒏𝒗𝒆𝒔𝒕𝒎𝒆𝒏𝒕 𝒐𝒇 $1,300, 𝒊𝒕'𝒔 𝒂𝒄𝒕𝒖𝒂𝒍𝒍𝒚 𝒎𝒚 𝒇𝒊𝒓𝒔𝒕 𝒕𝒊𝒎𝒆 𝒐𝒇 𝒐𝒏𝒍𝒊𝒏𝒆 𝒕𝒓𝒂𝒅𝒆 𝒊𝒏𝒗𝒆𝒔𝒕𝒎𝒆𝒏𝒕 𝒂𝒏𝒅 𝒂𝒎 𝒈𝒍𝒂𝒅 𝑰 𝒆𝒂𝒓𝒏𝒆𝒅 𝒂𝒍𝒍 𝒕𝒉𝒂𝒏𝒌𝒔 𝒕𝒐 @invest_with_danielle_d entrepreneur 𝒂𝒏𝒅 her 𝒕𝒓𝒂𝒅𝒊𝒏𝒈 𝒕𝒆𝒂𝒎", 1],
    ["Waking up every day to see my balance increased is the best ever happended to me. Thank you so much @harry87", 1]
]
df_example_loaded = pd.DataFrame(data_example_loaded, columns=['msg', 'scam'])
example_data_loader_loaded = create_data_loader(df_example_loaded, tokenizer_loaded, MAX_LEN, BATCH_SIZE)

y_ex_msgs, y_ex_pred, y_ex_pred_probs, y_ex_test = get_predictions(
  model_loaded,
  example_data_loader_loaded
)

for x in range(len(y_ex_msgs)): 
  print(round(y_ex_pred_probs[x][1].item(), 4), y_ex_msgs[x])








exit()
# Define a function for evaluating a single comment text
def evaluate_comment(comment):
    encoded_comment = tokenizer.encode_plus(comment, add_special_tokens=True, max_length=max_len, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
    input_ids = encoded_comment['input_ids'].to(device)
    attention_mask = encoded_comment['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = torch.nn.functional.softmax(outputs[0], dim=1).cpu().numpy()
    pred = torch.argmax(outputs[0], dim=1).cpu().numpy()[0]
    prediction = 'Scam' if pred == 1 else 'Not Scam'
    probability = probs[0][1]

    return prediction, probability

# Test the function with an example comment
comment = "Santos coin follow"
prediction, probability = evaluate_comment(comment)
print(f"Comment: {comment}")
print(f"Prediction: {prediction}")
print(f"Probability: {probability:.2f}")
