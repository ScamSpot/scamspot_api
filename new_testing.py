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
device_loaded = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ScamClassifier(nn.Module):
    def __init__(self, n_classes):
        super(ScamClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes) # self.classifier
    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[1]
        output = self.drop(pooled_output)
        return self.out(output) # self.classifier

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


def create_data_loader_loaded(df, tokenizer_loaded, max_len, batch_size):
    ds = ScamCollectionDataset(
        scam=df['scam'].to_numpy(),
        msgs=df['msg'].to_numpy(),
        tokenizer=tokenizer_loaded,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )
    
def get_predictions_loaded(model_loaded, data_loader_loaded):
    model_loaded = model_loaded.eval()

    msgs = []
    predictions = []
    predictions_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader_loaded:
            msg = d['msg']
            input_ids = d['input_ids'].to(device_loaded)
            attention_masks = d['attention_mask'].to(device_loaded)
            scam = d['scam'].to(device_loaded)

            outputs = model_loaded(
                input_ids=input_ids,
                attention_mask=attention_masks
            )

            _, preds = torch.max(outputs, dim=1)

            probs = torch.nn.functional.softmax(outputs, dim=1)

            msgs.extend(msg)
            predictions.extend(preds)
            predictions_probs.extend(probs)
            real_values.extend(scam)
    predictions = torch.stack(predictions).cpu()
    predictions_probs = torch.stack(predictions_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return msgs, predictions, predictions_probs, real_values

def predict_single_comment(model, tokenizer, comment):
    model.eval()
    
    encoded_comment = tokenizer.encode_plus(
        comment,
        add_special_tokens=True,
        max_length=MAX_LEN,
        truncation=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoded_comment['input_ids'].to(device_loaded)
    attention_mask = encoded_comment['attention_mask'].to(device_loaded)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, prediction = torch.max(outputs, dim=1)
        probability = torch.nn.functional.softmax(outputs, dim=1)

    predicted_class = 'spam' if prediction == 1 else 'ham'
    confidence = probability[0][1].item()
    
    return predicted_class, confidence


if __name__ == '__main__':
    
    # device = torch.device('cpu')
    device_loaded = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_loaded = ScamClassifier(n_classes=2)
    model_loaded = model_loaded.to(device_loaded)
    model_loaded.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, output_attentions=True, output_hidden_states=True)

    model_loaded.load_state_dict(torch.load("model_saved.pt", map_location=device_loaded))
    tokenizer_loaded = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    model_loaded.to(device_loaded) # Add this line to move the model to the specified device

    print("****************")
    comment = "Good reviews ofdaily from her techniques Almost €30,000 within the week payout on week days feels better @staci.elmafx"
    predicted_class, confidence = predict_single_comment(model_loaded, tokenizer_loaded, comment)
    print(f"Comment: {comment}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence level: {confidence}")
    print("****************")
    comment = "What's a king to a god, what's a god to a non beliver, who don't belive in.."
    predicted_class, confidence = predict_single_comment(model_loaded, tokenizer_loaded, comment)
    print(f"Comment: {comment}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence level: {confidence}")
    print("****************")
    comment = "HI EVERYONE 𝒊𝒏 𝒍𝒂𝒔𝒕 1 𝒘𝒆𝒆𝒌 𝑰 𝒉𝒂𝒗𝒆 𝒎𝒂𝒅𝒆 𝒐𝒗𝒆𝒓 $29,000 𝒘𝒊𝒕𝒉 𝒋𝒖𝒔𝒕 𝒂𝒏 𝒊𝒏𝒗𝒆𝒔𝒕𝒎𝒆𝒏𝒕 𝒐𝒇 $1,300, 𝒊𝒕'𝒔 𝒂𝒄𝒕𝒖𝒂𝒍𝒍𝒚 𝒎𝒚 𝒇𝒊𝒓𝒔𝒕 𝒕𝒊𝒎𝒆 𝒐𝒇 𝒐𝒏𝒍𝒊𝒏𝒆 𝒕𝒓𝒂𝒅𝒆 𝒊𝒏𝒗𝒆𝒔𝒕𝒎𝒆𝒏𝒕 𝒂𝒏𝒅 𝒂𝒎 𝒈𝒍𝒂𝒅 𝑰 𝒆𝒂𝒓𝒏𝒆𝒅 𝒂𝒍𝒍 𝒕𝒉𝒂𝒏𝒌𝒔 𝒕𝒐 @invest_with_danielle_d entrepreneur 𝒂𝒏𝒅 her 𝒕𝒓𝒂𝒅𝒊𝒏𝒈 𝒕𝒆𝒂𝒎"
    predicted_class, confidence = predict_single_comment(model_loaded, tokenizer_loaded, comment)
    print(f"Comment: {comment}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence level: {confidence}")

    
    
    exit(0)
    
    data_example_loaded = [
        ["Hi this is a test", 0],
        ["Santos coin follow", 0],
        ["Resources Elon! Mars or Bust?", 0],
        ["This seems dumb. So let’s say I’m an individual who wants to go get the Bitcoins, I have to pay to be taken to the moon, then I can’t just go off on my own to find them, I have to pay to be taken there with a suit, a crew, vehicles and all that. Then they would’ve taken multiple others who have also payed to be taken there at the same time. I don’t get it. Seems weird 😂", 0],
        ["What's a king to a god, what's a god to a non beliver, who don't belive in..", 0],
        ["HI EVERYONE 𝒊𝒏 𝒍𝒂𝒔𝒕 1 𝒘𝒆𝒆𝒌 𝑰 𝒉𝒂𝒗𝒆 𝒎𝒂𝒅𝒆 𝒐𝒗𝒆𝒓 $29,000 𝒘𝒊𝒕𝒉 𝒋𝒖𝒔𝒕 𝒂𝒏 𝒊𝒏𝒗𝒆𝒔𝒕𝒎𝒆𝒏𝒕 𝒐𝒇 $1,300, 𝒊𝒕'𝒔 𝒂𝒄𝒕𝒖𝒂𝒍𝒍𝒚 𝒎𝒚 𝒇𝒊𝒓𝒔𝒕 𝒕𝒊𝒎𝒆 𝒐𝒇 𝒐𝒏𝒍𝒊𝒏𝒆 𝒕𝒓𝒂𝒅𝒆 𝒊𝒏𝒗𝒆𝒔𝒕𝒎𝒆𝒏𝒕 𝒂𝒏𝒅 𝒂𝒎 𝒈𝒍𝒂𝒅 𝑰 𝒆𝒂𝒓𝒏𝒆𝒅 𝒂𝒍𝒍 𝒕𝒉𝒂𝒏𝒌𝒔 𝒕𝒐 @invest_with_danielle_d entrepreneur 𝒂𝒏𝒅 her 𝒕𝒓𝒂𝒅𝒊𝒏𝒈 𝒕𝒆𝒂𝒎", 1],
        ["Waking up every day to see my balance increased is the best ever happended to me. Thank you so much @harry87", 1],
        ["Good reviews ofdaily from her techniques Almost €30,000 within the week payout on week days feels better @staci.elmafx", 1]
        ]
    df_example_loaded = pd.DataFrame(data_example_loaded, columns=['msg', 'scam'])
    example_data_loader_loaded = create_data_loader_loaded(df_example_loaded, tokenizer_loaded, MAX_LEN, BATCH_SIZE)

    y_ex_msgs, y_ex_pred, y_ex_pred_probs, y_ex_test = get_predictions_loaded(
    model_loaded,
    example_data_loader_loaded
    )

    for x in range(len(y_ex_msgs)): 
        print(round(y_ex_pred_probs[x][1].item(), 4), y_ex_msgs[x])

    
