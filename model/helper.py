import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel

import warnings
warnings.filterwarnings('ignore')


PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
MAX_LEN=512
BATCH_SIZE=16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class ScamCollectionDataset(Dataset):
    def __init__(self, scam, msgs, tokenizer, max_len):
        self.scam = scam
        self.msgs = msgs
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
    
def get_predictions(model, data_loader):
    model = model.eval()

    msgs = []
    predictions = []
    predictions_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            msg = d['msg']
            input_ids = d['input_ids'].to(device)
            attention_masks = d['attention_mask'].to(device)
            scam = d['scam'].to(device)

            outputs = model(
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

    input_ids = encoded_comment['input_ids'].to(device)
    attention_mask = encoded_comment['attention_mask'].to(device)

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
