import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import scipy
import joblib
import os

# Define a custom PyTorch Dataset for our text data
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, tfidf_vector):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tfidf_vector = tfidf_vector

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'tfidf_vector': self.tfidf_vector[idx]
        }

# Define the model
class BERT_TFIDF_Classifier(nn.Module):
    def __init__(self, bert_model, num_classes, tfidf_size):
        super(BERT_TFIDF_Classifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size + tfidf_size, num_classes)

    def forward(self, input_ids, attention_mask, tfidf_vector):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #import ipdb;ipdb.set_trace()
        pooled_output = bert_output.pooler_output
        concatenated = torch.cat([pooled_output, tfidf_vector], dim=1)
        dropped = self.dropout(concatenated)
        logits = self.classifier(dropped)
        return logits




def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        tfidf_vector = torch.tensor(batch['tfidf_vector'], dtype=torch.float).to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, tfidf_vector)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct_predictions += (outputs.argmax(1) == labels).sum().item()

    return model, optimizer, total_loss / len(data_loader), correct_predictions / len(data_loader.dataset)

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            tfidf_vector = torch.tensor(batch['tfidf_vector'], dtype=torch.float).to(device)

            outputs = model(input_ids, attention_mask, tfidf_vector)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            total_loss += loss.item()
            correct_predictions += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(data_loader), correct_predictions / len(data_loader.dataset)



def get_all_features(data, tfidf_fields=None, count_fields=None, 
                     numerical_fields=None, tfidf_features=None,
                     count_features=None, **kwargs):
  features = []
  vectorizer = {"tfidf": [], "count": []}
  if not tfidf_fields:
    tfidf_fields = []

  if not count_fields:
    count_fields = []

  if not tfidf_features:
    tfidf_features = {"ngram_range": (1, 3), "max_df": 0.9, "min_df": 0.1}
  if not count_features:
    count_features = {"ngram_range": (1, 3), "max_df": 0.9, "min_df": 0.1}    

  if not numerical_fields:
    numerical_fields = []

  for field in tfidf_fields:
    tfidf_vec = TfidfVectorizer(**tfidf_features)
    features.append(tfidf_vec.fit_transform(data[field]))
    vectorizer["tfidf"].append((field, tfidf_vec)) 

  for field in count_fields:
    count_vec = CountVectorizer(**count_features)
    features.append(count_vec.fit_transform(data[field]))
    vectorizer["count"].append((field, count_vec))



  for field in numerical_fields:
    features.append(scipy.sparse.csr_matrix(data[field]).T)

  return scipy.sparse.hstack(features), vectorizer


def get_features_test(data, vectorizer=None, numerical_fields=None):
  if not vectorizer:
    raise Exception("Vectorizer need")
  
  tfidf = vectorizer.get("tfidf", {})
  count_vec = vectorizer.get("count", {})

  features = []
  if not numerical_fields:
    numerical_fields = []

  for ele in tfidf:
    features.append(ele[1].transform(data[ele[0]]))

  for ele in count_vec:
    features.append(ele[1].transform(data[ele[0]]))

  for field in numerical_fields:
    features.append(scipy.sparse.csr_matrix(data[field]).T)

  return scipy.sparse.hstack(features)



def data_batcher(tr_data, test_data, reddit_data,
                 labels2id, batch_size=8, max_len=512, model_name="bert-base-uncased",
                 **tfidf_params):
                          
    tr_data["alg_id"] = tr_data["alg"].apply(lambda x: labels2id[x])    
    test_data["alg_id"] = test_data["alg"].apply(lambda x: labels2id[x])
    reddit_data["alg_id"] = reddit_data["alg"].apply(lambda x: labels2id[x])

    train_data, val_data = train_test_split(tr_data, stratify=tr_data["alg_id"], test_size=0.2, random_state=42)
    
    train_features, vectorizer = get_all_features(train_data, **tfidf_params)
    
    val_features = get_features_test(val_data, vectorizer, numerical_fields=tfidf_params.get("numerical_fields"))
    
    test_features = get_features_test(test_data, vectorizer, numerical_fields=tfidf_params.get("numerical_fields"))

    reddit_features = get_features_test(reddit_data, vectorizer, numerical_fields=tfidf_params.get("numerical_fields"))

    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    
    train_dataset = TextDataset(train_data["generation"].to_list(), train_data["alg_id"].to_list(), tokenizer, max_len, train_features.toarray())
                                                      
    val_dataset = TextDataset(val_data["generation"].to_list(), val_data["alg_id"].to_list(), tokenizer, max_len, val_features.toarray())
    
    test_dataset = TextDataset(test_data["generation"].to_list(), test_data["alg_id"].to_list(), tokenizer, max_len, test_features.toarray())  

    reddit_dataset = TextDataset(reddit_data["generation"].to_list(), reddit_data["alg_id"].to_list(), tokenizer, max_len, reddit_features.toarray())

    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)       
                                                                                                   
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)   
    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)   
    
    reddit_dataloader = DataLoader(reddit_dataset, batch_size=batch_size)

    concat_model = BERT_TFIDF_Classifier(model, num_classes=len(labels2id), tfidf_size=train_features.shape[1])

    return concat_model, vectorizer, train_dataloader, val_dataloader, test_dataloader, reddit_dataloader


def data_batcher_evaluate(test_data, reddit_data, vectorizer, labels2id,
                          numerical_fields=None, batch_size=8, max_len=512,
                          model_name="bert-base-uncased"):

    test_data["alg_id"] = test_data["alg"].apply(lambda x: labels2id[x])
    reddit_data["alg_id"] = reddit_data["alg"].apply(lambda x: labels2id[x])

    test_features = get_features_test(
        test_data, vectorizer, numerical_fields=numerical_fields
    )

    reddit_features = get_features_test(
        reddit_data, vectorizer, numerical_fields=numerical_fields
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)


    test_dataset = TextDataset(test_data["generation"].to_list(),
                               test_data["alg_id"].to_list(), tokenizer,
                               max_len, test_features.toarray())

    reddit_dataset = TextDataset(reddit_data["generation"].to_list(),
                                 reddit_data["alg_id"].to_list(), tokenizer,
                                 max_len, reddit_features.toarray())

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    reddit_dataloader = DataLoader(reddit_dataset, batch_size=batch_size)

    return test_dataloader, reddit_dataloader


def modelling(model, train_dataloader, eval_dataloader, model_dir, vectorizer, num_epochs=3,
              parallel_gpu=False):
    
    if parallel_gpu:
        model = nn.DataParallel(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    best_opt = None
    best_model = model
    min_val_loss = float("inf")
    if not os.path.exists("models/" + model_dir):
        os.makedirs("models/" + model_dir)

    for epoch in range(num_epochs):
        model, optimizer, train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, device)
        val_loss, val_acc = evaluate(model, eval_dataloader, device)
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if min_val_loss > val_loss:
            min_val_loss = val_loss
            best_model = model

        state = {
            'epoch': num_epochs,
            'state_dict': model.state_dict(),
            'optimizer': model.state_dict(),
        }
        savepath = f'{"models/" + model_dir}/checkpoint_epoch={epoch}-val_loss={val_loss}.ckpt'
        torch.save(model, savepath)

        #print(f'Epoch {epoch + 1}/{num_epochs}')
        #print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        #print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    joblib.dump(vectorizer, f'{"models/" + model_dir}/vectorizer.pkl')
    return model, best_model


def evaluate_test(model, test_dataloader, labels2id, true_label):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    correct_predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            tfidf_vector = torch.tensor(batch['tfidf_vector'], dtype=torch.float).to(device)

            outputs = model(input_ids, attention_mask, tfidf_vector)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            correct_predictions.append(outputs.argmax(1).tolist())
            # correct_predictions += (outputs.argmax(1) == labels).sum().item()
            
    id_label = {v: k for k, v in labels2id.items()}
    from sklearn.metrics import classification_report
    pred = sum([pred for pred in correct_predictions], [])
    pred = list(map(lambda x: id_label[x], pred))
    print(classification_report(pred, true_label))

    return pred, true_label


