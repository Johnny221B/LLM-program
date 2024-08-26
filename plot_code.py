import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Define the device
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

class CodeExecutionTimeDataset(Dataset):
    def __init__(self, json_file, tokenizer, scaler=None):
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.codes = [item['code'] for item in data]
        self.times = [item['time'] for item in data]
        self.tokenizer = tokenizer

        if scaler is None:
            self.scaler = StandardScaler()
            
            self.times = self.scaler.fit_transform(np.array(self.times).reshape(-1, 1))
        else:
            self.scaler = scaler
            self.times = self.scaler.fit_transform(np.array(self.times).reshape(-1, 1))

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        time = self.times[idx]
        inputs = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True)
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), torch.tensor(time, dtype=torch.float)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('/home/jingxuan/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load the pre-trained LLaMA-3 model
model = AutoModel.from_pretrained('/home/jingxuan/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B').to(device)
model.eval()

# Define a simple linear layer
class LinearProbe(torch.nn.Module):
    def __init__(self, input_dim):
        super(LinearProbe, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

linear_probe = LinearProbe(model.config.hidden_size).to(device)
linear_probe.load_state_dict(torch.load('linear_probe_weights_data1.pth'))
linear_probe.eval()

linear_probe2 = LinearProbe(model.config.hidden_size).to(device)
linear_probe2.load_state_dict(torch.load('linear_probe_weights_data2.pth'))
linear_probe2.eval()

scaler1 = StandardScaler()
scaler2 = StandardScaler()
train_dataset1 = CodeExecutionTimeDataset('/home/jingxuan/linear_probing/train_dataset.json', tokenizer, scaler=scaler1)
train_dataset2 = CodeExecutionTimeDataset('/home/jingxuan/linear_probing/train_dataset2.json', tokenizer, scaler=scaler2)
# test_dataset1 = CodeExecutionTimeDataset('/home/jingxuan/linear_probing/test_dataset.json', tokenizer, train_dataset1.scaler)
# test_dataset2 = CodeExecutionTimeDataset('/home/jingxuan/linear_probing/test_dataset.json', tokenizer, train_dataset2.scaler)

train_data_loader1 = DataLoader(train_dataset1, batch_size=1, shuffle=True)
train_data_loader2 = DataLoader(train_dataset2, batch_size=1, shuffle=True)

# test_data_loader1 = DataLoader(test_dataset1, batch_size=1, shuffle=False)
# test_data_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False)

# Function to evaluate the model
def evaluate(model, linear_probe, data_loader, scaler):
    model.eval()
    linear_probe.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for input_ids, attention_mask, time in data_loader:
            input_ids, attention_mask, time = input_ids.to(device), attention_mask.to(device), time.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state[:, -1, :]
            pred_time = linear_probe(last_hidden_state)
            predictions.append(pred_time.cpu().item())
            actuals.append(time.cpu().item())

    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    # Inverse transform the predictions and actuals to original scale
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

    return predictions, actuals

# Evaluate the model
train_predictions, train_actuals = evaluate(model, linear_probe, train_data_loader1, train_dataset1.scaler)
train_predictions2, train_actuals2 = evaluate(model, linear_probe2, train_data_loader2, train_dataset2.scaler)
# test_predictions, test_actuals = evaluate(model, linear_probe, test_data_loader1, train_dataset1.scaler)
# test_predictions2, test_actuals2 = evaluate(model, linear_probe2, test_data_loader2, train_dataset2.scaler)
# print(test_predictions)
# print(test_predictions2)

print(train_actuals)
print(train_actuals2)

# Plot the results
def plot_predictions(train_predictions, train_actuals, train_predictions2, train_actuals2, title, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(train_actuals, train_predictions, label='Training Data Predictions 1', color='blue')
    plt.scatter(train_actuals2, train_predictions2, label='Training Data Predictions 2', color='green')
    plt.plot([min(train_actuals2), max(train_actuals2)], 
             [min(train_actuals2), max(train_actuals2)], 
             color='orange', label='Perfect Prediction')
    
    # Draw lines between points with the same integer part of actual time
    for actual in np.unique(np.floor(train_actuals)):
        train_pred = train_predictions[np.floor(train_actuals) == actual]
        train_pred2 = train_predictions2[np.floor(train_actuals2) == actual]
        if len(train_pred) > 0 and len(train_pred2) > 0:
            plt.plot([actual, actual], [train_pred, train_pred2], 'k--', linewidth=0.5)
    
    plt.xlabel('Actual Running Time')
    plt.ylabel('Predicted Running Time')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

plot_predictions(train_predictions, train_actuals, train_predictions2, train_actuals2, 'tarin set tual vs Predicted Running Time', 'train_running_time.png')
# plot_predictions(test_predictions, test_actuals, test_predictions2, test_actuals2, 'test set tual vs Predicted Running Time', 'test_running_time.png')

print("Plot saved as 'actual_vs_predicted_running_time.png'")
