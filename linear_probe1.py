import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import wandb
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 超参数和路径参数
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
batch_size = 1
learning_rate = 1e-4
num_epochs = 15000
patience = 30
lambda_l1 = 1e-4
progress_interval = 50
max_length = 512
model_path = '/home/jingxuan/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B'
train_data_path = '/home/jingxuan/linear_probing/train_dataset.json'
test_data_path = '/home/jingxuan/linear_probing/test_dataset.json'

train_pic_file = 'training_predictions.png'
test_pic_file = 'testing_predictions.png'
weight_name = 'linear_probe_weights_data1.pth'

# 初始化 W&B
wandb.init(project='linear-probe-new-method', name='llama3-data-smallerlr-nomaxlen')

# 自定义数据集类
class CodeExecutionTimeDataset(Dataset):
    def __init__(self, json_file, tokenizer, scaler=None, fit_scaler=True):
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.codes = [item['code'] for item in data]
        self.times = [item['time'] for item in data]
        self.tokenizer = tokenizer

        if scaler is None:
            self.scaler = StandardScaler()
            if fit_scaler:
                self.times = self.scaler.fit_transform(np.array(self.times).reshape(-1, 1))
            else:
                self.times = self.scaler.transform(np.array(self.times).reshape(-1, 1))
        else:
            self.scaler = scaler
            if fit_scaler:
                self.times = self.scaler.fit_transform(np.array(self.times).reshape(-1, 1))
            else:
                self.times = self.scaler.transform(np.array(self.times).reshape(-1, 1))

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        time = self.times[idx]
        inputs = self.tokenizer(code, return_tensors='pt', truncation=True)
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), torch.tensor(time, dtype=torch.float)

# 定义简单线性层
class LinearProbe(torch.nn.Module):
    def __init__(self, input_dim):
        super(LinearProbe, self).__init__()
        # self.batchnorm = torch.nn.BatchNorm1d(input_dim)
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        # x = self.batchnorm(x)
        return self.linear(x)

# 定义模型
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
            predictions.extend(pred_time.cpu().numpy().flatten())
            actuals.extend(time.cpu().numpy().flatten())

    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

    print("Predictions after inverse transform:", predictions[:])
    print("Actuals after inverse transform:", actuals[:])

    return predictions, actuals

def plot_predictions(predictions, actuals, title, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(actuals, predictions, label='LLM prediction')
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='orange', label='perfect prediction')
    plt.xlabel('Actual Running Time')
    plt.ylabel('Predicted Running Time')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)  # Save the plot as a file
    plt.show()

# 初始化分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

scaler = StandardScaler()

train_dataset = CodeExecutionTimeDataset(train_data_path, tokenizer, scaler=scaler, fit_scaler=True)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CodeExecutionTimeDataset(test_data_path, tokenizer, scaler=scaler, fit_scaler=False)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = AutoModel.from_pretrained(model_path).to(device)

for param in model.parameters():
    param.requires_grad = False

linear_probe = LinearProbe(model.config.hidden_size).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(linear_probe.parameters(), lr=learning_rate)

best_loss = float('inf')
patience_counter = 0

model.eval()
linear_probe.train()
epoch_losses = []

for epoch in range(0, num_epochs, progress_interval):
    running_loss = 0.0
    with tqdm(total=progress_interval * len(train_data_loader), unit="batch") as tepoch:
        for i in range(progress_interval):
            for input_ids, attention_mask, time in train_data_loader:
                input_ids, attention_mask, time = input_ids.to(device), attention_mask.to(device), time.to(device)
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state[:, -1, :]

                predictions = linear_probe(last_hidden_state)
                loss = criterion(predictions, time.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tepoch.update(1)

                wandb.log({"batch_loss": loss.item()})

    epoch_loss = running_loss / (progress_interval * len(train_data_loader))
    epoch_losses.append(epoch_loss)
    wandb.log({"epoch_loss": epoch_loss, "epoch": epoch + progress_interval})

    print(f'Epoch [{epoch + progress_interval}/{num_epochs}], Loss: {epoch_loss:.4f}')

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
        best_model_state_dict = linear_probe.state_dict()
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch + progress_interval}, Loss: {epoch_loss:.4f}")
        break

torch.save(best_model_state_dict, weight_name)
print("Linear layer weights saved.")

train_predictions, train_actuals = evaluate(model, linear_probe, train_data_loader, scaler)
test_predictions, test_actuals = evaluate(model, linear_probe, test_data_loader, scaler)

plot_predictions(train_predictions, train_actuals, 'Training Data Predictions', train_pic_file)
plot_predictions(test_predictions, test_actuals, 'Testing Data Predictions', test_pic_file)

wandb.finish()
