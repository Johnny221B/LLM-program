import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np

# 超参数和路径参数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 4
learning_rate = 5e-5
num_epochs = 15000
progress_interval = 50  # 每50个epoch记录一次
patience = 30
train_pic_file = 'training_predictions_data_modify.png'
test_pic_file = 'testing_predictions_data_modify.png'
weight_name = 'linear_probe_weights_data3_modify.pth'
embedding_path = '/home/jingxuan/linear_probing/embeddings_data_starcoder/'
code_name = 'starcoder3-test2-train3-p30-l13.4-linear2-128-lr5.5'
hidden_size = 128
weight_decay = 3e-4
lambda_l1 = 3e-4

# 定义数据集类
class EmbeddingDataset(Dataset):
    def __init__(self, embedding_dir):
        self.embedding_files = [os.path.join(embedding_dir, f) for f in sorted(os.listdir(embedding_dir)) if 'embedding' in f]
        self.scaler = torch.load(os.path.join(embedding_dir, 'scaler.pt'))

    def __len__(self):
        return len(self.embedding_files)

    def __getitem__(self, idx):
        embedding, time = torch.load(self.embedding_files[idx])
        return embedding.squeeze(), torch.tensor(time, dtype=torch.float).squeeze()

# 定义线性探测器
class LinearProbe(torch.nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(LinearProbe, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.batchnorm = torch.nn.BatchNorm1d(hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.linear2(x)
 
# 加载数据和模型
train_dataset = EmbeddingDataset(os.path.join(embedding_path, 'train'))
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = EmbeddingDataset(os.path.join(embedding_path, 'test'))
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
linear_probe = LinearProbe(4608, hidden_size).to(device)  # Assuming embedding size of 4096
optimizer = torch.optim.AdamW(linear_probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = ExponentialLR(optimizer, gamma=0.95)
criterion = nn.MSELoss()

# 初始化W&B
# wandb.init(project='linear-probe-adjust-parameters', name = code_name) 

# 训练循环
best_loss = float('inf')
patience_counter = 0
epoch_losses = []
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for embeddings, time in train_data_loader:
        embeddings, time = embeddings.to(device), time.to(device)
        optimizer.zero_grad()
        predictions = linear_probe(embeddings)
        loss = criterion(predictions, time.unsqueeze(1))
        l1_norm = sum(param.abs().sum() for param in linear_probe.parameters())
        total_loss = loss + lambda_l1 * l1_norm
        loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
    scheduler.step()

    if (epoch + 1) % progress_interval == 0:
        average_loss = epoch_loss / len(train_data_loader)
        epoch_losses.append(average_loss)
        # wandb.log({"epoch_loss": average_loss, "epoch": epoch + 1})
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')

        if average_loss < best_loss:
            best_loss = average_loss
            patience_counter = 0
            best_model_state_dict = linear_probe.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + progress_interval}, Loss: {epoch_loss:.4f}")
                break

# torch.save(best_model_state_dict, weight_name)

# Evaluate model function
def evaluate(model, data_loader, scaler):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for embeddings, time in data_loader:
            embeddings, time = embeddings.to(device), time.to(device)
            pred_time = model(embeddings)
            predictions.extend(pred_time.cpu().numpy().flatten())
            actuals.extend(time.cpu().numpy().flatten())
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

    predictions_out = ','.join(map(str, predictions))
    actuals_out = ','.join(map(str,actuals))

    print("predict:",predictions_out)
    print("actuals:",actuals_out)
    return predictions, actuals

# Plotting predictions function
def plot_predictions(predictions, actuals, title, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(actuals, predictions, label='Predicted vs Actual')
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='orange', label='Ideal Prediction')
    plt.xlabel('Actual Execution Time')
    plt.ylabel('Predicted Execution Time')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# Evaluate and plot for both train and test sets
train_predictions, train_actuals = evaluate(linear_probe, train_data_loader, train_dataset.scaler)
test_predictions, test_actuals = evaluate(linear_probe, test_data_loader, test_dataset.scaler)
plot_predictions(train_predictions, train_actuals, 'Training Data Predictions', train_pic_file)
plot_predictions(test_predictions, test_actuals, 'Testing Data Predictions', test_pic_file)

# wandb.finish()