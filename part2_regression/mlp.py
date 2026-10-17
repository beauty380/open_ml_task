
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

# 基本设置
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 200
BATCH_SIZE = 32
LR = 0.001

# 结果保存路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# 生成数据
X, y = make_friedman1(
    n_samples=1500,
    n_features=10,
    noise=1.0,
    random_state=42
)

# 划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# 标准化
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
X_val = x_scaler.transform(X_val)
X_test = x_scaler.transform(X_test)

y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_val = y_scaler.transform(y_val.reshape(-1, 1))
y_test = y_scaler.transform(y_test.reshape(-1, 1))

# 转换为 Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

trainset = TensorDataset(X_train, y_train)
trainloader = DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True
)

# MLP 模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


model = MLP().to(device)

print("模型结构：")
print(model)

print("参数量：", sum(p.numel() for p in model.parameters()))

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 训练
train_losses = []
val_losses = []

best_val_loss = float("inf")
best_state = None

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X_batch, y_batch in trainloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        pred = model(X_batch)
        loss = criterion(pred, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

    train_loss = total_loss / len(trainset)
    train_losses.append(train_loss)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val).item()

    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

# 恢复验证集效果最好的参数
model.load_state_dict(best_state)

# 测试
model.eval()

with torch.no_grad():
    y_pred = model(X_test).cpu().numpy()

y_pred = y_scaler.inverse_transform(y_pred).ravel()
y_true = y_scaler.inverse_transform(y_test).ravel()

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n最终测试结果：")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2: {r2:.4f}")

# Loss 曲线
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("MLP Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(RESULT_DIR, "mlp_loss.png"),
    dpi=200
)
plt.show()

# 预测效果图
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.5, s=15)

low = min(y_true.min(), y_pred.min())
high = max(y_true.max(), y_pred.max())

plt.plot([low, high], [low, high], "r--")

plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("MLP Predictions")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULT_DIR, "mlp_prediction.png"),
    dpi=200
)
plt.show()

# 保存模型
torch.save(
    model.state_dict(),
    os.path.join(RESULT_DIR, "mlp_model.pth")
)

print("\nMLP 实验完成！")
