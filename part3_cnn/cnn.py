
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split

# 基本设置
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 15
LR = 0.001
NUM_CLASSES = 43

# 文件路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

# 加载数据集
trainset = torchvision.datasets.GTSRB(
    root=DATA_DIR,
    split="train",
    download=True,
    transform=transform
)

testset = torchvision.datasets.GTSRB(
    root=DATA_DIR,
    split="test",
    download=True,
    transform=transform
)

# 从训练集中划分验证集
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size

train_data, val_data = random_split(
    trainset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

trainloader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

valloader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("运行设备：", device)
print("训练集数量：", len(train_data))
print("验证集数量：", len(val_data))
print("测试集数量：", len(testset))


# CNN 模型
class TrafficSignCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = TrafficSignCNN().to(device)

print("\n模型结构：")
print(model)

print(
    "参数量：",
    sum(p.numel() for p in model.parameters())
)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 训练记录
train_losses = []
val_losses = []
train_accs = []
val_accs = []

best_val_acc = -1.0
best_val_loss = float("inf")
best_state = None

# 训练
for epoch in range(EPOCHS):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / total
    train_acc = correct / total * 100

    # 验证
    model.eval()

    val_loss_sum = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in valloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_sum / val_total
    val_acc = val_correct / val_total * 100

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # 保存验证集上表现最好的模型
    if (
        val_acc > best_val_acc
        or (
            val_acc == best_val_acc
            and val_loss < best_val_loss
        )
    ):
        best_val_acc = val_acc
        best_val_loss = val_loss
        best_state = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f}, "
        f"Train Acc: {train_acc:.2f}%, "
        f"Val Loss: {val_loss:.4f}, "
        f"Val Acc: {val_acc:.2f}%"
    )

# 加载验证效果最好的模型
model.load_state_dict(best_state)

# 测试
model.eval()
test_correct = 0
test_total = 0
test_loss_sum = 0

with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss_sum += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)
        test_correct += (predictions == labels).sum().item()
        test_total += labels.size(0)

test_loss = test_loss_sum / test_total
test_acc = test_correct / test_total * 100

print("\n最终测试结果：")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

# 绘制 Loss 曲线
plt.figure(figsize=(8, 5))

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(RESULT_DIR, "cnn_loss.png"),
    dpi=200
)
plt.show()

# 绘制 Accuracy 曲线
plt.figure(figsize=(8, 5))

plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("CNN Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(RESULT_DIR, "cnn_accuracy.png"),
    dpi=200
)
plt.show()

# 保存模型
torch.save(
    model.state_dict(),
    os.path.join(RESULT_DIR, "cnn_model.pth")
)

print("\nCNN 实验完成！")
