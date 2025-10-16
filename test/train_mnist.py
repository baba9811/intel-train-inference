import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------
# 1. 디바이스 설정 (XPU가 있으면 사용)
# ---------------------
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ---------------------
# 2. 데이터셋 로드 (MNIST)
# ---------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# ---------------------
# 3. 간단한 모델 정의
# ---------------------
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleMLP().to(device)

# ---------------------
# 4. 손실함수와 옵티마이저
# ---------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------
# 5. 학습 루프
# ---------------------
for epoch in range(1, 3):  # 2 epochs만
    total_loss = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}")

print("🎯 Training complete!")
