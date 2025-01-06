import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 定義類別名稱
class_names = ['Angry', 'Disgusted', 'Happy', 'Normal', 'Sad', 'Scared', 'Surprised']

# 定義資料路徑
data_dir = '../after_crop'

# 定義圖片轉換
transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 定義自訂資料集類
class ImageDataset(Dataset):
    def __init__(self, data_dir, class_names, transform=None):
        self.data_dir = data_dir
        self.class_names = class_names
        self.transform = transform
        self.images = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.images.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 建立資料集與 DataLoader
dataset = ImageDataset(data_dir, class_names, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# 載入模型
model = models.squeezenet1_0()
model.classifier[1] = nn.Conv2d(512, 7, kernel_size=(1, 1), stride=(1, 1))
model.num_classes = 7  # 更新模型的類別數屬性
model.load_state_dict(torch.load('./SqueezeNet/squeezenet_finetuned.pth', weights_only=True))
model.eval()
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 預測
device = 'cuda' if torch.cuda.is_available() else 'cpu'
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# 計算混淆矩陣
y_true = np.array(all_labels)
y_pred = np.array(all_preds)
cm = confusion_matrix(y_true, y_pred)

# 計算 Precision, Recall, F1-score
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(len(class_names)))

# 計算綜合 Precision, Recall, F1-score
overall_precision, overall_recall, overall_f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

# 繪製混淆矩陣
plt.figure(figsize=(16, 9))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

# 在混淆矩陣上添加 Precision, Recall, F1-score
for i in range(len(class_names)):
    plt.text(len(class_names), i, f'P: {precision[i]:.2f}\nR: {recall[i]:.2f}\nF1: {f1_score[i]:.2f}',
             va='center', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# 添加綜合 Precision, Recall, F1-score
plt.text(len(class_names), len(class_names), f'Overall\nP: {overall_precision:.2f}\nR: {overall_recall:.2f}\nF1: {overall_f1_score:.2f}',
         va='center', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.tight_layout()
plt.savefig('sque_confusion_matrix.png')
