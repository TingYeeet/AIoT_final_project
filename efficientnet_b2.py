# -*- coding: utf-8 -*-
"""一、載入資料集"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # 啟用互動模式

print(torch.__version__)
print(torch.version.cuda)  # 顯示支持的 CUDA 版本
print(torch.cuda.is_available())  # 應返回 True

if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available. Using CPU instead.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 確認是否可以使用 GPU

# 訓練集的資料擴增
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    transforms.RandomRotation(degrees=10),  # 避免旋轉角度過大
    transforms.Resize((260, 260)),
    transforms.ToTensor(),  # 轉換為 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 歸一化
])

# 驗證集和測試集的基本處理
val_test_transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 設定資料夾路徑
# 假設資料夾結構如下：
# dataset/train/Surprised, Scared, Sad, Normal, Happy, Disgusted, Angry
# train 資料夾內每個子資料夾對應一個類別

data_dir = "./"  # 本地資料夾名稱

# 使用 ImageFolder 加載資料
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'after_crop'))
}

# 獲取資料集的大小
# dataset_sizes['train'] 表示訓練資料的數量
dataset_sizes = {
    'train': len(image_datasets['train'])
}
print("資料集大小:", dataset_sizes)

# 取得資料的類別名稱
# image_datasets['train'].classes 將返回類別的名稱列表
class_names = image_datasets['train'].classes
print("類別名稱:", class_names)

"""二、將資料集分割成80%訓練集，10%驗證集, 10%測試集"""

from torch.utils.data import random_split

# 取得資料量
total_size = len(image_datasets['train'])

# 設定分割比例
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size  # Remaining data for testing

# 執行分割
train_dataset, val_dataset, test_dataset = random_split(image_datasets['train'], [train_size, val_size, test_size])

# 將資料擴增應用到訓練數據集
train_dataset.dataset.transform = train_transform

# 將基本處理應用到驗證和測試數據集
val_dataset.dataset.transform = val_test_transform
test_dataset.dataset.transform = val_test_transform

# 建立dataloader
dataloaders = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0),
    'test': torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0),
}

# 顯示train、val、test的size
dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset),
    'test': len(test_dataset)
}
print("Dataset sizes:", dataset_sizes)

"""三、將EfficientNet-B2模型,修改輸出層為7個類別並設置learning rate scheduler和Checkpoint做特徵提取"""

# # 修改 EfficientNet-B2 模型以適應 7 個輸出類別
# # 使用最新的權重參數設置方法

# # 載入 EfficientNet-B2 預訓練模型
# model = models.efficientnet_b2(pretrained=True)

# # 凍結所有卷積層的權重（特徵提取部分）
# for param in model.features.parameters():
#     param.requires_grad = False

# # 修改分類器以適應 7 個輸出類別
# # EfficientNet 的分類器是一個 nn.Sequential，取出輸入特徵數並替換最後一層
# num_ftrs = model.classifier[1].in_features
# model.classifier[1] = nn.Linear(num_ftrs, 7)  # 替換為 7 類的新分類器

# # 將模型移動到 GPU
# model = model.to(device)

# # 設置損失函數和優化器
# criterion = nn.CrossEntropyLoss()

# # 只優化未凍結的層（即全連接層），並加入 L2 正則化
# optimizer = optim.SGD(
#     model.classifier.parameters(), 
#     lr=0.001, 
#     momentum=0.9, 
#     weight_decay=0.01  # L2 正則化權重
# )

# # 設置學習率調度器
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 定義訓練模型的函數
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=5, log_file='train_log.txt'):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_acc': [], 'val_acc': []}

    # 開啟 log 文件進行寫入
    with open(log_file, 'w') as log:
        log.write(f"Training Started for {num_epochs} epochs\n")
        log.write("-" * 40 + "\n")

        print(f"Training Started for {num_epochs} epochs")
        print("-" * 40)

        for epoch in range(num_epochs):
            log.write(f"Epoch {epoch+1}/{num_epochs}\n")
            log.write("-" * 20 + "\n")

            print(f"Epoch {epoch+1}/{num_epochs}")
            print("-" * 20)

            # 每個 epoch 包括訓練和驗證階段
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # 設置模型為訓練模式
                else:
                    model.eval()   # 設置模型為驗證模式

                running_loss = 0.0
                running_corrects = 0

                # 遍歷數據
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 清空參數梯度
                    optimizer.zero_grad()

                    # 前向傳播
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)

                        logits = outputs  # 如果是 Tensor，直接使用
                        _, preds = torch.max(logits, 1)
                        loss = criterion(logits, labels)

                        # 只有在訓練階段進行反向傳播和優化
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # 統計數據
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                # 更新學習率調度器
                if phase == 'train':
                    scheduler.step()

                # 計算每個 epoch 的損失和準確率
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                log.write(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n")
                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # 保存準確率歷史記錄
                if phase == 'train':
                    history['train_acc'].append(epoch_acc.item())
                else:
                    history['val_acc'].append(epoch_acc.item())

                # 如果驗證準確率提高，保存模型權重
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    log.write("Checkpoint: 更新最佳模型權重。\n")
                    print("Checkpoint: 更新最佳模型權重。")

            log.write("\n")
            print()

        # 加載最佳模型權重
        model.load_state_dict(best_model_wts)
        log.write(f"最佳驗證準確率: {best_acc:.4f}\n")
        print(f"最佳驗證準確率: {best_acc:.4f}\n")
    
    return model, history

# # 訓練模型
# num_epochs = 25
# best_model, history = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, num_epochs, log_file="efficientnetb2_fe.txt")

# # 保存最佳模型權重
# torch.save(best_model.state_dict(), 'efficientnetb2_feature_extraction.pth')

# # 繪製訓練和驗證準確率
# epochs = range(1, num_epochs+1)
# plt.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o')
# plt.plot(epochs, history['val_acc'], label='Val Accuracy', marker='o')

# # 添加數值標註
# for epoch, acc in zip(epochs, history['train_acc']):
#     plt.text(epoch, acc, f"{acc:.2f}", ha='center', va='bottom', fontsize=8, color='blue')

# for epoch, acc in zip(epochs, history['val_acc']):
#     plt.text(epoch, acc, f"{acc:.2f}", ha='center', va='bottom', fontsize=8, color='orange')

# plt.xlabel('Epoch')
# plt.ylabel('accuracy')
# plt.legend()
# plt.title('Train and Validation Accuracy-Feature Extraction')

# plt.savefig('efficientnetb2_feature_extraction.png')
# plt.clf()

"""四、進行fine-tuning"""

# 解凍 EfficientNet-B0 最後幾個卷積層進行微調
# 假設特徵提取的最佳模型已經訓練完成並保存為 "best_efficientnetb2_feature_extraction.pth"

# 載入 EfficientNet-B2 預訓練模型
model = models.efficientnet_b2(pretrained=True)

# 修改分類器以適應 7 個輸出類別
# EfficientNet 的分類器是一個 nn.Sequential，取出輸入特徵數並替換最後一層
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 7)  # 替換為 7 類的新分類器

# 載入已保存的權重
checkpoint = torch.load("efficientnetb2_feature_extraction.pth")
model.load_state_dict(checkpoint)

# 凍結所有卷積層的權重
for param in model.features.parameters():
    param.requires_grad = False

# 解凍 EfficientNet-B2 的最後兩個卷積層
for name, param in model.named_parameters():
    # EfficientNet 的卷積層結構通常分為 blocks (features 中的部分)
    if "features.6" in name or "features.5" in name:  # 選擇最後兩個 blocks 的層
        param.requires_grad = True
    else:
        param.requires_grad = False

# 將模型移動到 GPU
model = model.to(device)

# 定義優化器和學習率調度器
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005, momentum=0.9, weight_decay=0.08)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 使用與特徵提取時相同的損失函數
criterion = nn.CrossEntropyLoss()

# 繼續進行訓練 (微調)
num_epochs_fine_tune = 10

# 確保資料輸入和模型均在 GPU 上
for phase in ['train', 'val']:
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

finetuned_model, history_fine_tune = train_model(
    model, dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs_fine_tune, log_file="efficientnetb2_ft.txt"
)

# 保存微調後的模型權重
torch.save(finetuned_model.state_dict(), 'efficientnetb2_finetuned.pth')

# 繪製訓練和驗證準確率
epochs = range(1, num_epochs_fine_tune+1)
plt.plot(epochs, history_fine_tune['train_acc'], label='Train Accuracy', marker='o')
plt.plot(epochs, history_fine_tune['val_acc'], label='Val Accuracy', marker='o')

# 添加數值標註
for epoch, acc in zip(epochs, history_fine_tune['train_acc']):
    plt.text(epoch, acc, f"{acc:.2f}", ha='center', va='bottom', fontsize=8, color='blue')

for epoch, acc in zip(epochs, history_fine_tune['val_acc']):
    plt.text(epoch, acc, f"{acc:.2f}", ha='center', va='bottom', fontsize=8, color='orange')

plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.legend()
plt.title('Train and Validation Accuracy-Fine-tuning')

plt.savefig('efficientnetb2_finetuning.png')
plt.clf()

"""五、使用微調完的模型對測試資料及做推論，計算top-1至top-3 accuracy"""

# 測試資料集推論: 計算 Top-1、Top-2、Top-3 準確率
def calculate_topk_accuracy(model, dataloader, device, k=(1, 2, 3)):
    model.eval()
    correct_topk = {topk: 0 for topk in k}
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, pred_topk = outputs.topk(max(k), dim=1, largest=True, sorted=True)

            for topk in k:
                correct_topk[topk] += (pred_topk[:, :topk] == labels.view(-1, 1)).sum().item()

            total += labels.size(0)

    topk_accuracy = {f"Top-{topk} Accuracy": correct / total for topk, correct in correct_topk.items()}
    return topk_accuracy

# 測試模型並打印 Top-k 準確率
test_topk_accuracy = calculate_topk_accuracy(finetuned_model, dataloaders['test'], device, k=(1, 2, 3))
print("測試資料集的 Top-k 準確率:")
for topk, acc in test_topk_accuracy.items():
    print(f"{topk}: {acc:.4f}")

"""六、使用微調完的模型推論測試集每種類別的前五張圖片,顯示出圖片、真實標籤和預測標籤"""

import matplotlib.pyplot as plt

# Ensure model is in evaluation mode
finetuned_model.eval()

# Dictionary to track the number of images displayed per class
class_display_count = {class_name: 0 for class_name in class_names}
images_per_class = 7

# Plotting setup
fig, axes = plt.subplots(len(class_names), images_per_class, figsize=(10, 10))

# Inference on test dataset
with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass to get predictions
        outputs = finetuned_model(inputs)
        _, preds = torch.max(outputs, 1)

        # Loop through batch
        for i in range(inputs.size(0)):
            true_label = labels[i].item()
            pred_label = preds[i].item()
            class_name = class_names[true_label]

            # Display the image only if we haven't reached 5 images for this class
            if class_display_count[class_name] < images_per_class:
                row = true_label
                col = class_display_count[class_name]

                # Move tensor to CPU and convert to numpy for plotting
                img = inputs[i].cpu().permute(1, 2, 0).numpy()
                img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Unnormalize
                img = np.clip(img, 0, 1)

                # Display the image with true and predicted labels
                axes[row, col].imshow(img)
                axes[row, col].set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}")
                axes[row, col].axis('off')

                # Increment the count for the current class
                class_display_count[class_name] += 1

        # Break the loop if we have enough images for each class
        if all(count >= images_per_class for count in class_display_count.values()):
            break

plt.tight_layout()
plt.savefig("b2_result_" +str(round(test_topk_accuracy["Top-1 Accuracy"], 2))+ ".png")