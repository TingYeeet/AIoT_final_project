import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
from torchvision import models, transforms
import torch.nn as nn

# 定義函數：執行切割並保存結果
def process_and_save(image_path, detections, output_folder, resnet_model, transform, classes):
    # 讀取原圖
    img = Image.open(image_path)
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)

    # 字型（可選）
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except IOError:
        font = ImageFont.load_default()

    for i, detection in enumerate(detections):
        # 提取邊界框座標，格式：[x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, detection[:4])
        cropped_img = img.crop((x1, y1, x2, y2)).convert("RGB")  # 裁剪並轉換為 RGB

        # # 保存切割後的圖片
        # crop_path = os.path.join(output_folder, f"crop_{i + 1}.png")
        # cropped_img.save(crop_path)

        # 圖片預處理
        input_tensor = transform(cropped_img).unsqueeze(0).to(device)

        # 預測分類
        resnet_model.eval()
        with torch.no_grad():
            output = resnet_model(input_tensor)
            _, predicted = torch.max(output, 1)
            label = classes[predicted.item()]

        # 在原圖上繪製邊界框和標籤
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 20), f"{label}", fill="red", font=font)

    # 保存帶有標記的圖片
    annotated_path = os.path.join(output_folder, "annotated_result.png")
    draw_img.save(annotated_path)
    print(f"Saved annotated image to {annotated_path}")

# 載入 YOLOv11 模型
model = YOLO("best.pt")

# 定義測試圖片路徑
image_path = "./test img/test5.png"

# 定義輸出資料夾
output_folder = "prediction"
os.makedirs(output_folder, exist_ok=True)

# 對圖片執行推論
results = model(image_path)

# 取得偵測結果
detections = results[0].boxes.xyxy.cpu().numpy()  # 邊界框座標
confidences = results[0].boxes.conf.cpu().numpy()  # 置信度
classes = ["Surprised", "Scared", "Sad", "Normal", "Happy", "Disgusted", "Angry"]  # 修改為你的分類類別

# 載入 ResNet18 微調模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = models.resnet18()
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, len(classes))  # 假設有7個類別
resnet_model.load_state_dict(torch.load("resnet18_finetuned.pth", map_location=device))
resnet_model.to(device)

# 定義圖片轉換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 執行切割、儲存結果並進行預測
process_and_save(image_path, detections, output_folder, resnet_model, transform, classes)
print("完成處理！")
