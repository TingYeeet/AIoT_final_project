import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ImageTk
import os
from torchvision import models, transforms
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

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
    return draw_img

# 定義 UI 操作的核心函數
def select_image():
    # 選擇圖片文件
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not image_path:
        return

    # 清空舊圖片
    for widget in result_frame.winfo_children():
        widget.destroy()

    # 對圖片執行推論
    results = model(image_path)

    # 取得偵測結果
    detections = results[0].boxes.xyxy.cpu().numpy()  # 邊界框座標

    # 執行切割、儲存結果並進行預測
    annotated_img = process_and_save(image_path, detections, output_folder, resnet_model, transform, classes)

    # 更新 UI 顯示結果
    annotated_img.thumbnail((400, 400))
    tk_img = ImageTk.PhotoImage(annotated_img)
    img_label = tk.Label(result_frame, image=tk_img)
    img_label.image = tk_img
    img_label.pack()

# 設定路徑和模型
output_folder = "prediction"
os.makedirs(output_folder, exist_ok=True)

# 載入 YOLOv11 模型
model = YOLO("best.pt")

# 載入 ResNet18 微調模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["Angry", "Disgusted", "Happy", "Normal", "Sad", "Scared", "Surprised"]
resnet_model = models.resnet18()
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, len(classes))
resnet_model.load_state_dict(torch.load("resnet18_finetuned.pth", map_location=device))
resnet_model.to(device)

# 定義圖片轉換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 建立 Tkinter 主視窗
root = tk.Tk()
root.title("YOLO + ResNet Prediction")

# 結果顯示區
result_frame = tk.Frame(root, width=400, height=400, bg="white")
result_frame.pack(pady=10)

# 選擇圖片按鈕
btn_frame = tk.Frame(root)
btn_frame.pack()
select_btn = ttk.Button(btn_frame, text="選擇圖片", command=select_image)
select_btn.pack()

# 啟動主迴圈
root.mainloop()
