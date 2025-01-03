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
    img = Image.open(image_path)
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)

    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except IOError:
        font = ImageFont.load_default()

    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = map(int, detection[:4])
        cropped_img = img.crop((x1, y1, x2, y2)).convert("RGB")
        input_tensor = transform(cropped_img).unsqueeze(0).to(device)

        resnet_model.eval()
        with torch.no_grad():
            output = resnet_model(input_tensor)
            _, predicted = torch.max(output, 1)
            label = classes[predicted.item()]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 20), f"{label}", fill="red", font=font)

    annotated_path = os.path.join(output_folder, "annotated_result.png")
    draw_img.save(annotated_path)
    return draw_img

# 定義 UI 操作的核心函數
def select_image():
    global canvas  # 確保 Canvas 被全局引用
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not image_path:
        return

    results = model(image_path)
    detections = results[0].boxes.xyxy.cpu().numpy()
    annotated_img = process_and_save(image_path, detections, output_folder, resnet_model, transform, classes)

    for widget in result_frame.winfo_children():
        widget.destroy()

    canvas = tk.Canvas(result_frame, bg="white")  # 重新創建 Canvas
    canvas.pack()

    width, height = annotated_img.size
    canvas.config(width=width, height=height)
    tk_img = ImageTk.PhotoImage(annotated_img)
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
    canvas.image = tk_img

output_folder = "prediction"
os.makedirs(output_folder, exist_ok=True)

model = YOLO("best.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["Angry", "Disgusted", "Happy", "Normal", "Sad", "Scared", "Surprised"]
resnet_model = models.resnet18()
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, len(classes))
resnet_model.load_state_dict(torch.load("resnet18_finetuned.pth", map_location=device))
resnet_model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

root = tk.Tk()
root.title("YOLO + ResNet Prediction")

result_frame = tk.Frame(root)
result_frame.pack(pady=10)
canvas = tk.Canvas(result_frame, bg="white")
canvas.pack()

btn_frame = tk.Frame(root)
btn_frame.pack()
select_btn = ttk.Button(btn_frame, text="選擇圖片", command=select_image)
select_btn.pack()

root.mainloop()
