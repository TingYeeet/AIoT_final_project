訓練資料
--------------------------
/after_crop

應用程式
--------------------------
/application  
real_predict.py讀取/test_img的檔案  
將預測結果輸出至/prediction

real_predict_UI.py
載入訓練完成的YOLOv11、ResNet18權重進行分割和類別預測  
可直接用UI選擇圖片並顯示結果

模型訓練
--------------------------
/training code  
|efficientnet_b2.py|inception_v3.py|mobilenet_v2.py|resnet18.py|squeezenet.py|
| ------------- | ------------- |------------- |------------- |------------- |

訓練結果
--------------------------
/result  
|/EfficientNet-B2|/InceptionV3|/MobileNetV2|/ResNet18|/SqueezeNet|
| ------------- | ------------- |------------- |------------- |------------- |  
註:未加tensorboard



