import cv2
import os
import time
from datetime import datetime
import sys

save_folder = "webcam_captures"
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не удалось подключиться к веб-камере")
    sys.exit(1)

last_capture_time = time.time()
capture_interval = 1.0  # интервал в секундах
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break
    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        # Генерируем имя файла с временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = os.path.join(save_folder, f"capture_{timestamp}.jpg")
        
        # Сохраняем кадр
        cv2.imwrite(filename, frame)
        print(f"Сохранено: {filename} ({frame_count+1})")
        
        # Обновляем время последнего снимка
        last_capture_time = current_time
        frame_count += 1
    time.sleep(0.01)