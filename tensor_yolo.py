import cv2
import time
from ultralytics import YOLO

# 1. FP32 Engine modelini yükle
# Not: Engine dosyası hangi klasördeyse tam yolunu belirtin
model = YOLO('yolo_small_640_kovak_fp16.engine', task='detect')

# 2. Giriş videosu ve Kayıt ayarları
video_path = "test_video_3.mp4"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_fps = cap.get(cv2.CAP_PROP_FPS)

output_path = "output_test3_640_kovak_fp16.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))

print("TensorRT FP32 Engine ile video işleniyor...")
fpsacc = 0
counter = 0
fpsaccglobal = 0
printcount = 0
minfps = 1000
maxfps = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --- SADECE MODEL HIZINI ÖLÇMEK İÇİN ZAMANLAYICIYI BAŞLAT ---
    start_inference = time.time()

    # 3. Model Tahmini (Inference)
    # device=0: Kesinlikle GPU kullanması gerektiğini belirtiyoruz
    # imgsz=320: Export ederken verdiğin değerle aynı olmalı
    results = model.track(frame, 
                          imgsz=640, 
                          conf=0.40, 
                          device=0, 
                          verbose=False, 
                          persist=True,
                          tracker="bytetrack.yaml") # İsteğe bağlı: 'botsort.yaml' da kullanabilirsin.

    # --- ZAMANLAYICIYI BİTİR ---
    end_inference = time.time()
    inf_time = end_inference - start_inference
    fps = 1 / inf_time if inf_time > 0 else 0
    counter += 1
    fpsacc += fps
    if counter % 10 == 0:
        avg_fps = fpsacc / 10
        counter = 0
        fpsacc = 0
        printcount += 1
        if fps < minfps:
            minfps = fps
        if fps > maxfps:
            maxfps = fps
        fpsaccglobal += avg_fps
        print(f"FP32 Engine Inference FPS: {avg_fps:.2f} (avg 10 frame)")

    # 4. Görselleştirme (Bbox çizimi)
    annotated_frame = results[0].plot()

    # Ekrana FPS bilgisini yaz
    cv2.putText(annotated_frame, f"FP32 Engine FPS: {fps:.2f}", (30, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 5. Videoyu kaydet (Ölçüm dışında)
    out.write(annotated_frame)

# Kaynakları temizle
cap.release()
out.release()
    
avgtotal = fpsaccglobal / printcount
print(f"FP32 Engine Ortalama Inference FPS: {avgtotal:.2f}")
print(f"FP32 Engine Minimum Inference FPS: {minfps:.2f}")
print(f"FP32 Engine Maximum Inference FPS: {maxfps:.2f}")
print(f"İşlem tamamlandı. Çıkış videosu kaydedildi: {output_path}")
