# Video_Mosaic_and_Capture_Function
**YOLO 11** 모델 기반 동영상 내 사람 형상 모자이크 및 프레임 단위 이미지 캡쳐 기능 구현
**HuggingPace Spaces**를 통해 Gradio 기반 호스팅 [https://huggingface.co/spaces/shin0624/Video_Mosaic_and_Capture_Function](https://huggingface.co/spaces/shin0624/Video_Mosaic_and_Capture_Function)


## Licenses
- 본 프로젝트 코드: **MIT License**
- YOLOv11 모델: **AGPL-3.0** ([Ultralytics 공식 문서](https://ultralytics.com/license))
- Gradio: **Apache-2.0**
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 기술 스택
![License](https://img.shields.io/badge/License-MIT%2FAGPL--3.0-blue)
<img src="https://img.shields.io/badge/huggingface-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white">
<img src="https://img.shields.io/badge/yolo11-111F68?style=for-the-badge&logo=yolo&logoColor=white">
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)

# 사용자 흐름
1. 사용자가 동영상을 업로드
2. 동영상 내 사람 형상을 모두 모자이크
3. 처리된 영상을 mp4 포맷으로 저장(파일명 : output_video.mp4)
4. 영상의 프레임을 PNG 이미지, 1980*1080으로 저장 (파일명 : frame_0001.png ~ )
5. 변환 완료 시 mp4와 zip파일로 다운로드 가능.

# 유의사항
- HuggingPace Spaces는 무료 사용자에게 GPU 자원 할당을 해주지 않기 때문에, cpu로 돌려야 함
- 추후 GPU 전환 코드 추가 예정
- GPU 자원 사용 불가 시 Google Colab환경에서 돌리면 됨.( 아래 참조 )

# Google Colab 환경에서 작동하려면
## 1. YOLO 11 기반 동영상 내 사람 형상 모자이크
```
# ▶️ 필요한 패키지 설치
!pip install -q ultralytics moviepy opencv-python-headless

# ▶️ YOLOv11 로드 (person 감지용)
from ultralytics import YOLO
import cv2
import os
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip

# ▶️ YOLOv11 모델 불러오기
model = YOLO("yolo11n.pt")

# ▶️ 동영상 업로드
from google.colab import files
uploaded = files.upload()

# 업로드된 파일명 가져오기
input_path = list(uploaded.keys())[0]
output_path = "output_with_audio.mp4"
temp_video_path = "temp_video_no_audio.mp4"

# ▶️ 모자이크 적용 함수
def apply_mosaic(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        small = cv2.resize(roi, (16, 16), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = mosaic
    return frame

# ▶️ 영상 프레임 단위 처리 및 저장
cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(temp_video_path, fourcc, fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, classes=[0], verbose=False)  # person class only
    boxes = []
    for r in results:
        for b in r.boxes:
            boxes.append(b.xyxy[0].cpu().numpy())
    frame = apply_mosaic(frame, boxes)
    out.write(frame)

cap.release()
out.release()

# ▶️ 오디오 결합 (moviepy 사용)
video_clip = VideoFileClip(temp_video_path)
original_audio = VideoFileClip(input_path).audio
final_clip = video_clip.set_audio(original_audio)
final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

# ▶️ 결과 다운로드 링크 제공
from google.colab import files
files.download(output_path)
```
### 결과
![Image](https://github.com/user-attachments/assets/408ef488-f5e4-42a8-8152-a47b4caa00a4)

## 2. 동영상 프레임 단위 이미지 캡쳐(PNG, 1980*1028)
```
import cv2
import os
from google.colab import files

# 동영상 업로드
uploaded = files.upload()
video_path = list(uploaded.keys())[0]

# 프레임 저장 폴더 생성
output_dir = 'frames_output'
os.makedirs(output_dir, exist_ok=True)

# 비디오 열기
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"FPS: {fps}, Total frames: {frame_count}")

count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 해상도 체크 (자동 유지) – frame 자체가 1980x1080이면 그대로 저장
    if frame.shape[1] != 1980 or frame.shape[0] != 1080:
        frame = cv2.resize(frame, (1980, 1080))

    # PNG 형식으로 저장
    filename = os.path.join(output_dir, f"frame_{count:05d}.png")
    cv2.imwrite(filename, frame)
    count += 1

    if count % 1000 == 0:
        print(f"Saved {count}/{frame_count} frames...")

cap.release()
print("✅ 모든 프레임을 PNG로 저장 완료.")

import shutil
shutil.make_archive('frames_output', 'zip', 'frames_output')
files.download('frames_output.zip')
```
### 결과
![Image](https://github.com/user-attachments/assets/1be787e8-a271-4403-9008-972d4cb35109)


