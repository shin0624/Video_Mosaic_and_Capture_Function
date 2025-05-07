# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 [shin0624]
# YOLOv11 �κ��� AGPL-3.0 ���̼��� ����

import gradio as gr
import cv2
import tempfile
import os
import shutil
import zipfile
from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

def zip_frames(frames_dir):
    zip_path = str(Path(frames_dir).with_suffix('.zip'))
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in Path(frames_dir).glob('*.png'):
            zipf.write(file, arcname=file.name)
    return zip_path

def process_video(video_file):
    temp_dir = tempfile.mkdtemp()  # �ӽ� ���丮 ���
    try:
        cap = cv2.VideoCapture(video_file)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames_dir = Path(temp_dir) / "frames"
        frames_dir.mkdir(exist_ok=True)
        output_video_path = str(Path(temp_dir) / "output.mp4")
        
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # CPU�� �̵� �� numpy ��ȯ
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue
                    # ������ũ ó��
                    face = cv2.resize(face, (8,8), interpolation=cv2.INTER_LINEAR)
                    face = cv2.resize(face, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
                    frame[y1:y2, x1:x2] = face
            
            frame_path = frames_dir / f"frame_{frame_count:05d}.png"
            cv2.imwrite(str(frame_path), frame)
            out.write(frame)
            frame_count += 1
        
        out.release()
        cap.release()
        return output_video_path, zip_frames(frames_dir)
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)  # �ӽ� ���� ����

demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="������ ���ε�"),
    outputs=[
        gr.Video(label="������ũ ����"), 
        gr.File(label="������ ZIP ����")
    ],
    title="YOLO ��� �� ������ũ",
    description="������ ���ε� �� �� ������ũ ���� �� ���� �� ������ ZIP ����"
)

if __name__ == "__main__":
    demo.launch()