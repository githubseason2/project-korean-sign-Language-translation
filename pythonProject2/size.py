import os
import cv2
from tqdm import tqdm

# 폴더 경로 설정
folder_path = "./g"

# 폴더 내의 이미지 파일 목록 가져오기
image_files = [file for file in os.listdir(folder_path) if file.endswith((".jpg", ".jpeg", ".png"))]

# 이미지 리사이즈 설정
target_width = 270
target_height = 360

# 진행률 바 생성
progress_bar = tqdm(total=len(image_files), desc="Resizing Images")

# 이미지 리사이즈 및 저장
for file in image_files:
    # 이미지 경로
    image_path = os.path.join(folder_path, file)
    print(image_path)
    # 이미지 읽기
    image = cv2.imread(image_path)

    # 이미지 리사이즈
    resized_image = cv2.resize(image, (target_width, target_height))

    # 리사이즈된 이미지 저장
    resized_image_path = os.path.join(folder_path, "resized", file)
    cv2.imwrite(resized_image_path, resized_image)

    # 진행률 업데이트
    progress_bar.update(1)

progress_bar.close()
