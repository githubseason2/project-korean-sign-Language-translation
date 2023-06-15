import cv2
import os
import json
from tqdm import tqdm

# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = {
    "Wrist": 0,
    "ThumbMetacarpal": 1, "ThumbProximal": 2, "ThumbMiddle": 3, "ThumbDistal": 4,
    "IndexFingerMetacarpal": 5, "IndexFingerProximal": 6, "IndexFingerMiddle": 7, "IndexFingerDistal": 8,
    "MiddleFingerMetacarpal": 9, "MiddleFingerProximal": 10, "MiddleFingerMiddle": 11, "MiddleFingerDistal": 12,
    "RingFingerMetacarpal": 13, "RingFingerProximal": 14, "RingFingerMiddle": 15, "RingFingerDistal": 16,
    "LittleFingerMetacarpal": 17, "LittleFingerProximal": 18, "LittleFingerMiddle": 19, "LittleFingerDistal": 20,
}

POSE_PAIRS = [
    ["Wrist", "ThumbMetacarpal"], ["ThumbMetacarpal", "ThumbProximal"],
    ["ThumbProximal", "ThumbMiddle"], ["ThumbMiddle", "ThumbDistal"],
    ["Wrist", "IndexFingerMetacarpal"], ["IndexFingerMetacarpal", "IndexFingerProximal"],
    ["IndexFingerProximal", "IndexFingerMiddle"], ["IndexFingerMiddle", "IndexFingerDistal"],
    ["Wrist", "MiddleFingerMetacarpal"], ["MiddleFingerMetacarpal", "MiddleFingerProximal"],
    ["MiddleFingerProximal", "MiddleFingerMiddle"], ["MiddleFingerMiddle", "MiddleFingerDistal"],
    ["Wrist", "RingFingerMetacarpal"], ["RingFingerMetacarpal", "RingFingerProximal"],
    ["RingFingerProximal", "RingFingerMiddle"], ["RingFingerMiddle", "RingFingerDistal"],
    ["Wrist", "LittleFingerMetacarpal"], ["LittleFingerMetacarpal", "LittleFingerProximal"],
    ["LittleFingerProximal", "LittleFingerMiddle"], ["LittleFingerMiddle", "LittleFingerDistal"]
]

# 각 파일 path
protoFile = "C:/Users/SK/Downloads/openpose-master/models/hand/pose_deploy.prototxt"
weightsFile = "C:/Users/SK/Downloads/openpose-master/models/hand/pose_iter_102000.caffemodel"

# 위의 path에 있는 network 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 입력 이미지 폴더
image_folder = [".\\resized\\r",".\\resized\\s",".\\resized\\e",".\\resized\\f",".\\resized\\a",".\\resized\\q",".\\resized\\t"
                ,".\\resized\\d",".\\resized\\w",".\\resized\\c",".\\resized\\z",".\\resized\\x",".\\resized\\v",".\\resized\\g"
                ,".\\resized\\k",".\\resized\\i",".\\resized\\j",".\\resized\\u",".\\resized\\h",".\\resized\\y",".\\resized\\n"
                ,".\\resized\\b",".\\resized\\m",".\\resized\\l",".\\resized\\o",".\\resized\\oo",".\\resized\\p",".\\resized\\pp"
                ,".\\resized\\ml",".\\resized\\hl",".\\resized\\nl"]
print(image_folder)
label=["ㄱ","ㄴ","ㄷ","ㄹ","ㅁ","ㅂ","ㅅ"
                ,"ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"
                ,"ㅏ","ㅑ","ㅓ","ㅕ","ㅗ","ㅛ","ㅜ"
                ,"ㅠ","ㅡ","ㅣ","ㅐ","ㅒ","ㅔ","ㅖ"
                ,"ㅢ","ㅚ","ㅟ"]

# 결과 JSON 파일 경로
output_json = "./a.json"

# 이미지 파일 목록
image_files=[]
total=0
for i in range(31):
    image_files.append( os.listdir(image_folder[i]) )
    total+=len(image_files[i])
print(image_files)

# 진행 상황 표시를 위한 tqdm 설정
progress_bar = tqdm(total=total, desc="Processing Images", unit="image")

# 결과 저장용 리스트
result = []
for j in range(31):
    for image_file in image_files[j]:
        # 이미지 읽어오기
        # print(image_folder[j])
        image_path = os.path.join(image_folder[j], image_file)
        # print(image_path)
        image = cv2.imread(image_path)

        # frame.shape = 불러온 이미지에서 height, width, color 받아옴
        imageHeight, imageWidth, _ = image.shape

        # network에 넣기위해 전처리
        inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

        # network에 넣어주기
        net.setInput(inpBlob)

        # 결과 받아오기
        output = net.forward()

        # output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
        H = output.shape[2]
        W = output.shape[3]

        # 키포인트 검출시 이미지에 그려줌
        points = []
        for i in range(0, 21):
            # 해당 신체부위 신뢰도 얻음.
            probMap = output[0, i, :, :]

            # global 최대값 찾기
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # 원래 이미지에 맞게 점 위치 변경
            x = (imageWidth * point[0]) / W
            y = (imageHeight * point[1]) / H

            # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 (0,0)으로
            if prob > 0.1:
                cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)  # circle(그릴곳, 원의 중심, 반지름, 색)
                cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                            lineType=cv2.LINE_AA)
                points.append((int(x), int(y)))
            else:
                points.append((0, 0))

        # # 이미지 복사
        # imageCopy = image
        #
        # # 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, …)
        # for pair in POSE_PAIRS:
        #     partA = pair[0]  # Head
        #     partA = BODY_PARTS[partA]  # 0
        #     partB = pair[1]  # Neck
        #     partB = BODY_PARTS[partB]  # 1
        #
        #     # print(partA," 와 ", partB, " 연결\n")
        #     if points[partA] and points[partB]:
        #         cv2.line(imageCopy, points[partA], points[partB], (0, 255, 0), 2)


        # 결과 저장
        result.append({
            'coords': points,
            'label' : [label[j]]
        })

        progress_bar.update(1)

# JSON 파일로 저장
with open(output_json, 'w') as f:
    json.dump(result, f)

progress_bar.close()