import cv2
import numpy as np
import mediapipe as mp
import csv

# 미디어파이프 핸드 모듈 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 라벨을 숫자로 매핑
action_label_map = {'hello': 0}
action = 'hello'  # 액션 설정
label = action_label_map[action]  # 해당 액션에 대한 숫자 라벨

# 저장할 데이터를 담을 리스트
data = []

# 시퀀스 길이 설정
seq_length = 30  # 시퀀스 길이 추가

# 랜드마크와 각도 값의 총 길이 (양손: 42개 랜드마크, 각 3차원 좌표, 각도 15개)
total_landmarks_length = (42 * 3) + 15 + 1  # 라벨 추가

# 웹캠 입력 시작
cap = cv2.VideoCapture(0)
frame_number = 0  # 프레임 번호 초기화

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1  # 프레임 번호 증가
    # BGR을 RGB로 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    # 다시 BGR로 변환
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 각 손의 랜드마크 처리
    x_values = []
    y_values = []
    z_values = []
    angles = []  # 각도 값 저장용

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            joint = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            # 랜드마크 그리기
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 부모-자식 관절 설정
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]

            # 벡터 계산 및 정규화
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # 각도 계산
            angle = np.arccos(np.einsum('nt,nt->n',
                                          v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                          v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
            angle = np.degrees(angle)

            # 랜드마크 좌표 저장
            for lm in hand_landmarks.landmark:
                x_values.append(lm.x)
                y_values.append(lm.y)
                z_values.append(lm.z)

            angles.extend(angle.tolist())

    # 랜드마크가 감지되지 않은 경우 0으로 채움
    if len(x_values) == 0:
        continue  # 랜드마크가 감지되지 않으면 다음 프레임으로 이동

    # 랜드마크가 42개가 되도록 0으로 채움
    while len(x_values) < 42:
        x_values.append(0.0)
        y_values.append(0.0)
        z_values.append(0.0)

    # 각도 값이 없는 경우 0으로 채움
    if len(angles) < 15:
        angles += [0.0] * (15 - len(angles))  # 부족한 만큼 0으로 채움

    # 각도 값이 15개로 유지되도록 처리
    if len(angles) == 15:
        angles += [0.0] * 15  # 두 번째 손에 대한 각도 값은 0.0으로 채움

    # 숫자로 변환된 라벨 추가
    label_float = float(label)

    # 데이터에 추가
    frame_data = x_values + y_values + z_values + angles + [label_float]
    data.append(frame_data)

    # 데이터 개수 출력
    print(f"Frame: {frame_number}")
    print(f"X Count: {len(x_values)}")
    print(f"Y Count: {len(y_values)}")
    print(f"Z Count: {len(z_values)}")
    print(f"Angles Count: {len(angles)}")
    print(f"Label Count: 1")  # 라벨은 항상 1개
    print(f"Data Dimensions: {len(frame_data)}")

    # 화면에 출력
    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 데이터 numpy 배열로 변환
try:
    data = np.array(data, dtype=float)  # dtype=float으로 변경하여 데이터 배열 통일
except ValueError as e:
    print(f"Error converting data to numpy array: {e}")

# 시퀀스 데이터 생성
full_seq_data = []

# 데이터가 seq_length보다 클 경우에만 시퀀스 생성
if len(data) >= seq_length:
    for seq in range(len(data) - seq_length + 1):
        full_seq_data.append(data[seq:seq + seq_length])

# numpy 배열로 변환
try:
    full_seq_data = np.array(full_seq_data)
except ValueError as e:
    print(f"Error converting sequence data to numpy array: {e}")

# npy 파일로 저장
np.save(f'{action}_hand_data.npy', full_seq_data)

# seq_length를 npy 파일로 저장
np.save('seq_length.npy', seq_length)

# CSV 파일로 저장
csv_filename = f'{action}_hand_data.csv'
with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['X'] * 42 + ['Y'] * 42 + ['Z'] * 42 + ['Angles'] * 15 + ['Action'])  # 첫 번째 줄에 헤더 추가
    for row in full_seq_data.reshape(-1, len(frame_data)):  # 시퀀스 데이터를 평탄화
        writer.writerow(row)

cap.release()
cv2.destroyAllWindows()
