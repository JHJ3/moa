import cv2
import numpy as np
import os
import mediapipe as mp

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 시각화할 액션 및 데이터 파일 경로
action = 'a'  # 사용한 액션
created_time = 'your_created_time_here'  # 생성된 시퀀스의 시간
seq_file_path = os.path.join('dataset', f'seq_a_1727528181.npy')

# 데이터 로드
full_seq_data = np.load(seq_file_path)

# 비디오 캡처
cap = cv2.VideoCapture(0)  # 적절한 카메라 인덱스 설정

with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    for seq in full_seq_data:
        for frame in seq:
            # 프레임에서 랜드마크 데이터 추출
            joint = frame[:84].reshape(21, 4)  # 21개의 랜드마크 (x, y, z, visibility)

            # 이미지 생성
            img = np.zeros((480, 640, 3), dtype=np.uint8)  # 640x480 이미지 생성

            # 랜드마크 그리기
            for j in range(21):
                x = int(joint[j, 0] * img.shape[1])  # x 좌표
                y = int(joint[j, 1] * img.shape[0])  # y 좌표
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # 랜드마크 그리기

            # 랜드마크 연결
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if joint[start_idx, 3] > 0 and joint[end_idx, 3] > 0:  # visibility가 0보다 큰 경우
                    start_point = (int(joint[start_idx, 0] * img.shape[1]), int(joint[start_idx, 1] * img.shape[0]))
                    end_point = (int(joint[end_idx, 0] * img.shape[1]), int(joint[end_idx, 1] * img.shape[0]))
                    cv2.line(img, start_point, end_point, (255, 0, 0), 2)  # 손 연결 그리기

            # 이미지 표시
            cv2.imshow('Hand Landmarks', img)

            if cv2.waitKey(100) & 0xFF == ord('q'):  # 100ms 대기 후 종료
                break

cap.release()
cv2.destroyAllWindows()
