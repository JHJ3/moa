import cv2
import numpy as np

# npy 파일 경로
npy_file_path = 'hello_hand_data.npy'  # 변경 필요
seq_length = 30  # 시퀀스 길이

# npy 파일 로드
data = np.load(npy_file_path)

# 데이터의 shape을 확인하여 시퀀스 데이터로 나누기
num_sequences = data.shape[0] // seq_length
full_seq_data = data[:num_sequences * seq_length].reshape((num_sequences, seq_length, -1))

# 시각화
for sequence in full_seq_data:
    frame_index = 0  # 현재 프레임 인덱스 초기화

    while frame_index < seq_length:
        # 빈 검정 화면 생성
        black_screen = np.zeros((480, 640, 3), dtype=np.uint8)

        # 랜드마크 데이터를 가져오기
        frame = sequence[frame_index]
        x_values = frame[:42]
        y_values = frame[42:84]

        # 랜드마크를 그리기 위한 좌표 변환 (화면 크기에 맞게 조정)
        for x, y in zip(x_values, y_values):
            if x != 0.0 and y != 0.0:  # 유효한 좌표인 경우
                # 좌표를 픽셀 위치로 변환
                pixel_x = int(x * 640)
                pixel_y = int(y * 480)
                # 랜드마크 점 그리기
                cv2.circle(black_screen, (pixel_x, pixel_y), 5, (0, 255, 0), -1)

        # 빈 화면에 랜드마크 그리기
        cv2.imshow('Landmark Visualization', black_screen)

        # 프레임을 보여주고 q 키를 눌러서 다음 프레임으로 넘어감
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q'):
            frame_index += 1  # 다음 프레임으로 이동
        elif key & 0xFF == 27:  # ESC 키를 눌렀을 경우 종료
            break

    # 시퀀스가 끝났을 때
    print("Finished displaying the current sequence.")

cv2.destroyAllWindows()
