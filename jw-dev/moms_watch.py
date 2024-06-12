import cv2
import time

from gaze_analysis_cam_ver import gaze_analysis, draw_gaze
from state_awareness_cam_ver import state_awareness


# 웹캠 초기화
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("웹캠을 사용할 수 없습니다.")

while True:
    # 비디오 읽기
    success, frame = cap.read()

    if not success:
        print("프레임을 읽어올 수 없습니다.")
        time.sleep(0.1)
        continue    # 다음 프레임

    detected, face_bbox, yaw, pitch = gaze_analysis(frame)
    cur_state = state_awareness(frame)

    if detected:
        # bbox & gaze 렌더링
        frame = draw_gaze(face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3], frame, (yaw, pitch))

    cv2.putText(frame, f'Yaw: {yaw:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.putText(frame, f'Pitch: {pitch:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    if detected and -0.3 <= yaw <= 0.3 and -0.3 <= pitch <= 0.3:
        cv2.putText(frame, 'Look at me, look at me', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.putText(frame, f'Your now in {cur_state}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'Hey, what r u doing?', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    # 프레임을 화면에 표시
    cv2.imshow("Mom's watch", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠과 창 해제
cap.release()
cv2.destroyAllWindows()
