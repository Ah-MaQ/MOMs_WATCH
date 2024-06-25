import cv2
import numpy as np
from ultralytics import YOLO

def yolo_initialize():
    global model, classNames
    # Load the YOLOv8 model
    model = YOLO('./state_dicts/yolo_trained_2.pt')

    classNames = ["r_iris", "l_iris", "r_eyelid", "l_eyelid", "r_center", "l_center"]

def detect(frame):
    results = model(frame)

    detected_cls = {}

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = classNames[int(box.cls[0])]

            if cls in classNames[:4]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                xc = (x1 + x2) // 2
                yc = (y1 + y2) // 2
                width = x2 - x1
                height = y2 - y1
                eye_r = (width + height) // 4

                detected_cls[cls] = [xc, yc, eye_r]
    return detected_cls

if __name__ == "__main__":
    yolo_initialize()

    # 웹캠 초기화
    cap = cv2.VideoCapture(0)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        if ret:
            # Run YOLOv8 inference on the frame
            frame = cv2.flip(frame, 1)

            detected_cls = {}
            detected_cls = detect(frame)

            if "r_iris" in detected_cls and "r_eyelid" in detected_cls:
                xc, yc, eye_r = detected_cls["r_iris"]
                cv2.circle(frame, (xc, yc), eye_r, (0, 0, 255), thickness=1)

            if "l_iris" in detected_cls and "l_eyelid" in detected_cls:
                xc, yc, eye_r = detected_cls["l_iris"]
                cv2.circle(frame, (xc, yc), eye_r, (0, 0, 255), thickness=1)

            # Display the annotated frame
            cv2.imshow("Eye Detect", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()