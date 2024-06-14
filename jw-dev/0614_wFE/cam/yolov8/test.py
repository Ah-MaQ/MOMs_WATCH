import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('weights/best.pt')

classNames = ["r_iris", "l_iris", "r_eyelid", "l_eyelid", "r_center", "l_center"]

# webcam 사용시
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if ret:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        detected_cls = {}

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = classNames[int(box.cls[0])]

                if cls in classNames[:4]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    xc_ = (x1 + x2) // 2
                    yc_ = (y1 + y2) // 2
                    width = x2 - x1
                    height = y2 - y1
                    eye_r_ = (width + height) // 4

                    detected_cls[cls] = [xc_, yc_, eye_r_]

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