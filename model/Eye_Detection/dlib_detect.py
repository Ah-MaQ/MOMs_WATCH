import dlib, cv2
red_color = (0, 0, 255)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit(1)

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    dets = detector(img, 1)

    ## 이제부터 인식된 얼굴 개수만큼 반복하여 얼굴 윤곽을 표시한다.
    # k: 얼굴 인덱스, d: 얼굴 좌표

    for k, d in enumerate(dets):
        shape = predictor(img, d)  # shape: 얼굴 랜드마크 추출
        print(shape.num_parts)  # 추출된 점은 68개.
        print(' ')

        # 얼굴 영역 표시
        ## 색깔
        color_f = (0, 0, 255)  # face - 빨강
        color_l_out = (255, 0, 0)  # 랜드마크 바깥쪽(out) - 파랑
        color_l_in = (0, 255, 0)  # 랜드마크 안쪽(in) - 초록
        ## 표시할 선, 도형
        line_width = 3
        circle_r = 3
        ## 글씨
        fontType = cv2.FONT_HERSHEY_SIMPLEX
        fontSize = 2

        # 얼굴(detector)에 사각형 그림
        cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), color_f, line_width)

        # 이제 랜드마크에 점을 찍어보자.
        num_of_points_out = 17
        num_of_points_in = shape.num_parts - num_of_points_out
        gx_out = 0;
        gy_out = 0;
        gx_in = 0;
        gy_in = 0

        # 점을 찍으려면 필요한 건 좌표!  -> 이는 shape.part(번호) 에 (x,y로) 들어있다.
        # 번호값을 하나씩 바꿔가며 좌표를 찍자.
        for i in range(shape.num_parts):  # 총 68개
            shape_point = shape.part(i)
            print('얼굴 랜드마크 No.{} 좌표위치: ({}, {})'.format(i, shape_point.x, shape_point.y))

            # 얼굴 랜드마크마다 그리기
            ## i(랜드마크 번호)가 17보다 작으면 out(바깥쪽)을 그린다 - 파란색 점
            if i < num_of_points_out:
                cv2.circle(img, (shape_point.x, shape_point.y), circle_r, color_l_out, line_width)
                gx_out = gx_out + shape_point.x / num_of_points_out
                gy_out = gy_out + shape_point.y / num_of_points_out

            ##반면 i가 17이상이면 in(안쪽)을 그린다 - 초록색 점
            else:
                cv2.circle(img, (shape_point.x, shape_point.y), circle_r, color_l_in, line_width)
                gx_in = gx_in + shape_point.x / num_of_points_in
                gy_in = gy_in + shape_point.y / num_of_points_in

        # 랜드마크에 톡톡톡 찍힌 점들 중에서도, 가장 중심위치를 표시해보자.
        # 먼저 out(바깥쪽)은 빨강색
        cv2.circle(img, (int(gx_out), int(gy_out)), circle_r, (0, 0, 255), line_width)
        # 그리고 in(안쪽)은 검은색
        cv2.circle(img, (int(gx_in), int(gy_in)), circle_r, (0, 0, 0), line_width)

        # 얼굴 방향 표시하기(정면인지? 측면인지? -> 앞서 만든 out, in 좌표로 계산!)
        try:
            theta = math.asin(2 * (gx_in - gx_out) / (d.right() - d.left()))
            radian = theta * 180 / math.pi
        except:
            theta = 0
            radian = 0
        print('얼굴방향: {0:.3f} (각도: {1:.3f}도)'.format(theta, radian))

        # 이 얼굴방향과 각도를 face('d') 사각형 위에 출력
        if radian < 0:
            textPrefix = 'left'
        else:
            textPrefix = 'right'

        textShow = textPrefix + str(round(abs(radian), 1)) + " deg."
        cv2.putText(img, textShow, (d.left(), d.top()), fontType, fontSize, color_f, line_width)

        cv2.imshow("webcam", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

