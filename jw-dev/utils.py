def draw_gaze(a,b,c,d,image_in, pitchyaw, thickness=2, color=(255, 255, 0),sclae=2.0):
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = w/2
    pos = (int(a+c / 2.0), int(b+d / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[1])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.18)
    return image_out


def crop_face_area(detection, image, factor=1.0, square=True):
    image_rows, image_cols, _ = image.shape
    location = detection.location_data
    relative_bounding_box = location.relative_bounding_box
    rect_start_point = mp_drawing._normalized_to_pixel_coordinates(
        relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
        image_rows)
    rect_end_point = mp_drawing._normalized_to_pixel_coordinates(
        relative_bounding_box.xmin + relative_bounding_box.width,
        relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
        image_rows)

    cx = int(0.5 * (rect_start_point[0] + rect_end_point[0]))
    cy = int(0.5 * (rect_start_point[1] + rect_end_point[1]))

    dx = int(factor * 0.5 * (rect_end_point[0] - rect_start_point[0]))
    dy = int(factor * 0.5 * (rect_end_point[1] - rect_start_point[1]))

    dx = min([dx, cx])  # , image_cols-cx]) # left boundary check
    dy = min([dy, cy])  # , image_rows-cy])

    dx = dy = min([dx, dy])

    xmin = cx - dx
    xmax = cx + dx
    ymin = cy - dy
    ymax = cy + dy

    return image[ymin:ymax, xmin:xmax, :]