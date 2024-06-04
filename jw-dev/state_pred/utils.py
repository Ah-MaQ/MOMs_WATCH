import cv2

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

state_keys={"F":0,
            "S":1,
            "D":2,
            "A":3,
            "N":4,
            }

def crop_face_area(detection, image, factor=1.0, square=True):
    """crop an image around the detected face.

    NOTE
    ----
    1.
    mp_drawing._normalized_to_pixel_coordinates() is necessary.
    Handheld devices store their 'orientation' in the meta data. 
    When rotated, x and y axes need to be translated properly. 
    -90, -180, +90 degree rotation all need different translation. 
    mp_drawing._normalized_to_pixel_coordinates() deals with that. 
    
    1.1 
    The orientation meta data can be accessed by 
    orientation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    
    2. Even though the coordinate is normalied properly, 
    the orientation can STILL be different from what you see in 
    a media player. Some media players can detect the 'upward' direction and 
    automatically rotate the image so taht you see a person standing upright. 

    parameter
    ---------
    detection: mediapipe Detection
    factor: size of cropped image around the face
    
    todo
    ----
    check for boundary
    
    """
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
    
#     print(relative_bounding_box.xmin)
#     print(relative_bounding_box.width)
#     print("rect_start", rect_start_point)
#     print("rect_end", rect_end_point)
    
    cx = int(0.5*(rect_start_point[0] + rect_end_point[0]))
    cy = int(0.5*(rect_start_point[1] + rect_end_point[1]))

    dx = int(factor*0.5*(rect_end_point[0] - rect_start_point[0]))
    dy = int(factor*0.5*(rect_end_point[1] - rect_start_point[1]))
    
    dx = min([dx, cx])#, image_cols-cx]) # left boundary check
    dy = min([dy, cy])#, image_rows-cy])

    dx = dy = min([dx,dy])

    xmin = cx - dx
    xmax = cx + dx
    ymin = cy - dy
    ymax = cy + dy

    #print("xy range", xmin, xmax, ymin, ymax)
    return image[ymin:ymax, xmin:xmax, :]


def get_state(fn_video):
    return state_keys[fn_video.split("/")[-1].split("_")[-3]]

def crop_frames(fn_video, dir_out, 
                    nframes=32, 
                    sampling_freq = 8, 
                    img_size=(112,112),
                    factor=1.5, 
                    rot=None):
    cap = cv2.VideoCapture(fn_video)
    #frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:

        iframe=0
        read_frame=0
        while iframe < nframes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, read_frame)
            
            success, image = cap.read()


            if image is None:
                return iframe
            if not success:
                print("Error? Skipping..")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            if rot != None:
                image = cv2.rotate(image, rot)
            
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = face_detection.process(image)

            img_cols, img_rows, _ = image.shape
            try:
                if results.detections:
                    detection = results.detections[0] # Assume only one detection

                    #mp_drawing.draw_detection(image, detection)
                    #cv2.imwrite(dir_out+f"frame_det_{iframe}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    cropped = crop_face_area(detection, image, factor=factor)                    

                    # Flip the image horizontally for a selfie-view display.
                    resized = cv2.resize(cropped, img_size)
                    cv2.imwrite(dir_out+f"frame_{iframe}.png", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
                    iframe +=1
                read_frame += sampling_freq
            except:
                read_frame += 1

    cap.release()
    return iframe

    