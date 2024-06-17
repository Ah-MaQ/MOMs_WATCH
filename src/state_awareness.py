import numpy as np
import cv2
from PIL import Image
from collections import deque
import time
import torch
import torchvision

from models.dfer.ST_Former import GenerateModel

class GroupResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        return [img.resize((self.size, self.size), self.interpolation) for img in img_group]

class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode in ['L', 'F']:
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            return np.concatenate([np.array(x)[:, :, ::-1] if self.roll else np.array(x) for x in img_group], axis=2)

class ToTorchFormatTensor(object):
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode)).transpose(0, 1).transpose(0, 2).contiguous()
        return img.to(torch.float32).div(255) if self.div else img.to(torch.float32)

# Global variables to hold the model and state
model = None
device = None
transform = None
frames = deque(maxlen=16)

state_labels = ['Concentration', 'Drowsiness', 'Lack of Concentration',
                'Decrease in Concentration', 'Negligence']

def state_initialize():
    global model, device, transform

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model
    model = GenerateModel()
    model = torch.nn.DataParallel(model).to(device)
    saved_state_dict = torch.load("./state_dicts/dfer_trained.pth", map_location=device)
    model.load_state_dict(saved_state_dict)
    model.eval()  # Set model to evaluation mode

    # Image transformation pipeline
    transform = torchvision.transforms.Compose([
        GroupResize(112),
        Stack(),
        ToTorchFormatTensor()
    ])

def state_aware(frame):
    """
    Given a frame, returns the predicted state based on the last 16 frames.
    If the model has not received 16 frames yet, returns None.
    """
    global frames

    # Convert the frame to PIL format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = [Image.fromarray(frame_rgb)]
    frames.extend(pil_frame)

    if len(frames) == 16:
        # Process and predict
        images = transform(frames)
        images = torch.reshape(images, (-1, 3, 112, 112))
        images = images.to(device)

        # Model prediction
        output = model(images).cpu().detach()
        max_idx = torch.argmax(output[0])
        return state_labels[max_idx]
    return None

def main():
    state_initialize()  # Initialize the model and settings

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Webcam could not be opened.")

    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame.")
                time.sleep(0.1)
                continue

            cur_state = state_aware(frame)
            if cur_state:
                cv2.putText(frame, f'You are now in {cur_state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

            cv2.imshow('state_awareness', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# This block allows the script to be run directly or imported as a module
if __name__ == "__main__":
    main()
