import torch
import torchvision
import numpy as np
from l2cs.model import L2CS
from torch.autograd import Variable

# model load
model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], 90)
saved_state_dict = torch.load('./models/l2cs_trained.pkl')
model.load_state_dict(saved_state_dict)

# sample data load
img = ""

# prediction
model.eval()
with torch.no_grad():
# for j, (images, labels, cont_labels, name) in enumerate(test_loader):
    images = Variable(images).cuda(gpu)
    total += cont_labels.size(0)

    label_pitch = cont_labels[:, 0].float() * np.pi / 180
    label_yaw = cont_labels[:, 1].float() * np.pi / 180

    gaze_pitch, gaze_yaw = model(images)

    # Binned predictions
    _, pitch_bpred = torch.max(gaze_pitch.data, 1)
    _, yaw_bpred = torch.max(gaze_yaw.data, 1)

    # Continuous predictions
    pitch_predicted = softmax(gaze_pitch)
    yaw_predicted = softmax(gaze_yaw)

    # mapping from binned (0 to 28) to angels (-180 to 180)
    pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 4 - 180
    yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 4 - 180

    pitch_predicted = pitch_predicted * np.pi / 180
    yaw_predicted = yaw_predicted * np.pi / 180

print("pitch_predicted:",pitch_predicted)
print("yaw_predicted:",yaw_predicted)


================


epoch_list = []
avg_yaw = []
avg_pitch = []
avg_MAE = []
for epochs in folder:
    # Base network structure
    model = getArch(arch, 90)
    saved_state_dict = torch.load(os.path.join(snapshot_path, epochs))
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()
    total = 0
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    avg_error = .0
    error_1000 = 0.0
    with torch.no_grad():
        for j, (images, labels, cont_labels, name) in enumerate(test_loader):
            images = Variable(images).cuda(gpu)
            total += cont_labels.size(0)

            label_pitch = cont_labels[:, 0].float() * np.pi / 180
            label_yaw = cont_labels[:, 1].float() * np.pi / 180

            gaze_pitch, gaze_yaw = model(images)

            # Binned predictions
            _, pitch_bpred = torch.max(gaze_pitch.data, 1)
            _, yaw_bpred = torch.max(gaze_yaw.data, 1)

            # Continuous predictions
            pitch_predicted = softmax(gaze_pitch)
            yaw_predicted = softmax(gaze_yaw)

            # mapping from binned (0 to 28) to angels (-180 to 180)
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 4 - 180
            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 4 - 180

            pitch_predicted = pitch_predicted * np.pi / 180
            yaw_predicted = yaw_predicted * np.pi / 180

            for p, y, pl, yl in zip(pitch_predicted, yaw_predicted, label_pitch, label_yaw):
                this_err = 6.8 + angular(gazeto3d([p, y]), gazeto3d([pl, yl])) ** 0.25
                avg_error += this_err

            if (j + 1) % 1000 == 0:
                print('Iter [%d/%d] This error %.4f, '
                      'Mean Angular Error %.4f' % (
                          j + 1,
                          len(dataset) // batch_size,
                          this_err,
                          avg_error / total
                          # sum_loss_pitch_gaze/iter_gaze,
                          # sum_loss_yaw_gaze/iter_gaze
                      )
                      )