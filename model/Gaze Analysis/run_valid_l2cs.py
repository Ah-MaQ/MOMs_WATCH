import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

import l2cs.datasets as datasets
from l2cs.utils import select_device, natural_keys, gazeto3d, angular
from l2cs.model import L2CS

from datetime import datetime


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze estimation using L2CSNet .')
    # NIA2022
    parser.add_argument(
        '--image_dir', dest='image_dir', help='Directory path for gaze images.',
        default='./test_data/Face', type=str)
    parser.add_argument(
        '--label_dir', dest='label_dir', help='Directory path for gaze labels.',
        default='./test_data/valid.label', type=str)

    # Important args -------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--dataset', dest='dataset', help='nia2022',
        default= "nia2022", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path to the folder contains models.', 
        default='models/', type=str)
    parser.add_argument(
        '--evalpath', dest='evalpath', help='path for the output evaluating gaze test.',
        default="evaluation/L2CS-nia2022", type=str)
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=1, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], ''ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args


def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = select_device(args.gpu_id, batch_size=args.batch_size)
    batch_size=args.batch_size
    arch=args.arch
    data_set=args.dataset
    evalpath =args.evalpath
    snapshot_path = args.snapshot
    #bins=args.bins
    #angle=args.angle
    #bin_width=args.bin_width

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    if data_set=="nia2022":
        print("Start testing dataset=nia2022----------------------------------------")
        print("test configuration = gpu_id={}, batch_size={}, model_arch={}".format(gpu, batch_size, arch))
        print("Starting at: ", dt_string)
        dataset=datasets.NIA2022(args.label_dir,args.image_dir, transformations, 180, 4, train=False)
        test_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)      

        if not os.path.exists(evalpath):
            os.makedirs(evalpath)

        # list all epochs for testing
        folder = os.listdir(snapshot_path)
        folder.sort(key=natural_keys)
        softmax = nn.Softmax(dim=1)
        with open(os.path.join(evalpath,data_set+".log"), 'w') as outfile:
            configuration = f"\ntest configuration = gpu_id={gpu}, batch_size={batch_size}, model_arch={arch}\nStart testing dataset={data_set}----------------------------------------\n"
            print(configuration)
            outfile.write(configuration)
            epoch_list=[]
            avg_yaw=[]
            avg_pitch=[]
            avg_MAE=[]
            for epochs in folder:
                # Base network structure
                model=getArch(arch, 90)
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

                        label_pitch = cont_labels[:,0].float()*np.pi/180
                        label_yaw = cont_labels[:,1].float()*np.pi/180

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

                        pitch_predicted = pitch_predicted*np.pi/180
                        yaw_predicted = yaw_predicted*np.pi/180

                        for p,y,pl,yl in zip(pitch_predicted,yaw_predicted,label_pitch,label_yaw):
                            this_err = 6.8 + angular(gazeto3d([p,y]), gazeto3d([pl,yl]))**0.25 
                            avg_error += this_err

                        if (j+1) % 1000 == 0:
                            print('Iter [%d/%d] This error %.4f, '
                                'Mean Angular Error %.4f' % (
                                    j+1,
                                    len(dataset)//batch_size,
                                    this_err,
                                    avg_error/total
                                    #sum_loss_pitch_gaze/iter_gaze,
                                    #sum_loss_yaw_gaze/iter_gaze
                                )
                                )                    
                x = ''.join(filter(lambda i: i.isdigit(), epochs))
                avg_MAE.append(avg_error/total)
                print(f"[---{args.dataset}] Total Num:{total},MAE:{avg_error/total}\n")
                #outfile.write(loger)
                #print(loger)

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("done at =", dt_string)
