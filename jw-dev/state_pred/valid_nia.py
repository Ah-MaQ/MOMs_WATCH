#!/usr/bin/env python
# coding: utf-8

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#from torchmetrics.functional.classification import multiclass_recall
from time import time
import torch.nn.parallel
#import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
#import numpy as np
import datetime
from state_pred.data_loader.dataset_NIA import custom_data_loader
from state_pred.runner_helper import *


def test(val_loader, model, criterion, args):
    top1 = AverageMeter('Accuracy(WAR)', ':6.2f')
    progress = ProgressMeter(len(val_loader),
                             [top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1, 5))
            #war1 = multiclass_recall(output, target, num_classes=5, average='weighted')
            top1.update(acc1[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i, log_txt_path)

        # TODO: this should also be done with the ProgressMeter
        print('Current Accuracy: {top1.avg:.2f}'.format(top1=top1))
        with open(log_txt_path, 'a') as f:
            f.write('Current Accuracy: {top1.avg:.2f}'.format(top1=top1) + '\n')
            
    return top1.avg

class Pseudoarg():
    def __init__(self):
        self.workers = 0
        #self.epochs = 100
        #self.start_epoch = 0
        self.batch_size = 32
        self.print_freq = 10
        #self.resume = None
        self.data_set = 10
        
args = Pseudoarg()

fn_model = "./checkpoint/trained.pth"

now = datetime.datetime.now()
time_str = now.strftime("%m%d_%H%M%S_")
print("NOW", time_str)
project_path = './'
os.mkdir("log/")
log_txt_path = project_path + 'log/' + time_str + 'set' + str(args.data_set) + '-log.txt'

print("batch_size: ", args.batch_size)

# create model and load pre_trained parameters
model = GenerateModel()
#model = model.cuda()
model = torch.nn.DataParallel(model).cuda()


print("=> loading checkpoint '{}'".format(fn_model))
checkpoint = torch.load(fn_model)
model.load_state_dict(checkpoint['state_dict'])
#cudnn.benchmark = True


# Data loading code
test_data = custom_data_loader(project_path + "test.txt")

val_loader = torch.utils.data.DataLoader(test_data,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.workers,
                                         pin_memory=True)


start_time = time()
# evaluate on validation set
criterion = nn.CrossEntropyLoss().cuda()
val_acc = test(val_loader, model, criterion, args)

print(f"Took {time() - start_time:.2f} seconds for validation")
