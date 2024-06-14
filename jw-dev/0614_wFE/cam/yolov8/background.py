import shutil

path = '/media/jinsu/My Passport/yolov8/data/train/'

for i in range(9700):
    shutil.copy('background.png',path+'/images/background_'+str(i)+'.png')
    shutil.copy('background.txt',path+'/labels/background_'+str(i)+'.txt')
