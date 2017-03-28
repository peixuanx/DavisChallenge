from scipy import misc
from config import *
import numpy as np
from PIL import Image

class PascalReader:
    
    def __init__(self, current_train_image=0, current_test_image=0):
        self.trainlist_filename = '../Semantic-Segmentation/VOC2011/ImageSets/Segmentation/train.txt'
        self.testlist_filename = '../Semantic-Segmentation/VOC2011/ImageSets/Segmentation/val.txt'
        self.image_dir = '../Semantic-Segmentation/VOC2011/JPEGImages/'
        self.label_dir = '../Semantic-Segmentation/VOC2011/SegmentationClass/'
        train_list = open(self.trainlist_filename)
        self.train_names = train_list.readlines()
        self.train_names = [x.strip() for x in self.train_names]
        test_list = open(self.testlist_filename)
        self.test_names = test_list.readlines()
        self.test_names = [x.strip() for x in self.test_names]
        self.current_train_image = current_train_image;
        self.current_test_image = current_test_image;
        # self.patch_length = 224;
        self.padding_length = INIT_PADDING

    def next_batch(self, batch_size=BATCH_SIZE):
        imageName = self.image_dir + self.train_names[self.current_train_image] + '.jpg'
        labelName = self.label_dir + self.train_names[self.current_train_image] + '.png'
        #print(imageName)
        image = misc.imread(imageName)
        label = Image.open(labelName)
        label = np.array(label, dtype=np.uint8)
        HEIGHT = image.shape[0]
        WIDTH = image.shape[1]
        # print 'image size: ',HEIGHT,WIDTH
        tmpImage = np.zeros((1, 2*self.padding_length+image.shape[0], 2*self.padding_length+image.shape[1], image.shape[2]))
        tmpImage[0,self.padding_length:self.padding_length+image.shape[0],self.padding_length:self.padding_length+image.shape[1],:] = image[:,:,:]
        tmpLabels = np.zeros((1, label.shape[0], label.shape[1], 21))
        for i in range(21):
            tmpLabels[0, :, :, i] = (label == i)
        
        tmpLabel = np.zeros((1, 2*self.padding_length+label.shape[0], 2*self.padding_length+label.shape[1], NUM_CLASSES))
        tmpLabel[0,self.padding_length:self.padding_length+label.shape[0],self.padding_length:self.padding_length+label.shape[1],:] = tmpLabels[0,:,:,:]
        '''
        for i in range(21):
            print(np.sum(tmpLabel[0,:,:,i] == 1))
        '''
        self.current_train_image += 1

        return tmpImage.tolist(), tmpLabel.tolist(), self.train_names[self.current_train_image-1]

        # for x in np.linspace(0, image.shape[0]-self.padding_length-1 , 4, dtype = np.int16):
        #     for y in np.linspace(0, image.shape[1]-self.padding_length-1 , 5, dtype = np.int16):
        #         tmpImage = np.zeros((1, 2*self.padding_length+self.patch_length, 2*self.padding_length+self.patch_length, 3))
        #         tmpImage[0,self.padding_length,self.padding_length] = image[x:x+elf.patch_length,p:p+elf.patch_length,:]
        #         tmpLabel = np.zeros((1, self.patch_length, self.patch_length, 3))
        #         tmpLabel[0,:,:,:] = label[x:x+elf.patch_length,p:p+elf.patch_length,:]


    def next_test(self):
        imageName = self.image_dir + self.test_names[self.current_test_image] + '.jpg'
        labelName = self.label_dir + self.test_names[self.current_test_image] + '.png'
        image = misc.imread(imageName)
        label = Image.open(labelName)
        label = np.array(label, dtype=np.uint8)
        tmpImage = np.zeros((1, 2*self.padding_length+image.shape[0], 2*self.padding_length+image.shape[1], image.shape[2]))
        tmpImage[0,self.padding_length:self.padding_length+image.shape[0],self.padding_length:self.padding_length+image.shape[1],:] = image[:,:,:]
        tmpLabels = np.zeros((1, label.shape[0], label.shape[1], 21))
        for i in range(21):
            tmpLabels[0, :, :, i] = (label == i)
        tmpLabel = np.zeros((1, 2*self.padding_length+label.shape[0], 2*self.padding_length+label.shape[1], NUM_CLASSES))
        tmpLabel[0,self.padding_length:self.padding_length+label.shape[0],self.padding_length:self.padding_length+label.shape[1],:] = tmpLabels[0,:,:,:]
        self.current_test_image += 1
        return tmpImage.tolist(), tmpLabel.tolist(), self.test_names[self.current_test_image-1]
