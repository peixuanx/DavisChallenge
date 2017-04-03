from scipy import misc
from scipy import ndimage
from config import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import Data_Distor

class DavisReader:
    
    def __init__(self, currentTrainImageId=0, currentTestImageId=0):
        self.davisDir = '../../DAVIS'
        trainListFileName = self.davisDir + '/ImageSets/480p/train.txt'
        trainList = open(trainListFileName)
        self.trainNames = trainList.readlines()

        testListFileName = self.davisDir + '/ImageSets/480p/val.txt'
        testList = open(testListFileName)
        self.testNames = testList.readlines()

        self.currentTrainImageSet = None
        self.currentTrainLabelSet = None
        self.currentTrainImageSetId = 0
        self.currentTrainImageSetSize = 0
        self.currentTrainImageId = currentTrainImageId
        self.currentTestImageId = currentTestImageId
        self.augMultiplier = ROTATE_NUM * CROP_HEIGHT_NUM * CROP_WIDTH_NUM * 4 * 2 # 4 for flip, 2 for mask distortion

    def next_batch(self):
        if self.currentTrainImageSet is None:
            self.augmentData()

        if self.currentTrainImageSetSize - self.currentTrainImageSetId < BATCH_SIZE:
            retImages = self.currentTrainImageSet[self.currentTrainImageSetId:]
            retLabels = self.currentTrainLabelSet[self.currentTrainImageSetId:]
            remain = BATCH_SIZE - (self.currentTrainImageSetSize - self.currentTrainImageSetId)
            self.augmentData()
            retImages = np.concatenate((retImages, self.currentTrainImageSet[:remain]))
            retLabels = np.concatenate((retLabels, self.currentTrainLabelSet[:remain]))
            self.currentTrainImageSetId = remain
        else:
            retImages = self.currentTrainImageSet[self.currentTrainImageSetId:self.currentTrainImageSetId+BATCH_SIZE]
            retLabels = self.currentTrainLabelSet[self.currentTrainImageSetId:self.currentTrainImageSetId+BATCH_SIZE]
            self.currentTrainImageSetId += BATCH_SIZE

        return retImages, retLabels
        # return retImages.tolist(), retLabels.tolist()

    def next_test(self):
        names = self.testNames[self.currentTestImageId].split()
        imageName = self.davisDir + names[0]
        labelName = self.davisDir + names[1]
        image = misc.imread(imageName)
        label = misc.imread(labelName) / 255
        self.currentTestImageId += 1

        retImages = np.zeros((BATCH_SIZE,) + image.shape)
        retLabels = np.zeros((BATCH_SIZE,) + label.shape)
        retImages[0] = image
        retLabels[0] = label
        for i in range(1,BATCH_SIZE):
            names = self.testNames[self.currentTestImageId].split()
            imageName = self.davisDir + names[0]
            labelName = self.davisDir + names[1]
            retImages[i,:,:,:] = misc.imread(imageName)
            retLabels[i,:,:] = misc.imread(labelName) / 255
            self.currentTestImageId += 1

        return retImages, retLabels

    def augmentData(self):
        # reset image set id and check training data
        self.currentTrainImageSetId = 0
        if self.currentTrainImageId >= len(self.trainNames):
            self.currentTrainImageSet = None
            self.currentTrainLabelSet = None
            return

        # read image and label
        names = self.trainNames[self.currentTrainImageId].split()
        imageName = self.davisDir + names[0]
        labelName = self.davisDir + names[1]
        image = misc.imread(imageName)
        label = misc.imread(labelName) / 255
        self.currentTrainImageId += 1

        self.currentTrainImageSet = np.zeros((self.augMultiplier, CROP_HEIGHT, CROP_WIDTH, image.shape[2]), 'uint8')
        self.currentTrainLabelSet = np.zeros((self.augMultiplier, CROP_HEIGHT, CROP_WIDTH), 'uint8')
        idx = 0
        # rotate
        for angle in np.linspace(-90, 90, ROTATE_NUM):
            imageR = self.rotateImage(angle, image)
            labelR = self.rotateImage(angle, label)

            # crop
            yBegins = np.linspace(0, image.shape[0]-CROP_HEIGHT, CROP_HEIGHT_NUM)
            yBegins = yBegins.astype(np.uint32)
            xBegins = np.linspace(0, image.shape[1]-CROP_WIDTH, CROP_WIDTH_NUM)
            xBegins = xBegins.astype(np.uint32)
            for y in yBegins:
                for x in xBegins:
                    imageRC = imageR[y:y+CROP_HEIGHT, x:x+CROP_WIDTH, :]
                    labelRC = labelR[y:y+CROP_HEIGHT, x:x+CROP_WIDTH]

                    if np.any(labelRC):
                        # no flip
                        self.currentTrainImageSet[idx,:,:,:] = imageRC
                        distortedLabel = self.data_distort(labelRC)
                        self.currentTrainLabelSet[idx,:,:] = distortedLabel[0]
                        idx += 1
                        self.currentTrainImageSet[idx,:,:,:] = imageRC
                        self.currentTrainLabelSet[idx,:,:] = distortedLabel[1]
                        idx += 1
                        
                        # flip ud
                        self.currentTrainImageSet[idx,:,:,:] = np.flipud(imageRC)
                        distortedLabel = self.data_distort(np.flipud(labelRC))
                        self.currentTrainLabelSet[idx,:,:] = distortedLabel[0]
                        idx += 1
                        self.currentTrainImageSet[idx,:,:,:] = np.flipud(imageRC)
                        self.currentTrainLabelSet[idx,:,:] = distortedLabel[1]
                        idx += 1

                        # flip lr
                        self.currentTrainImageSet[idx,:,:,:] = np.fliplr(imageRC)
                        distortedLabel = self.data_distort(np.fliplr(labelRC))
                        self.currentTrainLabelSet[idx,:,:] = distortedLabel[0]
                        idx += 1
                        self.currentTrainImageSet[idx,:,:,:] = np.fliplr(imageRC)
                        self.currentTrainLabelSet[idx,:,:] = distortedLabel[1]
                        idx += 1

                        # flip udlr
                        self.currentTrainImageSet[idx,:,:,:] = np.fliplr(np.flipud(imageRC))
                        distortedLabel = self.data_distort(np.fliplr(np.flipud(labelRC)))
                        self.currentTrainLabelSet[idx,:,:] = distortedLabel[0]
                        idx += 1
                        self.currentTrainImageSet[idx,:,:,:] = np.fliplr(np.flipud(imageRC))
                        self.currentTrainLabelSet[idx,:,:] = distortedLabel[1]
                        idx += 1
        print(idx)
        self.currentTrainImageSetSize = idx

    # Rotate the image and zoom. Angle is in degree
    def rotateImage(self, angle, image):
        retImage = ndimage.interpolation.rotate(image, angle)
        widthO = image.shape[0]
        heightO = image.shape[1]
        widthR = retImage.shape[0]
        heightR = retImage.shape[1]
        theta = angle / 180.0 * np.pi

        # find the region
        if theta >= 0:
            a = np.array([[np.cos(theta), np.sin(theta)], [2*heightO, -2*widthO]])
            b = np.array([heightO*np.sin(theta)*np.cos(theta), heightO*widthR-widthO*heightR])
        else:
            a = np.array([[1, np.tan(theta)], [2*heightO, 2*widthO]])
            b = np.array([-widthO*np.sin(theta)*np.tan(theta), heightO*widthR+widthO*heightR])
        p0 = np.linalg.solve(a, b).astype('uint32')
        p1 = np.array([widthR, heightR]) - p0
        x0 = np.min([p0[0], p1[0]])
        x1 = np.max([p0[0], p1[0]])
        y0 = np.min([p0[1], p1[1]])
        y1 = np.max([p0[1], p1[1]])

        retImage = misc.imresize(retImage[x0:x1, y0:y1], image.shape)
        return retImage

    def data_distort(self, label):
        distort = Data_Distor.Data_Distor(label)
        mask = distort.genMasks()
        return mask

def showImageLabel(image, label):
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(label)
    plt.show()

if __name__ == '__main__':
    print('test read davis')
    reader = DavisReader()
    import matplotlib.pyplot as plt

    for _ in range(200):
        # images, labels = reader.next_test()
        images, labels = reader.next_batch()
        print(images.shape)
        for i in range(BATCH_SIZE):
            image = images[i]
            label = labels[i]
            print(image.shape)
            # print (image.shape)
            # misc.imsave('label.png', label*255)
            # showImageLabel(image, label)