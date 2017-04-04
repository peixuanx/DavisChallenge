from scipy import misc
from scipy import ndimage
from config import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import Data_Distor

class DavisReader:

    def __init__(self, currentTrainImageId=0, currentTestImageId=0):
        #self.davisDir = '../../DAVIS'
        self.datalist = open('./datalist.txt').readlines()
        self.trainDir = self.datalist[2].split()[2]
        print(self.trainDir)
        self.trainImgFolder = self.datalist[2].split()[3]
        self.trainLabFolder = self.datalist[2].split()[4]
        trainListFileName = self.trainDir + '/ImageSets/train.txt'
        trainList = open(trainListFileName)
        self.trainNames = trainList.readlines()

        self.testDir = self.datalist[3].split()[2]
        testListFileName = self.testDir + 'test.txt'
        testList = open(testListFileName)
        self.testNames = testList.readlines()

        self.currentTrainImageSet = None
        self.currentTrainLabelSet = None
        self.trainImageSetUsedTime = 0
        self.currentTrainImageSetSize = 0
        # self.currentTrainImageId = currentTrainImageId
        self.currentTestImageId = currentTestImageId
        self.augMultiplier = ROTATE_NUM * CROP_HEIGHT_NUM * CROP_WIDTH_NUM * 4 * 1 # 4 for flip, 2 for mask distortion

    def next_batch(self):
        if self.currentTrainImageSet is None:
            self.augmentData()

        if self.currentTrainImageSetSize * 0.7 < self.trainImageSetUsedTime * BATCH_SIZE:
            self.augmentData()

        self.trainImageSetUsedTime += 1
        id = np.random.randint(self.currentTrainImageSetSize, size=BATCH_SIZE)
        retImages = self.currentTrainImageSet[id]
        retLabels = self.currentTrainLabelSet[id]

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
        # print(len(self.trainNames))
        self.trainImageSetUsedTime = 0
        self.currentTrainImageSet = np.zeros((self.augMultiplier*10, CROP_HEIGHT, CROP_WIDTH,
                                                4), 'uint8')
        self.currentTrainLabelSet = np.zeros((self.augMultiplier*10, CROP_HEIGHT, CROP_WIDTH,
                                                NUM_CLASSES), 'uint8')
        idx = 0
        for imageId in np.random.randint(len(self.trainNames), size=10):
            # print("image id ", imageId)
            # read image and label
            names = self.trainNames[imageId].split()
            imageName = self.trainDir + self.trainImgFolder + names[0] + '.jpg'
            labelName = self.trainDir + self.trainLabFolder + names[0] + '.png'
            image = misc.imread(imageName)
            label = misc.imread(labelName) / 255

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
                            self.currentTrainImageSet[idx,:,:,0:3] = imageRC
                            distortedLabel = self.data_distort(labelRC)
                            self.currentTrainImageSet[idx,:,:,-1] = distortedLabel[0]
                            self.currentTrainLabelSet[idx,:,:,0] = 1-labelRC
                            self.currentTrainLabelSet[idx,:,:,1] = labelRC
                            idx += 1
                            '''
                            self.currentTrainImageSet[idx,:,:,:] = imageRC
                            self.currentTrainLabelSet[idx,:,:] = distortedLabel[1]
                            idx += 1
                            '''

                            # flip ud
                            self.currentTrainImageSet[idx,:,:,0:3] = np.flipud(imageRC)
                            distortedLabel = self.data_distort(np.flipud(labelRC))
                            self.currentTrainImageSet[idx,:,:,-1] = distortedLabel[0]
                            self.currentTrainLabelSet[idx,:,:,0] = np.flipud(1-labelRC)
                            self.currentTrainLabelSet[idx,:,:,1] = np.flipud(labelRC)
                            idx += 1
                            '''
                            self.currentTrainImageSet[idx,:,:,:] = np.flipud(imageRC)
                            self.currentTrainLabelSet[idx,:,:] = distortedLabel[1]
                            idx += 1
                            '''

                            # flip lr
                            self.currentTrainImageSet[idx,:,:,0:3] = np.fliplr(imageRC)
                            distortedLabel = self.data_distort(np.fliplr(labelRC))
                            self.currentTrainImageSet[idx,:,:,-1] = distortedLabel[0]
                            self.currentTrainLabelSet[idx,:,:,0] = np.fliplr(1-labelRC)
                            self.currentTrainLabelSet[idx,:,:,1] = np.fliplr(labelRC)
                            idx += 1
                            '''
                            self.currentTrainImageSet[idx,:,:,:] = np.fliplr(imageRC)
                            self.currentTrainLabelSet[idx,:,:] = distortedLabel[1]
                            idx += 1
                            '''

                            # flip udlr
                            self.currentTrainImageSet[idx,:,:,0:3] = np.fliplr(np.flipud(imageRC))
                            distortedLabel = self.data_distort(np.fliplr(np.flipud(labelRC)))
                            self.currentTrainImageSet[idx,:,:,-1] = distortedLabel[0]
                            self.currentTrainLabelSet[idx,:,:,0] = np.fliplr(np.flipud(1-labelRC))
                            self.currentTrainLabelSet[idx,:,:,1] = np.fliplr(np.flipud(labelRC))
                            idx += 1
                            '''
                            self.currentTrainImageSet[idx,:,:,:] = np.fliplr(np.flipud(imageRC))
                            self.currentTrainLabelSet[idx,:,:] = distortedLabel[1]
                            idx += 1
                            '''
            # print("idx", idx)
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

def read_list():
    f = open('./datalist.txt')


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

    for _ in range(1):
        # images, labels = reader.next_test()
        images, labels = reader.next_batch()
        print(images.shape)
        print(labels.shape)
        '''
        for i in range(BATCH_SIZE):
            image = images[i,:,:,:3]
            label = labels[i,:,:,0]
            print(image.shape)
            # print (image.shape)
            misc.imsave('label.png', label*255)
            misc.imsave('image.png', image)
            showImageLabel(image, label)
        '''