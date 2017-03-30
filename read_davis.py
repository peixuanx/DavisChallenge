from scipy import misc
from scipy import ndimage
from config import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class DavisReader:
    
    def __init__(self, currentTrainImageId=0, currentTestImageId=0):
        self.davisDir = '../../DAVIS'
        trainListFileName = self.davisDir + '/ImageSets/480p/train.txt'
        trainList = open(trainListFileName)
        self.trainNames = trainList.readlines()

        testListFileName = self.davisDir + '/ImageSets/480p/val.txt'
        testList = open(testListFileName)
        self.testNames = testList.readlines()

        # self.trainlist_filename = '../Semantic-Segmentation/VOC2011/ImageSets/Segmentation/train.txt'
        # self.testlist_filename = '../Semantic-Segmentation/VOC2011/ImageSets/Segmentation/val.txt'
        # self.image_dir = '../Semantic-Segmentation/VOC2011/JPEGImages/'
        # self.label_dir = '../Semantic-Segmentation/VOC2011/SegmentationClass/'
        # train_list = open(self.trainlist_filename)
        # self.train_names = train_list.readlines()
        # self.train_names = [x.strip() for x in self.train_names]
        # test_list = open(self.testlist_filename)
        # self.test_names = test_list.readlines()
        # self.test_names = [x.strip() for x in self.test_names]

        self.currentTrainImageSet = None
        self.currentTrainLabelSet = None
        self.currentTrainImageSetId = 0
        self.currentTrainImageId = currentTrainImageId
        self.currentTestImageId = currentTestImageId
        self.augMultiplier = ROTATE_NUM * CROP_HEIGHT_NUM * CROP_WIDTH_NUM * 4 * 2 # 4 for flip, 2 for mask distortion

    def next_batch(self):
        if self.currentTrainImageSet is None:
            self.augmentData()

        if self.augMultiplier - self.currentTrainImageSetId < BATCH_SIZE:
            retImages = self.currentTrainImageSet[self.currentTrainImageSetId:]
            retLabels = self.currentTrainLabelSet[self.currentTrainImageSetId:]
            remain = BATCH_SIZE - (self.augMultiplier - self.currentTrainImageSetId)
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
        # showImageLabel(retImages[0,:,:,:],retLabels[0,:,:])
        # showImageLabel(image,label)
        # print(image == retImages[0,:,:,:])
        # misc.imsave('image.jpg', image)
        # misc.imsave('return.jpg', retImages[0])
        for i in range(1,BATCH_SIZE):
            names = self.testNames[self.currentTestImageId].split()
            imageName = self.davisDir + names[0]
            labelName = self.davisDir + names[1]
            retImages[i,:,:,:] = misc.imread(imageName)
            retLabels[i,:,:] = misc.imread(labelName) / 255
            # showImageLabel(retImages[i],retLabels[i])
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
        # showImageLabel(image, label)

        self.currentTrainImageSet = np.zeros((self.augMultiplier, CROP_HEIGHT, CROP_WIDTH, image.shape[2]), 'uint8')
        self.currentTrainLabelSet = np.zeros((self.augMultiplier, CROP_HEIGHT, CROP_WIDTH), 'uint8')
        idx = 0
        # rotate
        for angle in np.linspace(-90, 90, ROTATE_NUM):
            imageR = self.rotateImage(angle, image)
            labelR = self.rotateImage(angle, label)
            # showImageLabel(imageR, labelR)

            # crop
            yBegins = np.linspace(0, image.shape[0]-CROP_HEIGHT, CROP_HEIGHT_NUM)
            yBegins = yBegins.astype(np.uint32)
            xBegins = np.linspace(0, image.shape[1]-CROP_WIDTH, CROP_WIDTH_NUM)
            xBegins = xBegins.astype(np.uint32)
            for y in yBegins:
                for x in xBegins:
                    imageRC = imageR[y:y+CROP_HEIGHT, x:x+CROP_WIDTH, :]
                    labelRC = labelR[y:y+CROP_HEIGHT, x:x+CROP_WIDTH]
                    # showImageLabel(imageRC, labelRC)

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

                    # for i in [8, 7, 6, 5, 4, 3, 2, 1]:
                    #     showImageLabel(self.currentTrainImageSet[idx-i], self.currentTrainLabelSet[idx-i])


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
        label = label.reshape((1,)+label.shape)
        return np.concatenate((label,label))

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

    for _ in range(10):
        images, labels = reader.next_test()
        images, labels = reader.next_batch()
        # print(images.shape)
        for i in range(BATCH_SIZE):
            image = images[i]
            label = labels[i]
            # print (image.shape)
            # showImageLabel(image, label)