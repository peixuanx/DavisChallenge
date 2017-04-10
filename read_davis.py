from scipy import misc
from scipy import ndimage
from config import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import Data_Distor
import epicflow

class DavisReader:

    def __init__(self, currentTrainImageId=0, currentTestImageId=0, mode="random"):
        self.davisDir = DATA_DIR
        print(self.davisDir)
        trainListFileName = self.davisDir + '/ImageSets/480p/train.txt'
        trainList = open(trainListFileName)
        self.trainNames = trainList.readlines()

        testListFileName = self.davisDir + '/ImageSets/480p/val.txt'
        testList = open(testListFileName)
        self.testNames = testList.readlines()

        self.currentTrainImageSet = None
        self.currentTrainLabelSet = None
        self.trainImageSetUsedTime = 0
        self.currentTrainImageSetSize = 0
        # self.currentTrainImageId = currentTrainImageId
        self.currentTestImageId = currentTestImageId
        self.augMultiplier = ROTATE_NUM * CROP_HEIGHT_NUM * CROP_WIDTH_NUM * 4 * 2 # 4 for flip, 2 for mask distortion
        self.videoAugMultiplier = ROTATE_NUM * 4 * 2 # 4 for flip, 2 for mask distortion

        self.mode = mode
        self.videoId = 0
        self.videoSize = 0

    def next_batch(self):
        if self.mode == "random":
            if self.currentTrainImageSet is None:
                self.augmentData()

            if self.currentTrainImageSetSize * 0.7 < self.trainImageSetUsedTime * BATCH_SIZE:
                self.augmentData()

            self.trainImageSetUsedTime += 1
            id = np.random.randint(self.currentTrainImageSetSize, size=BATCH_SIZE)
            retImages = self.currentTrainImageSet[id]
            retLabels = self.currentTrainLabelSet[id]

            return retImages, retLabels

        elif self.mode == "video":
            if self.currentTrainImageSet is None:
                self.videoSize = self.findVideoSize(self.trainNames, self.videoId)
                self.VideoAugmentData()

            if self.trainImageSetUsedTime == self.videoAugMultiplier:
                self.videoSize = self.findVideoSize(self.trainNames, self.videoId)
                self.VideoAugmentData()

            rotateId = self.trainImageSetUsedTime%ROTATE_NUM
            id = np.arange(int(self.trainImageSetUsedTime/ROTATE_NUM), self.videoAugMultiplier/ROTATE_NUM*self.videoSize, self.videoAugMultiplier/ROTATE_NUM)
            retImages = self.currentTrainImageSet[rotateId][id]
            retLabels = self.currentTrainLabelSet[rotateId][id]
            self.trainImageSetUsedTime += 1
            return retImages, retLabels # return video size?
        # return retImages.tolist(), retLabels.tolist()

    def next_test(self):
        names = self.testNames[self.currentTestImageId].split()
        imageName = self.davisDir + names[0]
        labelName = self.davisDir + names[1]
        image = misc.imread(imageName)
        label = misc.imread(labelName) / 255
        self.currentTestImageId += 1

        retImages = np.zeros((BATCH_SIZE,) + image.shape + np.array([0,0,0,1]))
        retLabels = np.zeros((BATCH_SIZE,) + label.shape + (2,))
        retImages[0,:,:,0:3] = image
        retImages[0,:,:,-1] = self.data_distort(label)[:,:,0]
        retLabels[0,:,:,0] = 1-label
        retLabels[0,:,:,1] = label
        for i in range(1,BATCH_SIZE):
            names = self.testNames[self.currentTestImageId].split()
            imageName = self.davisDir + names[0]
            labelName = self.davisDir + names[1]
            image = misc.imread(imageName)
            label = misc.imread(labelName) / 255
            retImages[0,:,:,0:3] = image
            retImages[0,:,:,-1] = self.data_distort(label)[:,:,0]
            retLabels[0,:,:,0] = 1-label
            retLabels[0,:,:,1] = label
            self.currentTestImageId += 1

        return retImages, retLabels, names[0].split('/')[-1]

    def augmentData(self):
        # reset image set id and check training data
        # print(len(self.trainNames))
        self.trainImageSetUsedTime = 0
        if self.mode == "random":
            self.currentTrainImageSet = np.zeros((self.augMultiplier*10, CROP_HEIGHT, CROP_WIDTH,
                                                    4), 'uint8')
            self.currentTrainLabelSet = np.zeros((self.augMultiplier*10, CROP_HEIGHT, CROP_WIDTH,
                                                    NUM_CLASSES), 'uint8')
            imageIdList = np.random.randint(len(self.trainNames), size=10)
        elif self.mode == "video":
            imageIdList = range(self.videoId+1, self.videoId+self.videoSize)
            names = self.trainNames[imageId].split()
            imageName = self.davisDir + names[0]
            labelName = self.davisDir + names[1]
            image = misc.imread(imageName)
            label = misc.imread(labelName)
            distLabel = self.data_distort(label)*255

            # self.currentTrainImageSet = np.zeros((self.videoSize*self.videoAugMultiplier, image.shape[0], image.shape[1], 7))
            # self.currentTrainLabelSet = np.zeros((self.videoSize*self.videoAugMultiplier, image.shape[0], image.shape[1], NUM_CLASSES))
            self.currentTrainImageSet = []
            self.currentTrainLabelSet = []

            self.currentTrainImageSet[0,:,:,:3] = image
            self.currentTrainImageSet[0,:,:,3] = distLabel
            self.currentTrainLabelSet[0,:,:,]

        idx = 0
        nthImage = 0
        for imageId in imageIdList:
            # print("image id ", imageId)
            # read image and label
            names = self.trainNames[imageId].split()
            imageName = self.davisDir + names[0]
            labelName = self.davisDir + names[1]
            image = misc.imread(imageName)
            label = misc.imread(labelName) / 255
            if np.any(label):
                distLabel = self.data_distort(label)
            else:
                distLabel = label

            print(names[0])
            # rotate
            for angle in np.linspace(-90, 90, ROTATE_NUM):
                imageR, scale = self.rotateImage(angle, image)
                labelR, scale = self.rotateImage(angle, label)
                distLabelR, scale = self.rotateImage(angle, distLabel)


                # crop
                yBegins = np.linspace(0, image.shape[0]-CROP_HEIGHT, CROP_HEIGHT_NUM)
                yBegins = yBegins.astype(np.uint32)
                xBegins = np.linspace(0, image.shape[1]-CROP_WIDTH, CROP_WIDTH_NUM)
                xBegins = xBegins.astype(np.uint32)
                for y in yBegins:
                    for x in xBegins:
                        imageRC = imageR[y:y+CROP_HEIGHT, x:x+CROP_WIDTH, :]
                        labelRC = labelR[y:y+CROP_HEIGHT, x:x+CROP_WIDTH]
                        distLabelRC = distLabelR[y:y+CROP_HEIGHT, x:x+CROP_WIDTH]


                        if np.any(labelRC) :
                            # no flip
                            self.currentTrainImageSet[idx,:,:,0:3] = imageRC
                            self.currentTrainImageSet[idx,:,:,-1] = distLabelRC[:,:,0]
                            self.currentTrainLabelSet[idx,:,:,0] = 1-labelRC
                            self.currentTrainLabelSet[idx,:,:,1] = labelRC
                            idx += 1

                            self.currentTrainImageSet[idx,:,:,0:3] = imageRC
                            self.currentTrainImageSet[idx,:,:,-1] = distLabelRC[:,:,1]
                            self.currentTrainLabelSet[idx,:,:,0] = 1-labelRC
                            self.currentTrainLabelSet[idx,:,:,1] = labelRC
                            idx += 1


                            # flip ud
                            self.currentTrainImageSet[idx,:,:,0:3] = np.flipud(imageRC)
                            self.currentTrainImageSet[idx,:,:,-1] = np.flipud(distLabelRC[:,:,0])
                            self.currentTrainLabelSet[idx,:,:,0] = np.flipud(1-labelRC)
                            self.currentTrainLabelSet[idx,:,:,1] = np.flipud(labelRC)
                            idx += 1

                            self.currentTrainImageSet[idx,:,:,0:3] = np.flipud(imageRC)
                            self.currentTrainImageSet[idx,:,:,-1] = np.flipud(distLabelRC[:,:,1])
                            self.currentTrainLabelSet[idx,:,:,0] = np.flipud(1-labelRC)
                            self.currentTrainLabelSet[idx,:,:,1] = np.flipud(labelRC)
                            idx += 1


                            # flip lr
                            self.currentTrainImageSet[idx,:,:,0:3] = np.fliplr(imageRC)
                            self.currentTrainImageSet[idx,:,:,-1] = np.fliplr(distLabelRC[:,:,0])
                            self.currentTrainLabelSet[idx,:,:,0] = np.fliplr(1-labelRC)
                            self.currentTrainLabelSet[idx,:,:,1] = np.fliplr(labelRC)
                            idx += 1

                            self.currentTrainImageSet[idx,:,:,0:3] = np.fliplr(imageRC)
                            self.currentTrainImageSet[idx,:,:,-1] = np.fliplr(distLabelRC[:,:,1])
                            self.currentTrainLabelSet[idx,:,:,0] = np.fliplr(1-labelRC)
                            self.currentTrainLabelSet[idx,:,:,1] = np.fliplr(labelRC)
                            idx += 1


                            # flip udlr
                            self.currentTrainImageSet[idx,:,:,0:3] = np.fliplr(np.flipud(imageRC))
                            self.currentTrainImageSet[idx,:,:,-1] = np.fliplr(np.flipud(distLabelRC[:,:,0]))
                            self.currentTrainLabelSet[idx,:,:,0] = np.fliplr(np.flipud(1-labelRC))
                            self.currentTrainLabelSet[idx,:,:,1] = np.fliplr(np.flipud(labelRC))
                            idx += 1

                            self.currentTrainImageSet[idx,:,:,0:3] = np.fliplr(np.flipud(imageRC))
                            self.currentTrainImageSet[idx,:,:,-1] = np.fliplr(np.flipud(distLabelRC[:,:,1]))
                            self.currentTrainLabelSet[idx,:,:,0] = np.fliplr(np.flipud(1-labelRC))
                            self.currentTrainLabelSet[idx,:,:,1] = np.fliplr(np.flipud(labelRC))
                            idx += 1

            # print("idx", idx)
            # self.filenameList.append(names[1])
            nthImage += 1
        self.currentTrainImageSetSize = idx

    def VideoAugmentData(self):
        # reset image set id and check training data
        # print(len(self.trainNames))
        imageIdList = range(self.videoId, self.videoId+self.videoSize)
        self.videoId += self.videoSize
        self.trainImageSetUsedTime = 0
        self.currentTrainImageSet = []
        self.currentTrainLabelSet = []

        # idx = 0
        nthImage = 0
        for imageId in imageIdList:
            # print("image id ", imageId)
            # read image and label
            names = self.trainNames[imageId].split()
            imageName = self.davisDir + names[0]
            labelName = self.davisDir + names[1]
            image = misc.imread(imageName)
            label = misc.imread(labelName) / 255
            if np.any(label):
                distLabel = self.data_distort(label)
            else:
                distLabel = label

            if nthImage > 0:
                tmp = epicflow.computeOpticalFlow(self.davisDir+self.trainNames[imageId-1].split()[0], imageName)
                edge = tmp[:,:,0:1]
                flow = tmp[:,:,1:]
                # edge = np.zeros((image.shape[0], image.shape[1], 1))
                # flow = np.zeros((image.shape[0], image.shape[1], 2))

            print(names[0])
            # rotate
            angles = np.linspace(-90, 90, ROTATE_NUM)
            for rid in range(ROTATE_NUM):
                angle = angles[rid]
                imageR, scale = self.rotateImage(angle, image)
                labelR, scale = self.rotateImage(angle, label)
                distLabelR, scale = self.rotateImage(angle, distLabel)

                if nthImage == 0:
                    # self.currentTrainImageSet = np.zeros((self.videoSize*2, image.shape[0], image.shape[1], 7))
                    # self.currentTrainLabelSet = np.zeros((self.videoSize*2, image.shape[0], image.shape[1], NUM_CLASSES), 'uint8')
                    self.currentTrainImageSet.append(np.zeros((self.videoSize*self.videoAugMultiplier/ROTATE_NUM, image.shape[0], image.shape[1], 7)))
                    self.currentTrainLabelSet.append(np.zeros((self.videoSize*self.videoAugMultiplier/ROTATE_NUM, image.shape[0], image.shape[1], NUM_CLASSES), 'uint8'))


                idx = nthImage * 8
                # no flip
                print(self.currentTrainImageSet[rid].shape)
                print(imageR.shape)
                self.currentTrainImageSet[rid][idx,:,:,0:3] = imageR
                self.currentTrainImageSet[rid][idx,:,:,3] = distLabelR[:,:,0]
                if nthImage > 0:
                    self.currentTrainImageSet[rid][idx,:,:,4:5] = edge
                    self.currentTrainImageSet[rid][idx,:,:,5:] = epicflow.affine(flow, angle, scale, 0)
                self.currentTrainLabelSet[rid][idx,:,:,0] = 1-labelR
                self.currentTrainLabelSet[rid][idx,:,:,1] = labelR
                idx += 1

                self.currentTrainImageSet[rid][idx,:,:,0:3] = imageR
                self.currentTrainImageSet[rid][idx,:,:,3] = distLabelR[:,:,1]
                if nthImage > 0:
                    self.currentTrainImageSet[rid][idx,:,:,4:5] = edge
                    self.currentTrainImageSet[rid][idx,:,:,5:] = epicflow.affine(flow, angle, scale, 0)
                self.currentTrainLabelSet[rid][idx,:,:,0] = 1-labelR
                self.currentTrainLabelSet[rid][idx,:,:,1] = labelR
                idx += 1


                # flip ud
                self.currentTrainImageSet[rid][idx,:,:,0:3] = np.flipud(imageR)
                self.currentTrainImageSet[rid][idx,:,:,3] = np.flipud(distLabelR[:,:,0])
                if nthImage > 0:
                    self.currentTrainImageSet[rid][idx,:,:,4:5] = edge
                    self.currentTrainImageSet[rid][idx,:,:,5:] = epicflow.affine(flow, angle, scale, 1)
                self.currentTrainLabelSet[rid][idx,:,:,0] = np.flipud(1-labelR)
                self.currentTrainLabelSet[rid][idx,:,:,1] = np.flipud(labelR)
                idx += 1

                self.currentTrainImageSet[rid][idx,:,:,0:3] = np.flipud(imageR)
                self.currentTrainImageSet[rid][idx,:,:,3] = np.flipud(distLabelR[:,:,1])
                if nthImage > 0:
                    self.currentTrainImageSet[rid][idx,:,:,4:5] = edge
                    self.currentTrainImageSet[rid][idx,:,:,5:] = epicflow.affine(flow, angle, scale, 1)
                self.currentTrainLabelSet[rid][idx,:,:,0] = np.flipud(1-labelR)
                self.currentTrainLabelSet[rid][idx,:,:,1] = np.flipud(labelR)
                idx += 1


                # flip lr
                self.currentTrainImageSet[rid][idx,:,:,0:3] = np.fliplr(imageR)
                self.currentTrainImageSet[rid][idx,:,:,3] = np.fliplr(distLabelR[:,:,0])
                if nthImage > 0:
                    self.currentTrainImageSet[rid][idx,:,:,4:5] = edge
                    self.currentTrainImageSet[rid][idx,:,:,5:] = epicflow.affine(flow, angle, scale, 2)
                self.currentTrainLabelSet[rid][idx,:,:,0] = np.fliplr(1-labelR)
                self.currentTrainLabelSet[rid][idx,:,:,1] = np.fliplr(labelR)
                idx += 1

                self.currentTrainImageSet[rid][idx,:,:,0:3] = np.fliplr(imageR)
                self.currentTrainImageSet[rid][idx,:,:,3] = np.fliplr(distLabelR[:,:,1])
                if nthImage > 0:
                    self.currentTrainImageSet[rid][idx,:,:,4:5] = edge
                    self.currentTrainImageSet[rid][idx,:,:,5:] = epicflow.affine(flow, angle, scale, 2)
                self.currentTrainLabelSet[rid][idx,:,:,0] = np.fliplr(1-labelR)
                self.currentTrainLabelSet[rid][idx,:,:,1] = np.fliplr(labelR)
                idx += 1


                # flip udlr
                self.currentTrainImageSet[rid][idx,:,:,0:3] = np.fliplr(np.flipud(imageR))
                self.currentTrainImageSet[rid][idx,:,:,3] = np.fliplr(np.flipud(distLabelR[:,:,0]))
                if nthImage > 0:
                    self.currentTrainImageSet[rid][idx,:,:,4:5] = edge
                    self.currentTrainImageSet[rid][idx,:,:,5:] = epicflow.affine(flow, angle, scale, 3)
                self.currentTrainLabelSet[rid][idx,:,:,0] = np.fliplr(np.flipud(1-labelR))
                self.currentTrainLabelSet[rid][idx,:,:,1] = np.fliplr(np.flipud(labelR))
                idx += 1

                self.currentTrainImageSet[rid][idx,:,:,0:3] = np.fliplr(np.flipud(imageR))
                self.currentTrainImageSet[rid][idx,:,:,3] = np.fliplr(np.flipud(distLabelR[:,:,1]))
                if nthImage > 0:
                    self.currentTrainImageSet[rid][idx,:,:,4:5] = edge
                    self.currentTrainImageSet[rid][idx,:,:,5:] = epicflow.affine(flow, angle, scale, 3)
                self.currentTrainLabelSet[rid][idx,:,:,0] = np.fliplr(np.flipud(1-labelR))
                self.currentTrainLabelSet[rid][idx,:,:,1] = np.fliplr(np.flipud(labelR))
                idx += 1

            nthImage += 1
        self.currentTrainImageSetSize = idx


    # Rotate the image and zoom. Angle is in degree
    def rotateImage(self, angle, image):
        retImage = ndimage.interpolation.rotate(image, angle)
        widthO = image.shape[0]
        heightO = image.shape[1]
        widthR = retImage.shape[0]
        heightR = retImage.shape[1]
        theta = angle / 180.0 * np.pi
        scale = float(widthR) / widthO

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
        return retImage, scale

    def data_distort(self, label):
        distort = Data_Distor.Data_Distor(label)
        mask = distort.genMasks().astype(np.uint8)
        return mask

    # for davis video
    def findVideoSize(self, names, id):
        if id == len(names):
            print("No images.")
            return None

        videoName = names[id].split('/')[3]
        id += 1
        size = 1
        while id < len(names):
            if names[id].split('/')[3] == videoName:
                size += 1
                id += 1
            else:
                id += 1
                break

        # return 3
        return size



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
    reader = DavisReader(mode="video")
    import matplotlib.pyplot as plt

    for _ in range(1):
        # images, labels = reader.next_test()
        images, labels = reader.next_batch()
        print(images.shape)
        print(labels.shape)


        for i in range(10):
            image = images[i,:,:,:3]
            label = labels[i,:,:,1]
            print(image.shape)
            # print (image.shape)
            misc.imsave('label.png', label*255)
            print('aaa')
            misc.imsave('image.png', image)
            print('aaaaaa')
            showImageLabel(image, label)

