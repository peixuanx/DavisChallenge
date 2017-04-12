from random import uniform
import numpy as np
from scipy.ndimage.interpolation import affine_transform
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.ndimage.morphology import binary_dilation

class Data_Distor:
    def __init__(self, image):
        self.image = image
        self.affineMasks = [image, image]
        self.masks = np.zeros(image.shape+(3,))
        # Original mask parameters
        #print(image.shape)
        y,x = image.nonzero()
        self.bbox = [min(x),max(x),min(y),max(y)]
        self.height = self.bbox[3]-self.bbox[2]
        self.width = self.bbox[1]-self.bbox[0]

    def affine(self):
        for i in range(2):
            s = uniform(0.95,1.05)
            ht = uniform(-0.1,0.1)*self.height
            wt = uniform(-0.1,0.1)*self.width
            matrix = [[s,0],
                      [0,s]]
            self.affineMasks[i] = affine_transform(self.affineMasks[i], 
                                                   matrix=matrix, offset=[ht,wt])

    def non_rigid(self):

        def makeT(cp):
            # cp: [K x 2] control points
            # T: [(K+3) x (K+3)]
            K = cp.shape[0]
            T = np.zeros((K+3, K+3))
            T[:K, 0] = 1
            T[:K, 1:3] = cp
            T[K, 3:] = 1
            T[K+1:, 3:] = cp.T
            R = squareform(pdist(cp, metric='euclidean'))
            R = R * R
            R[R == 0] = 1 # a trick to make R ln(R) 0
            R = R * np.log(R)
            np.fill_diagonal(R, 0)
            T[:K, 3:] = R
            return T

        def liftPts(p, cp):
            # p: [N x 2], input points
            # cp: [K x 2], control points
            # pLift: [N x (3+K)], lifted input points
            N, K = p.shape[0], cp.shape[0]
            pLift = np.zeros((N, K+3))
            pLift[:,0] = 1
            pLift[:,1:3] = p
            R = cdist(p, cp, 'euclidean')
            R = R * R
            R[R == 0] = 1
            R = R * np.log(R)
            pLift[:,3:] = R
            return pLift
        
        def genControlPts():
            ratio = 0.1
            # source control points
            #b = np.random.randint(0,4,size=5)
            #xs = np.zeros(b.shape)
            #xs[b==0] = self.bbox[0] #+ np.random.uniform(-0.1*self.width, 0.1*self.width, size=xs[b==0].size)
            #xs[b==1] = self.bbox[1] #+ np.random.uniform(-0.1*self.width, 0.1*self.width, size=xs[b==1].size)
            #xs[b>1]  = np.random.uniform(self.bbox[0],self.bbox[1],size=xs[b>1].size)#self.bbox[0] + np.random.uniform(-0.1*self.width, 1.1*self.width, size=xs[b>1].size)
            #ys = np.zeros(b.shape)
            #ys[b==2] = self.bbox[2] #+ np.random.uniform(-0.1*self.height, 0.1*self.height, size=ys[b==2].size)
            #ys[b==3] = self.bbox[3] #+ np.random.uniform(-0.1*self.height, 0.1*self.height, size=ys[b==3].size)
            #ys[b<2]  = np.random.uniform(self.bbox[2],self.bbox[3],size=xs[b<2].size)#self.bbox[2] + np.random.uniform(-0.1*self.height, 1.1*self.height, size=ys[b<2].size)
            xs = np.array([self.bbox[0],self.bbox[1],self.bbox[0],self.bbox[1],(self.bbox[0]+self.bbox[1])/2])
            ys = np.array([self.bbox[2],self.bbox[2],self.bbox[3],self.bbox[3],(self.bbox[2]+self.bbox[3])/2])

            # target control points
            xt = np.random.uniform(-ratio*self.width, ratio*self.width, size=5)
            xt += xs
            yt = np.random.uniform(-ratio*self.height, ratio*self.height, size=5)
            yt += ys
            return [xs, ys, xt, yt]
            
        for i in range(2):

            succeed = False
            while not succeed:
                try:
                    xs, ys, xt ,yt = genControlPts()
                    cps = np.vstack([xs, ys]).T

                    # construct T
                    T = makeT(cps)
                    
                    # solve cx, cy (coefficients for x and y)
                    xtAug = np.concatenate([xt, np.zeros(3)])
                    ytAug = np.concatenate([yt, np.zeros(3)])
                    cx = np.linalg.solve(T, xtAug) # [K+3]
                    cy = np.linalg.solve(T, ytAug)
                    succeed = True
                except:
                    succeed = False

            # dense grid
            ygs, xgs = np.nonzero(self.affineMasks[i])
            gps = np.vstack([xgs, ygs]).T

            # transform
            pgLift = liftPts(gps, cps) # [N x (K+3)]
            xgt = np.rint(np.dot(pgLift, cx.T)).astype(int)
            ygt = np.rint(np.dot(pgLift, cy.T)).astype(int)

            # crop
            idx = np.all([xgt>=0, xgt<self.image.shape[1]],axis=0)
            xgt, ygt = xgt[idx], ygt[idx]
            idx = np.all([ygt>=0, ygt<self.image.shape[0]],axis=0)
            xgt, ygt = xgt[idx], ygt[idx]
            self.masks[ygt,xgt,i] = 1

    def genMasks(self):
        self.affine()
        self.non_rigid()

        for i in range(2):
            self.masks[:,:,i] = binary_dilation(self.masks[:,:,i],iterations=5)
        return self.masks
