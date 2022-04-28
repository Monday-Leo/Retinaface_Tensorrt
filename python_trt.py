from ctypes import *
import cv2
import numpy as np
import numpy.ctypeslib as npct
import time as t

class Detector():
    def __init__(self,model_path,dll_path):
        self.retinanet = CDLL(dll_path)
        self.retinanet.Detect.argtypes = [c_void_p,c_int,c_int,POINTER(c_ubyte),c_float,npct.ndpointer(dtype = np.float32, ndim = 2, shape = (1000, 15), flags="C_CONTIGUOUS")]
        self.retinanet.Init.restype = c_void_p
        self.retinanet.Init.argtypes = [c_void_p]
        self.retinanet.cuda_free.argtypes = [c_void_p]
        self.c_point = self.retinanet.Init(model_path)

    def predict(self,img,threshold):
        rows, cols = img.shape[0], img.shape[1]
        res_arr = np.zeros((1000,15),dtype=np.float32)
        self.retinanet.Detect(self.c_point,c_int(rows), c_int(cols), img.ctypes.data_as(POINTER(c_ubyte)),c_float(threshold),res_arr)
        self.bbox_array = res_arr[~(res_arr==0).all(1)]
        return self.bbox_array

    def free(self):
        self.retinanet.cuda_free(self.c_point)

def visualize(img,bbox_array):
    for temp in bbox_array:
        #bbox = [temp[0],temp[1],temp[2],temp[3]]  #xywh
        cv2.rectangle(img,(int(temp[0]),int(temp[1])),(int(temp[0])+int(temp[2]),int(temp[1])+int(temp[3])), (105, 237, 249), 2)
        for k in range(5,15,2):
                cv2.circle(img, (temp[k],temp[k+1]), 1, (255 * (k > 7), 255 * (k > 5 and k < 13), 255 * (k < 11)), 4)
    return img

det = Detector(model_path=b"./retina_mnet.engine",dll_path="./retina_mnet.dll")  # b'' is needed
img = cv2.imread("test.jpg")


for i in range(10):
    t1 = t.time()
    result = det.predict(img,threshold=0.75)
    t2 = t.time()
    print((t2-t1)*1000,"ms")
img = visualize(img,result)
cv2.imshow("img",img)
cv2.waitKey(0)
det.free()
cv2.destroyAllWindows()