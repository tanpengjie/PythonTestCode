import cv2
import numpy as np
import random
import json
import os



def CropFixedSizeImage(ImagePath,CropImageFolderPath,CropSize):
    ImageName = ImagePath.split('\\')[-1].split('.')[0]
    Image = cv2.imread(ImagePath,1)
    ImageHeight, ImageWidth, _ = Image.shape
    #resize

    Image = cv2.resize(Image,(int(0.6*ImageWidth),int(0.6*ImageHeight)))
    ImageHeight, ImageWidth, _ = Image.shape
    #crop
    Image = Image[int(ImageHeight/4):int(ImageHeight*3/4),int(ImageWidth/4):int(ImageWidth*3/4)]
    ImageHeight, ImageWidth, _ = Image.shape

    CropImageSize =CropSize
    # X和Y方向循环次数
    XCircleNumber = int(ImageHeight / CropImageSize) + 1
    YCircleNumber = int(ImageWidth / CropImageSize) + 1

    if XCircleNumber <= 1:
        raise Exception("Width of input image is too low")

    if YCircleNumber <= 1:
        raise Exception("Height of input image is too low")

    for i in range(0,XCircleNumber):
        for j in range(0,YCircleNumber):
            if i == XCircleNumber - 1 and j == YCircleNumber - 1 :
                CropImage = Image[ImageHeight - CropImageSize:ImageHeight, ImageWidth-CropImageSize:ImageWidth]
            elif i == XCircleNumber - 1 :
                CropImage = Image[ImageHeight-CropImageSize:ImageHeight,j*CropImageSize:j*CropImageSize + CropImageSize]
            elif j == YCircleNumber - 1 :
                CropImage = Image[i*CropImageSize:i*CropImageSize + CropImageSize,ImageWidth-CropImageSize:ImageWidth]
            else:
                CropImage = Image[i * CropImageSize:i * CropImageSize + CropImageSize, j*CropImageSize:j*CropImageSize + CropImageSize]

            # CropImageHeight,CropImageWidth,_ = CropImage.shape
            # matRotate = cv2.getRotationMatrix2D((CropImageHeight * 0.5, CropImageWidth * 0.5), -90, 1)
            # dst = cv2.warpAffine(CropImage, matRotate, (CropImageHeight, CropImageWidth))

            CropImagePath = CropImageFolderPath + "\\" +ImageName + "_" + str(i) + "_" + str(j) + ".jpg"
            print(CropImagePath)
            #cv2.imwrite(CropImagePath,dst)
            cv2.imwrite(CropImagePath, CropImage)


def BatchCropFixedSizeImage(SrcDir,DstDir,CropSize):
    suffix = ['jpg', 'png', 'jpeg', 'bmp']
    for i in os.listdir(SrcDir):
        if i.split('.')[-1] not in suffix:
            continue
        print(i)
        ImagePath = SrcDir + "\\" + i
        CropFixedSizeImage(ImagePath,DstDir,CropSize)



if __name__ == '__main__':

    SrcDir = "E:\\ImageData\\DenimClassificationDataset\\211112CameraImage\\RightHandTwill"
    DstDir = "E:\\ImageData\\DenimClassificationDataset\\211112CameraImage\\train_0.6\\RightHandTwill"
    CropSize = 224
    BatchCropFixedSizeImage(SrcDir,DstDir,CropSize)



