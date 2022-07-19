import cv2
import os



def RotateImage(ImagePath,CropImageFolderPath):
    ImageName = ImagePath.split('\\')[-1].split('.')[0]
    Image = cv2.imread(ImagePath,1)
    #ImageWidth,ImageHeight,ImageChannel = Image.shape
    ImageWidth, ImageHeight, _ = Image.shape

    matRotate = cv2.getRotationMatrix2D((ImageHeight * 0.5, ImageWidth * 0.5), -90, 1)
    dst = cv2.warpAffine(Image, matRotate, (ImageWidth, ImageWidth))

    CropImagePath = CropImageFolderPath + "\\" + ImageName + "_90"  + ".jpg"
    print(CropImagePath)
    cv2.imwrite(CropImagePath, dst)


def BatchCropFixedSizeImage(SrcDir,DstDir):
    suffix = ['jpg', 'png', 'jpeg', 'bmp']
    for i in os.listdir(SrcDir):
        if i.split('.')[-1] not in suffix:
            continue
        print(i)
        ImagePath = SrcDir + "\\" + i
        RotateImage(ImagePath,DstDir)




if __name__ == '__main__':

    SrcDir = "E:\\ImageData\\DenimClassificationDataset\\CameraImage\\train\BrokenTwill"
    DstDir = "E:\\ImageData\\DenimClassificationDataset\\CameraImage\\train\BrokenTwill"

    BatchCropFixedSizeImage(SrcDir,DstDir)