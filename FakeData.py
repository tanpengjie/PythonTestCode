import cv2
import numpy as np
import random
import json
import os

#判断点是否在多边形内
def judgePointIsInsidePolygon(Point,Polygon):
    """
    Determine if a point is inside a polygon
    :param Point:
    :param Polygon:
    :return: True or False
    """
    if isinstance(Point,tuple):
        raise Exception("Please input right variable type ...")

    if isinstance(Polygon,tuple):
        raise Exception("Please input right variable type ...")

    PointX = Point[0]['x']
    PointY = Point[0]['y']

    CurrentIndex = 0
    PointCount = len(Polygon)
    NextIndex = PointCount - 1
    Flag = False

    while CurrentIndex < PointCount:
        CurrentPointX = Polygon[CurrentIndex]['x']
        CurrentPointY = Polygon[CurrentIndex]['y']
        NextPointX = Polygon[NextIndex]['x']
        NextPointY = Polygon[NextIndex]['y']

        #判断点是否与多边形的顶点重合
        if (PointX == CurrentPointX and PointY == CurrentPointY) or (PointX == NextPointX and PointY == NextPointY):
            return True
        #判断点是否在多边形的两侧
        if (PointY < CurrentPointY and PointY >= NextPointY) or (PointY < NextPointY and PointY >= CurrentPointY):
            # 线段上与射线 Y 坐标相同的点的 X 坐标
            x = CurrentPointX + (PointY - CurrentPointY) * (NextPointX - CurrentPointX) / (NextPointY - CurrentPointY)
            # 点在多边形的边上
            if x == PointX:
                return True
            # 射线穿过多边形的边界
            if x > PointX:
                Flag = not Flag

        NextIndex = CurrentIndex
        CurrentIndex = CurrentIndex + 1

    return True if Flag else False

def ReadXMLData(JsonPath):
    """
    # Read xml rectangle data and polygon data
    :param JsonPath:
    :return:
    """
    AnnotationData = []
    with open(JsonPath,'r',encoding='utf8') as JsonFile:
        JsonData = json.load(JsonFile)
        RectangleDatas = []
        PolygonDatas = []
        for i in range(len(JsonData['shapes'])):
            if JsonData['shapes'][i]['shape_type'] == 'rectangle':
                RectangleData = JsonData['shapes'][i]['points']
                RectangleDatas.append({ 'RectangleData' : list(RectangleData) })

            if JsonData['shapes'][i]['shape_type'] == 'polygon':
                PolygonData = JsonData['shapes'][i]['points']
                PolygonDatas.append({ 'PolygonData' : list(PolygonData) })

    AnnotationData.append({'RectangleDatas':RectangleDatas})

    #将多边形的坐标换算到相对于矩形框的
    for i in range(len(PolygonDatas)):
        for j in range(len(PolygonDatas[i]['PolygonData'])):
            PolygonDatas[i]['PolygonData'][j][0] = PolygonDatas[i]['PolygonData'][j][0] - RectangleDatas[i]['RectangleData'][0][0]
            PolygonDatas[i]['PolygonData'][j][1] = PolygonDatas[i]['PolygonData'][j][1] - RectangleDatas[i]['RectangleData'][0][1]

    AnnotationData.append({'PolygonDatas': PolygonDatas})
    return AnnotationData

def XMLToTXT(JsonPath):
    """
    #将Json数据转换成txt数据,数据的格式为 x,y,w,h
    :param JsonPath:
    :return:
    """
    AnnotationData = []
    RectangleDatas = []
    TxtPath = JsonPath.replace('.json','.txt')

    with open(JsonPath, 'r', encoding='utf8') as JsonFile:
        JsonData = json.load(JsonFile)
        for i in range(len(JsonData['shapes'])):
            if JsonData['shapes'][i]['shape_type'] == 'rectangle':
                RectangleData = JsonData['shapes'][i]['points']
                RectangleLabel = JsonData['shapes'][i]['label']
                RectangleDatas.append({'RectangleData': list(RectangleData), 'RectangleLabel': int(RectangleLabel)})

    with open(TxtPath, 'w', encoding='utf8') as W:
        for i in range(len(RectangleDatas)):
            if len(RectangleData) == 2:
                [LeftTopPoint, RightBottomPoint] = RectangleDatas[i]['RectangleData'][0], RectangleDatas[i]['RectangleData'][1]
                if len(LeftTopPoint) != 2 or len(RightBottomPoint) != 2:
                    continue
                [c,x, y, w, h] = [RectangleDatas[i]['RectangleLabel'], LeftTopPoint[0], LeftTopPoint[1], RightBottomPoint[0]-LeftTopPoint[0], RightBottomPoint[1]-LeftTopPoint[1]]
                W.write( str(c) + " " +  str(x) + " " + str(y) + " " + str(w) + " " + str(h) + '\n')
        print(TxtPath)

def XMLToTXT_4Points(JsonPath):
    """
    #将Json数据转换成txt数据,数据的格式为：四个点坐标
    :param JsonPath:
    :return:
    """
    AnnotationData = []
    RectangleDatas = []
    TxtPath = JsonPath.replace('.json','.txt')

    with open(JsonPath, 'r', encoding='utf8') as JsonFile:
        JsonData = json.load(JsonFile)
        for i in range(len(JsonData['shapes'])):
            if JsonData['shapes'][i]['shape_type'] == 'polygon':
                RectangleData = JsonData['shapes'][i]['points']
                RectangleLabel = JsonData['shapes'][i]['label']
                RectangleDatas.append({'RectangleData': list(RectangleData), 'RectangleLabel': int(RectangleLabel)})

    with open(TxtPath, 'w') as W:
        for i in range(len(RectangleDatas)):
            if len(RectangleData) == 4:
                [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] =[RectangleDatas[i]['RectangleData'][0], RectangleDatas[i]['RectangleData'][1],
                                                    RectangleDatas[i]['RectangleData'][2], RectangleDatas[i]['RectangleData'][3]]
                c = RectangleDatas[i]['RectangleLabel']
                W.write( str(c) + " " +  str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + str(x3) + " " + str(y3) +
                         " " + str(x4) + " " + str(y4) + '\n')
        print(TxtPath)

def BatchXMLToTxt(FolderPath):
    suffix = ['json']
    for i in os.listdir(FolderPath):
        if i.split('.')[-1] not in suffix:
            continue
        print(i)
        JsonPath = FolderPath + "\\" + i

        if not os.path.exists(JsonPath):
            continue
        #XMLToTXT(JsonPath)
        XMLToTXT_4Points(JsonPath)


def ReadJsonTargetRegionRectangle(JsonPath):
    """
    读取Json文件中的目标区域的矩形框信息
    :param JsonPath:
    :return:
    """
    if not os.path.exists(JsonPath):
        raise Exception("   Please input right Json path ...")

    RectangleDatas = []
    with open(JsonPath,'r',encoding='utf8') as JsonFile:
        JsonData = json.load(JsonFile)
        for i in range(len(JsonData['shapes'])):
            if JsonData['shapes'][i]['shape_type'] == 'rectangle' and JsonData['shapes'][i]['label'] ==  'Region':
                RectangleData = JsonData['shapes'][i]['points']
                RectangleDatas.append({ 'RectangleData' : list(RectangleData) })

    return RectangleDatas

def GetPolygonPoints(Polygon):
    """
    get the vertices of the polygon
    :param Polygon:
    :return: Points
    """
    if isinstance(Polygon,dict):
        raise Exception(" Please input right variable type")

    Points =[]
    for i in range(0,len(Polygon)):
        point = Polygon[i]
        Points.append({'x':point[0],'y':point[1]})

    return Points

def TransformImage(Image,TransformType):
    """
    Transform Image,such as Zooming, Shrinking, Rotate, Affine transformation, Perspective transformation
    :param Image:
    :param TransformType:
    :return:
    """
    if TransformType == "Zooming":
        ZoomingScale = random.uniform(0.7,1.0)
        NewImage = cv2.resize(Image,None,fx=ZoomingScale,fy=ZoomingScale,interpolation=cv2.INTER_CUBIC)
        return NewImage

    elif TransformType == "Shrinking":
        ShrinkingScale = random.uniform(0.7,1.0)
        NewImage = cv2.resize(Image,None,fx=ShrinkingScale,fy=ShrinkingScale,interpolation=cv2.INTER_AREA)
        return NewImage

    elif TransformType == "Rotate":
        [ImageHeight,ImageWidth,_] = Image.shape
        RotateAngle = random.randint(45,90)
        ZoomingCoefficient = 1

        ZoomingImageLength = int(np.sqrt(ImageWidth*ImageWidth + ImageHeight*ImageHeight))
        ZoomingImage = np.zeros((ZoomingImageLength,ZoomingImageLength,3),dtype=Image.dtype)

        StartY = int((ZoomingImageLength - ImageHeight)/2.0)
        StartX = int((ZoomingImageLength - ImageWidth)/2.0)
        ZoomingImage[StartY : StartY + ImageHeight,StartX :StartX + ImageWidth] = Image
        M = cv2.getRotationMatrix2D(((ImageHeight - 1) / 2.0,(ImageWidth - 1) / 2.0),RotateAngle,ZoomingCoefficient)
        NewImage = cv2.warpAffine(Image,M,(ImageHeight,ImageWidth))
        return NewImage

    elif TransformType == "AffineTransform":
        [ImageHeight, ImageWidth, _] = Image.shape
        Point1 = np.float32([[0,0],[ImageWidth - 1,0],[0,ImageHeight - 1]])
        Point2 = np.float32([[0,ImageHeight*random.uniform(0.1,1.0)],[ImageWidth*random.uniform(0.1,1.0),ImageHeight*random.uniform(0.1,1.0)],
                             [ImageWidth * random.uniform(0.1, 1.0), ImageHeight * random.uniform(0.1, 1.0)]])
        M = cv2.getAffineTransform(Point1,Point2)
        NewImage = cv2.warpAffine(Image,M,(ImageWidth,ImageHeight))
        return NewImage

    elif TransformType == "PerspectiveTransform":
        [ImageHeight, ImageWidth, _] = Image.shape
        PerspectiveCoefficient = random.uniform(0.8,0.9)
        Point1 = np.float32([[0, 0], [ImageWidth - 1, 0], [0, ImageWidth - 1], [ImageWidth - 1, ImageHeight - 1]])
        Point2 = np.float32([[0, ImageHeight*random.uniform(0.3,0.35)], [ImageWidth*random.uniform(0.8,9.0),ImageHeight*random.uniform(0.3,0.35)],
                         [ImageWidth*random.uniform(0.1,0.3),ImageHeight*random.uniform(0.7,0.8)],
                         [ImageWidth*PerspectiveCoefficient,ImageHeight*PerspectiveCoefficient]])
        M = cv2.getPerspectiveTransform(Point1,Point2)
        NewImage = cv2.warpPerspective(Image,M,(ImageWidth,ImageHeight))
        return NewImage

    else:
        raise Exception("   Please input right Transform parameter")

def BatchFakeImageData(ImageFolderPath,DefectImageFolderPath,SaveImageFolder,Batch = 1):
    """
    批量造图像数据
    :param ImageFolderPath:
    :param JsonFolderPath:
    :param DefectImageFolderPath:
    :return:
    """
    if not os.path.exists(ImageFolderPath):
        raise Exception("   Please input right Image Folder path ...")

    if not os.path.exists(DefectImageFolderPath):
        raise Exception("   Please input right Defect Image Folder path ...")

    ImagePathLists = []
    for i in os.listdir(ImageFolderPath):
        if i.split('.')[-1] != 'jpg':
            continue
        ImagePath = os.path.join(ImageFolderPath,i)
        ImagePathLists.append(ImagePath)

    DefectImages = []
    for i in os.listdir(DefectImageFolderPath):
        DefectPath = os.path.join(DefectImageFolderPath,i)
        Image = cv2.imread(DefectPath)
        DefectImages.append(Image)

    for batch in range(0,Batch):
        count = 0
        for i in ImagePathLists:
            #判断图像的路径是否存在
            if not os.path.exists(i):
                continue
            print(i)
            Image = cv2.imread(i)
            JsonPath = i.split('.')[0] + '.json'
            #判断Json的路径是否存在/
            if not os.path.exists(JsonPath):
                continue
            #读取Json中目标区域的矩形框
            RectangleDatas = ReadJsonTargetRegionRectangle(JsonPath)
            RectangleData = RectangleDatas[random.randint(0,len(RectangleDatas) - 1)]
            RectangleHeight = RectangleData['RectangleData'][1][1] - RectangleData['RectangleData'][0][1]
            RectangleWidth = RectangleData['RectangleData'][1][0] - RectangleData['RectangleData'][0][0]
            #随机读取一个缺陷图片
            while 1:
                RandomNumber = random.randint(0,len(DefectImages) - 1)
                DefectImage = DefectImages[RandomNumber]
                TransformType = ["Zooming", "Shrinking", "Rotate"]
                RandomType = random.randint(0, len(TransformType) - 1)
                DefectImage = TransformImage(DefectImage, TransformType[RandomType])
                [DefectImageHeight,DefectImageWidth,_] = DefectImage.shape
                if RectangleHeight - DefectImageHeight > 5 and RectangleWidth - DefectImageWidth > 5:
                    break
            #将缺陷贴到指定区域
            [LeftTopPointY, LeftTopPointX] = RectangleData['RectangleData'][0][1],RectangleData['RectangleData'][0][0]
            BinaryImage = np.zeros((DefectImageHeight, DefectImageWidth), dtype=DefectImage.dtype)
            #将腐蚀缺陷的边缘
            for y in range(0, DefectImageHeight):
                for x in range(0, DefectImageWidth):
                    if DefectImage[y, x][0] < 15 and DefectImage[y, x][1] < 15 and DefectImage[y, x][0] < 15:
                        continue
                    BinaryImage[y, x] = 255

            kernel = np.ones((5, 5), np.uint8)
            ErosionImage = cv2.erode(BinaryImage, kernel)
            #得到缺陷区域的宽和高
            contours, hierarchy = cv2.findContours(ErosionImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
                #cv2.rectangle(ErosionImage,(x,y),(x+w,y+h),255,3)

            # 将缺陷随机贴到指定区域
            RandomHeight = random.randint(0,int(RectangleHeight) - int(DefectImageHeight) - 1)
            RandomWidth = random.randint(0,int(RectangleWidth) - int(DefectImageWidth) - 1)
            [LeftTopPointY, LeftTopPointX] = [int(LeftTopPointY) + RandomHeight, int(LeftTopPointX) + RandomWidth]
            for y in range(0, DefectImageHeight):
                for x in range(0, DefectImageWidth):
                    if ErosionImage[y, x] < 1:
                        continue
                    Image[LeftTopPointY + y, LeftTopPointX + x] = DefectImage[y, x]
            #保存图片
            ImageName = i.split('\\')[-1].split('.')[0]
            SavePath = SaveImageFolder  + "\\"  + str(batch) + "_" +  ImageName + "_"  + str(count) + ".jpg"
            TxtPath = SaveImageFolder  + "\\"  + str(batch) + "_" +  ImageName + "_"  + str(count) + ".txt"
            Class = 1
            [ImageHeight,ImageWidth,_] = Image.shape
            X1 = float(LeftTopPointX - 20) /  ImageWidth
            Y1 = float(LeftTopPointY - 20) / ImageHeight
            W1 = float(w + 40) / ImageWidth
            H1 = float(h + 40) / ImageHeight
            with open(TxtPath,'w',encoding='utf8') as W:
                W.write(str(Class) +' ' + str(X1) + ' ' + str(Y1) + ' ' + str(W1)  + ' ' +  str(H1) + '\n')

            cv2.imwrite(SavePath,Image)
            count = count + 1


def CropImage(JsonPath,ImagePath):
    """

    :param JsonPath:
    :return:
    """
    AnnotationData = ReadXMLData(JsonPath)
    RectangleDatas = AnnotationData[0]['RectangleDatas']
    RectangleImages = []
    image = cv2.imread(ImagePath)
    for i in range(0,len(RectangleDatas)):
        Rectangle = RectangleDatas[i]['RectangleData']
        CropImage = image[int(Rectangle[0][1]):int(Rectangle[1][1]),int(Rectangle[0][0]) :int(Rectangle[1][0])]
        #cv2.imwrite("E:\\ImageData\\ShoesImage\\no_18\\temp.jpg",CropImage)
        RectangleImages.append(CropImage)

    PolygonDatas = AnnotationData[1]['PolygonDatas']
    CropImages = []
    count = 0
    for i in range(0,len(PolygonDatas)):
        Polygon = PolygonDatas[i]['PolygonData']
        PolygonVertices = GetPolygonPoints(Polygon)
        RectangleImage = RectangleImages[i]
        [ImageHeight,ImageWidth,_] = RectangleImage.shape
        for y in range(0,ImageHeight):
            for x in range(0,ImageWidth):
                point = []
                point.append({'x':x, 'y':y})
                IsInsidePolygon = judgePointIsInsidePolygon(point,PolygonVertices)
                if not IsInsidePolygon:
                    RectangleImage[y,x] = [0,0,0]
        SavePath = ImagePath.split('.')[0] + '_' + str(count) + '.jpg'
        #cv2.imwrite("E:\\ImageData\\ShoesImage\\no_18\\temp1.jpg",RectangleImage)
        cv2.imwrite(SavePath, RectangleImage)
        CropImages.append(RectangleImage)



def PasteDefectImageToDesignatedArea(ImagePath,DefectImagePath):
    """

    :param ImagePath:
    :param DefectImagePath:
    :return:
    """
    Image = cv2.imread(ImagePath)
    DefectImage = cv2.imread(DefectImagePath)
    #TransformType = ["Zooming","Shrinking","Rotate","AffineTransform","PerspectiveTransform"]
    TransformType = ["Zooming", "Shrinking"]
    RandomNumber = random.randint(0,len(TransformType) - 1)
    if RandomNumber >= 0:
        print('The transformation that is going to take is %s'%TransformType[RandomNumber])
        DefectImage = TransformImage(DefectImage,TransformType[RandomNumber])
        cv2.imwrite("E:\\ImageData\\ShoesImage\\no_18\\transformDefect.jpg",DefectImage)

    [ImageHeight,ImageWidth,_] = Image.shape
    [DefectImageHeight,DefectImageWidth,_] = DefectImage.shape

    #DesignatedArea = [[841,1381]]
    DesignatedArea = [[541,1671]]
    count = 0
    for i in range(0,len(DesignatedArea)):
        [LeftTopPointY,LeftTopPointX] = DesignatedArea[i]

        if LeftTopPointX > ImageWidth - DefectImageWidth - 10 or LeftTopPointX < 0 or\
                LeftTopPointY > ImageHeight -DefectImageHeight - 10 or LeftTopPointY < 0:
            continue
        BinaryImage = np.zeros((DefectImageHeight,DefectImageWidth),dtype=DefectImage.dtype)
        for y in range(0,DefectImageHeight):
            for x in range(0,DefectImageWidth):
                if DefectImage[y,x][0] < 25 and DefectImage[y,x][1] < 25 and DefectImage[y,x][0] < 25:
                    continue
                BinaryImage[y,x] = 255

        kernel = np.ones((5, 5), np.uint8)
        ErosionImage = cv2.erode(BinaryImage, kernel)
        # 得到缺陷区域的宽和高
        contours, hierarchy = cv2.findContours(ErosionImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
            # cv2.rectangle(ErosionImage,(x,y),(x+w,y+h),255,3)

        Class = 1
        [ImageHeight, ImageWidth, _] = Image.shape
        X1 = float(LeftTopPointX - 20) / ImageWidth
        Y1 = float(LeftTopPointY - 20) / ImageHeight
        W1 = float(w + 40) / ImageWidth
        H1 = float(h + 40) / ImageHeight
        TxtPath = "E:\\ImageData\\ShoesImage\\WrinkleImage\\" + str(count) + "22.txt"
        with open(TxtPath, 'w', encoding='utf8') as W:
            W.write(str(Class) + ' ' + str(X1) + ' ' + str(Y1) + ' ' + str(W1) + ' ' + str(H1) + '\n')

        for y in range(0,DefectImageHeight):
            for x in range(0,DefectImageWidth):
                if ErosionImage[y,x] < 1:
                    continue
                Image[LeftTopPointY + y, LeftTopPointX + x] = DefectImage[y, x]

        cv2.imwrite("E:\\ImageData\\ShoesImage\\WrinkleImage\\" + str(count) + "22.jpg", Image)
        count = count + 1

def BatchCropImage(FolderPath):

    suffix = ['jpg', 'png', 'jpeg', 'bmp']
    for i in os.listdir(FolderPath):
        if i.split('.')[-1] not in suffix:
            continue
        print(i)
        ImagePath = FolderPath + "\\" + i
        JsonPath = FolderPath + "\\" + i.split('.')[0] + '.json'

        if not os.path.exists(JsonPath):
            continue
        CropImage(JsonPath,ImagePath)

def CropFixedSizeImage(ImagePath,CropImageFolderPath,CropSize):
    ImageName = ImagePath.split('\\')[-1].split('.')[0]
    Image = cv2.imread(ImagePath,1)
    #ImageWidth,ImageHeight,ImageChannel = Image.shape
    ImageWidth, ImageHeight, _ = Image.shape

    CropImageSize =CropSize
    # X和Y方向循环次数
    XCircleNumber = int(ImageWidth / CropImageSize) + 1
    YCircleNumber = int(ImageHeight / CropImageSize) + 1

    if XCircleNumber <= 1:
        raise Exception("Width of input image is too low")

    if YCircleNumber <= 1:
        raise Exception("Height of input image is too low")

    for i in range(0,XCircleNumber):
        for j in range(0,YCircleNumber):
            if i == XCircleNumber - 1 and j == YCircleNumber - 1 :
                CropImage = Image[ImageWidth - CropImageSize:ImageWidth, ImageHeight-CropImageSize:ImageHeight]
            elif i == XCircleNumber - 1 :
                CropImage = Image[ImageWidth-CropImageSize:ImageWidth,j*CropImageSize:j*CropImageSize + CropImageSize]
            elif j == YCircleNumber - 1 :
                CropImage = Image[i*CropImageSize:i*CropImageSize + CropImageSize,ImageHeight-CropImageSize:ImageHeight]
            else:
                CropImage = Image[i * CropImageSize:i * CropImageSize + CropImageSize, j*CropImageSize:j*CropImageSize + CropImageSize]


            CropImagePath = CropImageFolderPath + "\\" +ImageName + "_" + str(i) + "_" + str(j) + ".jpg"
            print(CropImagePath)
            cv2.imwrite(CropImagePath,CropImage)


def BatchCropFixedSizeImage(FolderPath):
    suffix = ['jpg', 'png', 'jpeg', 'bmp']
    for i in os.listdir(FolderPath):
        if i.split('.')[-1] not in suffix:
            continue
        print(i)
        ImagePath = FolderPath + "\\" + i
        CropFixedSizeImage(ImagePath,"E:\\ImageData\\202108111200WCameraImage\\ClassificationImage380",224)

def ColorToGrayImage():
    ImagePath = 'C:\\Users\\19255\\Desktop\\PolyUPhD\\4.jpg'
    image = cv2.imread(ImagePath,0)
    SavePath = ImagePath.split('.')[0] + '1.jpg'
    cv2.imwrite(SavePath,image)
    print("finished")

def Connect_two_txt(ori_folder_path,dst_folder_path):
    """
    Combining the contexts of two txt into one text.
    Args:
        ori_folder_path (str): The path of original txt
        dst_folder_path (str): .The path of destination txt

    Returns:
    """
    ori_txt_name_lists = os.listdir(ori_folder_path)
    dst_txt_name_lists = os.listdir(dst_folder_path)
    if len(ori_txt_name_lists)!=len(dst_txt_name_lists):
        raise Exception("")

    for txt_name in ori_txt_name_lists:
        if(txt_name.split('.')[-1]!='txt'):
            continue
        ori_txt_path = os.path.join(ori_folder_path,txt_name)

        total_polygons = []
        with open(ori_txt_path,'r') as R:
            lines = [line.strip() for line in R.readlines()]

            for line in lines:
                if len(line) < 2:
                    continue
                if len(line.split(',')) != 8:
                    Exception("   The type of the txt data is error")

                [c, x1, y1, x2, y2, x3, y3, x4, y4] = [coordinate for coordinate in line.split(' ')]
                total_polygons.append([c, x1, y1, x2, y2, x3, y3, x4, y4])

        dst_txt_path = os.path.join(dst_folder_path,txt_name)
        with open(dst_txt_path,'r') as R:
            lines = [line.strip() for line in R.readlines()]

            for line in lines:
                if len(line) < 2:
                    continue
                if len(line.split(',')) != 8:
                    Exception("   The type of the txt data is error")

                [c, x1, y1, x2, y2, x3, y3, x4, y4] = [coordinate for coordinate in line.split(' ')]
                total_polygons.append([c, x1, y1, x2, y2, x3, y3, x4, y4])

        with open(ori_txt_path,'w') as W:
            for i in total_polygons:
                if len(i) !=9:
                    continue
                [c, x1, y1, x2, y2, x3, y3, x4, y4] = i
                W.write(c + ' ' + x1 + ' ' + y1 + ' ' + x2 + ' ' + y2 + ' ' + x3 + ' ' + y3 + ' ' + x4 + ' ' + y4 + '\n')

        print(txt_name)

def BatchRotateImage(FolderPath):
    suffix = ['jpg', 'png', 'jpeg', 'bmp']
    RotateAngle = 5
    for i in os.listdir(FolderPath):
        if i.split('.')[-1] not in suffix:
            continue
        print(i)
        ImagePath = FolderPath + "\\" + i

        #angle = np.random.uniform(-self.RotateRange, self.RotateRange)
        Image = cv2.imread(ImagePath)
        rows, cols = Image.shape[0:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), RotateAngle, 1.0)

        Image = cv2.warpAffine(Image, M, (cols, rows), borderValue=[0, 0, 0])

        DstImagePath = FolderPath + "\\Rotate_15_" + i
        cv2.imwrite(DstImagePath,Image)


if __name__ == '__main__':

    #ReadXMLData('E:\\ImageData\\ShoesImage\\no_18\\10013_00_01_19012021.json')
    #CropImage('E:\\ImageData\\ShoesImage\\no_18\\10013_00_01_19012021.json','E:\\ImageData\\ShoesImage\\no_18\\10013_00_01_19012021.jpg')
    #CropImage('E:\\ImageData\\ShoesImage\\no_18\\0.json','E:\\ImageData\\ShoesImage\\no_18\\0.jpg')
    #PasteDefectImageToDesignatedArea('E:\\ImageData\\ShoesImage\\WrinkleImage\\00.jpg', 'E:\\ImageData\\ShoesImage\\DefectImage\\11.jpg')
    BatchCropFixedSizeImage("E:\\ImageData\\202108111200WCameraImage\\tempImage")
    #ColorToGrayImage()
    #BatchRotateImage("E:\\ImageData\\202109071200WCameraImage\\OriginalImage")
    #BatchXMLToTxt('E:\\ImageData\\WarpAndWeftSeparation\\WarpSeparation4')
    #Connect_two_txt('E:\\ImageData\\WarpAndWeftSeparation\\WeftSeparation4','E:\\ImageData\\WarpAndWeftSeparation\\WarpSeparation4')
    #BatchCropImage("E:\\ImageData\\ShoesImage\\SpecialShoes")
    #BatchFakeImageData("E:\\ImageData\\ShoesImage\\CompeteImage","E:\ImageData\ShoesImage\DefectImage","E:\\ImageData\\ShoesImage\\FakeImageData",10)
