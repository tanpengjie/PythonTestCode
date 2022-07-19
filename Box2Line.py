"""
Turn box into centerline
"""

import os
import numpy as np
import cv2


def adjust_order_of_points1(polygon):
    """
    Adjust the order of points. During labeling process, the order of second point and the fourth point may be out of order.
Args:
    polygon (list): The points of polygon.

Returns:
    points (list): Ordered points.
    """
    if isinstance(list,type(polygon)):
        raise Exception(" Please input right type")

    [x1, y1, x2, y2, x3, y3, x4, y4] = polygon
    if x4 > x2:
        temp_x = x2
        temp_y = y2
        x2 = x4
        y2 = y4
        x4 =temp_x
        y4 =temp_y

    return [x1, y1, x2, y2, x3, y3, x4, y4]

def Box2Line(FolderPath):
    suffix = ['txt']
    for i in os.listdir(FolderPath):
        if i.split('.')[-1] not in suffix:
            continue
        print(i)
        TxtPath = FolderPath + "\\" + i
        TotalPolygon = []
        ImagePath = FolderPath + "\\" + i.replace('.txt','.jpg')
        Image = cv2.imread(ImagePath,1)

        with open(TxtPath,'r') as R:
            lines = [line.strip() for line in R.readlines()]

            for line in lines:
                if len(line) < 2:
                    continue
                if len(line.split(',')) != 9:
                    Exception("   The type of the txt data is error")

                [c, x1, y1, x2, y2, x3, y3, x4, y4] = [coordinate for coordinate in line.split(' ')]
                TotalPolygon.append([float(x1), float(y1), float(x2), float(y2), float(x3), float(y3), float(x4), float(y4)])

        [Height,Width,_] = Image.shape
        dstHeight = 400
        dstWidth = 400
        Image = cv2.resize(Image,(dstHeight,dstWidth))

        cv2.imwrite(ImagePath,Image)
        ratio = dstWidth/Width
        ImagePath = FolderPath + "\\"  + i.replace('.txt', '_line.jpg')

        with open(TxtPath,'w') as W:
            for i in TotalPolygon:

                if len(i) !=8:
                    continue

                [x1, y1, x2, y2, x3, y3, x4, y4] = adjust_order_of_points1(i)
                centerx1 = int((x1 + x4) * ratio / 2)
                centery1 = int((y1 + y4) * ratio / 2)
                centerx2 = int((x2 + x3) * ratio / 2)
                centery2 = int((y2 + y3) * ratio / 2)

                if centerx1 < 1:
                    centerx1 = 1
                if centerx1 > Width - 1:
                    centerx1 = Width - 1

                if centerx2 < 1:
                    centerx2 = 1
                if centerx2 > Width - 1:
                    centerx2 = Width - 1

                if centery1 < 1:
                    centery1 = 1
                if centery1 > Height - 1:
                    centery1 = Height - 1

                if centery2 < 1:
                    centery2 = 1
                if centery2 > Height - 1:
                    centery2 = Height - 1

                cv2.line(Image,(centerx1,centery1),(centerx2,centery2),(0,0,255),3)
                # print(i)
                # cv2.namedWindow('output_image', cv2.WINDOW_AUTOSIZE)
                # cv2.imshow('output_image', Image)

                W.write(str(centerx1) + ',' + str(centery1) + ',' + str(centerx2) + ',' + str(centery2) + ',' + str(dstWidth) + ',' + str(dstHeight) + '\n')

        cv2.imwrite(ImagePath, Image)

if __name__ == '__main__':
    temp = np.zeros((400,400,3),dtype=np.float32)

    Box2Line("E:\\ImageData\\WarpSeperationTrainData\\gtLine")



