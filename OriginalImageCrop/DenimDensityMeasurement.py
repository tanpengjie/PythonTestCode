import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy.fft as nf

def MakeOneDimensionalGaussian1(W, H):
    W = int(W)
    H = int(H)

    k = 2 / H
    x = np.arange(-H / 2, H / 2, 1, float)
    x = abs(1 - abs(x * k))

    x = x[:, np.newaxis]
    y = np.repeat(x, W, 1)
    return y

def MakeOneDimensionalGaussian(W,H):
    W = int(W)
    H = int(H)

    x = np.arange(0, H, 1, float)
    x = np.exp(-((x - H / 2) ** 2) / (2*H ))
    x = x[:, np.newaxis]
    y = np.repeat(x, W, 1)
    return y

def get_polygon_minimum_rectangle( polygon):
    if polygon is None:
        raise Exception(" Polygon of the input is none")

    [x1, y1, x2, y2, x3, y3, x4, y4] = polygon
    xMin = min(min(x1,x2),min(x3,x4))
    yMin = min(min(y1,y2),min(y3,y4))
    xMax = max(max(x1,x2),max(x3,x4))
    yMax = max(max(y1,y2),max(y3,y4))

    [X1, Y1, X2, Y2, X3, Y3, X4, Y4] = [xMin, yMin, xMax, yMin, xMax, yMax, xMin, yMax]
    return [X1, Y1, X2, Y2, X3, Y3, X4, Y4]


def rotate_bound(image, angle):
    # 获取图像的尺寸
    # 旋转中心
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)

    # 设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像旋转后的新边界
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    return cv2.warpAffine(image, M, (nW, nH))

def create_empty_image():

    #EmptyImage = np.ones((512, 512, 3), np.uint8)
    #EmptyImage = np.zeros((512, 512, 3), np.uint8)
    EmptyImage = np.zeros((512, 512), np.float)
    EmptyImage = EmptyImage*255

    points = np.array([[74,271],[440,87],[448,235],[81,406]])

    for j in range(0, 4):
        #color = (0, 0, 0)
        color = (0)

        Point1 = (int(points[j % 4][0]), int(points[j % 4][1]))
        Point2 = (int(points[(j + 1) % 4][0]), int(points[(j + 1) % 4][1]))

        cv2.line(EmptyImage, Point1, Point2, color, 3)

    # cv2.namedWindow('output_image', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('output_image', EmptyImage)
    cv2.imwrite('EmptyImage.jpg',EmptyImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    bounding_box = cv2.minAreaRect(points)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
           points[index_3], points[index_4]]

    points = np.array(box)
    for j in range(0, 4):
        #color = (0, 0, 255)
        color = (255)

        Point1 = (int(points[j % 4][0]), int(points[j % 4][1]))
        Point2 = (int(points[(j + 1) % 4][0]), int(points[(j + 1) % 4][1]))

        cv2.line(EmptyImage, Point1, Point2, color, 3)

    # cv2.namedWindow('output_image', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('output_image', EmptyImage)
    cv2.imwrite('MiniBounddingBoxImage.jpg', EmptyImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    W = 464
    H = 135
    Heatmap = MakeOneDimensionalGaussian(W,H)

    #Heatmap = rotate_bound(Heatmap,-27)

    Heatmap = (np.clip(Heatmap, 0, 1) * 255).astype(np.uint8)
    Heatmap = cv2.applyColorMap(Heatmap, cv2.COLORMAP_JET)
    cv2.namedWindow('output_image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('output_image', Heatmap)
    cv2.imwrite('MiniBounddingBoxImage.jpg', Heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    quad = [74,271,440,87,448,235,81,406]
    [x1, y1, x2, y2, x3, y3, x4, y4] = quad
    [X1, Y1, X2, Y2, X3, Y3, X4, Y4] = get_polygon_minimum_rectangle(quad)
    # 得到任意四边形的最小外接矩形的左上角坐标和其宽和高
    [x, y, w, h] = [X1, Y1, X2 - X1, Y3 - Y2]

    W = w
    H = h
    # 得到相对于矩形框的任意四边形的四个点的坐标
    [x1, y1, x2, y2, x3, y3, x4, y4] = [x1 - X1, y1 - Y1, x2 - X1, y2 - Y1, x3 - X1, y3 - Y1, x4 - X1,
                                        y4 - Y1]

    GaussianArea = MakeOneDimensionalGaussian(W, H)
    original_point = np.float32([[0, 0], [W, 0], [W, H], [0, H]])

    dstination_point = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    M = cv2.getPerspectiveTransform(original_point, dstination_point)
    dst = cv2.warpPerspective(GaussianArea, M, (W, H))

    # dst = (np.clip(dst, 0, 1) * 255).astype(np.uint8)
    # dst = cv2.applyColorMap(dst, cv2.COLORMAP_JET)
    # cv2.namedWindow('output_image', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('output_image', dst)
    # cv2.imwrite('MiniBounddingBoxImage.jpg', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for i in range(0, W):
        for j in range(0, H):
            value = dst[j, i]
            if value > 0.000001:
                EmptyImage[y + j, x + i] = dst[j, i]

    EmptyImage = (np.clip(EmptyImage, 0, 1) * 255).astype(np.uint8)
    EmptyImage = cv2.applyColorMap(EmptyImage, cv2.COLORMAP_JET)
    cv2.namedWindow('output_image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('output_image', EmptyImage)
    cv2.imwrite('MiniBounddingBoxImage.jpg', EmptyImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == '__main__':

    create_empty_image()