"""
 Create gaussain heatmap in the rectangle, The pixel value at each point in the rectangle are equal to
 the average value of the Gaussian in the X direction and the Gaussain in the Y direction
"""

#author : TanPengjie by 2020.12.15
import numpy as np
import math
import cv2

def gaussain_2d_rect(height,width):
    """
    Create a 2-dimensional isotropic Gaussain map 生成二维高斯球形映射图
    :param height:
    :param width:
    :return:
    """
    mean = 0
    radius = 2.5

    a =1
    detla = 10.0 / height
    x0,x1 = np.meshgrid(np.arange(-5,5,detla),np.ones(width))
    m0 = (x0 - mean)**2
    m0 = m0.reshape(-1)

    gaussain_map = a * np.exp(-0.5*m0 / (radius**2))
    gaussain_map = gaussain_map.reshape(width,-1).T

    max_prob = np.max(gaussain_map)
    min_prob = np.min(gaussain_map)
    gaussain_map = (gaussain_map - min_prob) / (max_prob - min_prob)
    gaussain_map = np.clip(gaussain_map,0,1)

    return gaussain_map

def create_standard_normal_distribution1(length):
    x = np.arange(0,4,0.1,float)
    y = np.exp(- (x-2)*(x-2) / 2)
    return y.reshape(1,-1)

def create_standard_normal_distribution(length):
    delta = 6 / length
    x = np.arange(-3,3,delta,float)
    y = np.exp(- x*x / 2)
    #return y.reshape(1,-1)
    return y

def create_standard_normal_distribution2(W,H):
    delta = 6 / H
    x = np.arange(-3,3,delta,float)
    x = np.exp(- x*x / 2)
    x = x[:, np.newaxis]

    y = np.repeat(x, W, 1)
    return y

def MakeOneDimensionalGaussian(W,H):
    W = int(W)
    H = int(H)

    x = np.arange(0, H, 1, float)
    x = np.exp(-((x - H / 2) ** 2) / (3 * H / 2))
    x = x[:, np.newaxis]

    y = np.repeat(x, W, 1)
    return y


def create_linear_distribution(length):
    k = 2 / length
    x = np.arange(-length/2,length/2,1,float)
    y = x * k
    return y


def make_gaussain_heatmap(rect):
    if len(rect) != 4:
        raise Exception("     Please input right rectangle information......")

    x, y, w, h = rect
    if w < 5 or h < 5:
        print("    Rectangle's w or h is less than 5, Please be care of.......")

    xGaussainValuesRange = create_standard_normal_distribution(w)
    yGaussainValuesRange = create_standard_normal_distribution(h)

    rectGaussainValue = np.zeros((w,h),dtype=float)
    for i in range(0,w):
        for j in range(0,h):
            rectGaussainValue[i][j] = math.sqrt(math.sqrt(xGaussainValuesRange[i] * yGaussainValuesRange[j]))
            #rectGaussainValue[i][j] = (xGaussainValuesRange[i] * yGaussainValuesRange[j])
    print(rectGaussainValue.shape)

    return rectGaussainValue

def make_linear_heatmap(rect, flag=0):
    """
    #Create linear heatmap, the value of pixels in the heatmap is based on rectangle width,
    #The value of pixels in a rectangular box are gradually reduced from the middle to the sides.
    :param rect: x,y,w,h
    :param flag: if flag equal 0, it will create linear heatmap at Y axis direction, else it will create linear heatmap at X axis directions.
    :return:linear heatmap
    """
    if len(rect) != 4:
        raise Exception("     Please input right rectangle information......")

    x, y, w, h = rect
    if w < 5 or h < 5:
        print("    Rectangle's w or h is less than 5, Please be care of.......")

    xLinearValue = create_linear_distribution(w)
    yLinearValue = create_linear_distribution(h)

    rectGaussainValue = np.zeros((h, w), dtype=float)
    for i in range(0, h):
        for j in range(0, w):
            if flag==0:
                rectGaussainValue[i][j] = math.fabs(math.fabs(yLinearValue[i]) - 1)
            elif flag==1:
                rectGaussainValue[i][j] = math.fabs(math.fabs(xLinearValue[j]) - 1)
            else:
                raise Exception("   Please input right Linear Heatmap axisi's parameter")

    print(rectGaussainValue.shape)
    return rectGaussainValue

def read_image_cv2(image_path):
    image = cv2.imread(image_path,0)
    cv2.imwrite("1.jpg",image)

#得到多边形的最小外接矩形
def get_polygon_minimum_rectangle( polygon):
    if polygon is None:
        raise Exception(" Polygon of the input is none")

    [x1, y1, x2, y2, x3, y3, x4, y4] = polygon
    xMin = min(min(x1,x2),min(x3,x4))
    yMin = min(min(y1,y2),min(y3,y4))
    xMax = max(max(x1,x2),max(x3,x4))
    yMax = max(max(y1,y2),max(y3,y4))

    return [xMin, yMin, xMax, yMin, xMax, yMax, xMin, yMax]

if __name__ == "__main__":

    Rectangle = get_polygon_minimum_rectangle([0,0,400,0,400,50,0,50])
    disReactangle = [10,10,300,20,350,40,20,50]
    a = MakeOneDimensionalGaussian(400,50)
    a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    a = cv2.applyColorMap(a, cv2.COLORMAP_JET)

    pts = np.float32([[0,0],[400,0],[400,50],[0,50]])

    pts1 = np.float32([[10,10], [300,20], [350,40], [20, 50]])

    M = cv2.getPerspectiveTransform(pts, pts1)

    dst = cv2.warpPerspective(a, M, (400, 50))
    cv2.imwrite("E:\\PythonProjects\\PythonTestCode\\a.jpg", dst)

    gaussain_image1 = create_standard_normal_distribution2(200,20)
    gaussain_image1 = (np.clip(gaussain_image1, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite("E:\\PythonProjects\\PythonTestCode\\gaussain_image1.jpg", gaussain_image1)

    gaussain_image2 = MakeOneDimensionalGaussian(200,20)
    gaussain_image2 = (np.clip(gaussain_image2, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite("E:\\PythonProjects\\PythonTestCode\\gaussain_image2.jpg", gaussain_image2)

    rect = [50,50,300,100]
    #rectGaussainValue = make_gaussain_heatmap(rect)
    rectGaussainValue = make_linear_heatmap(rect)
    image = (np.clip(rectGaussainValue, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite("E:\\PythonProjects\\PythonTestCode\\gaussian.jpg", image)
    image = image.astype(np.uint8)
    img = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    cv2.imwrite("E:\\PythonProjects\\PythonTestCode\\color.jpg", img)
    print("finished")






