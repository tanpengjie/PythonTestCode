import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy.fft as nf


def read_txt_data(txt_path):
    """
    """
    with open(txt_path, 'r') as R:
        lines = [line.strip() for line in R.readlines()]
        TxtData = [int(L) for L in lines[0].split(' ')]

    TxtData = np.array(TxtData)
    return TxtData

def read_float_txt_data(txt_path):
    """
    """
    with open(txt_path, 'r') as R:
        lines = [line.strip() for line in R.readlines()]
        #TxtData = [float(L) for L in lines[0].split(' ')]
        TxtData = [int(float(L)) for L in lines[0].split(' ')]

    return TxtData
def Statistic_angle(TxtData,MaxAngle):
    AngleRange = np.arange(-MaxAngle, MaxAngle + 1, 1)
    result = np.zeros(MaxAngle*2 + 1,dtype=np.uint8)
    for i in AngleRange:
        for j in TxtData:
            if i ==j:
                result[i+MaxAngle] = result[i+MaxAngle] + 1
    return result

def get_max_value_from_array(DistanceArray):
    """
    """
    MaxValue = -1
    MaxValueIndex = 0
    ArrayLength = DistanceArray.shape[0]
    for i in range(0,ArrayLength):
        if MaxValue < DistanceArray[i]:
            MaxValue = DistanceArray[i]
            MaxValueIndex = i

    return [MaxValue,MaxValueIndex]

def Draw_scatter_points_picture(txt_path):
    """

    """
    y = read_txt_data(txt_path)
    ArrayLength = y.shape[0]
    x = np.arange(0,ArrayLength,1)

    plt.figure('Distance between warps', facecolor='lightgray')
    plt.subplot(111)
    plt.title('Distance between warps', fontsize=16)
    plt.grid(linestyle=':')
    plt.xlabel('The distance between two adjacent yarns (unit pixel)',fontsize=16)
    plt.ylabel('Number of pixels at the same distance',fontsize=16)

    Gap = np.arange(0,ArrayLength,2)
    plt.xticks(Gap)

    [MaxValue, MaxValueIndex] = get_max_value_from_array(y)
    #plt.axis([0, ArrayLength, 0, MaxValue + 100])
    plt.grid(True)
    plt.plot(x, y, label=r'$y$')
    plt.scatter(x,y)

    plt.scatter([MaxValueIndex],[MaxValue], color='red')
    plt.text(MaxValueIndex + 2, MaxValue, r'Distance=%s'%MaxValueIndex,fontsize='12')

    plt.plot([MaxValueIndex,MaxValueIndex],[0,MaxValue],color='black',linestyle='dashed')
    plt.show()

def Draw_histagram_picture(txt_path):
    """
    """
    MaxAngle = 10
    y = read_float_txt_data(txt_path)
    y = Statistic_angle(y,MaxAngle)
    AngleRange = np.arange(-MaxAngle,MaxAngle+1,1)
    plt.bar(AngleRange, y, label='Statistic Angle')
    # params
    # x: 条形图x轴
    # y：条形图的高度
    # width：条形图的宽度 默认是0.8
    # bottom：条形底部的y坐标值 默认是0
    # align：center / edge 条形图是否以x轴坐标为中心点或者是以x轴坐标为边缘
    plt.legend()
    plt.xlabel('Angle')
    plt.ylabel('Number')
    plt.xticks(AngleRange)
    plt.title(u'Statistic certain range angle')

    for i in range(0,MaxAngle*2 + 1):
        if y[i]==0:
            continue
        plt.text(i - MaxAngle, y[i] + 1, "%s" % y[i], va='center')
    plt.show()

if __name__ == '__main__':

    #Draw_scatter_points_picture('E:\\VSProjects\\MeasureWeftAngle\\distance.txt')
    Draw_histagram_picture('E:\\VSProjects\\MeasureWeftAngle\\Horizontal_angle.txt')