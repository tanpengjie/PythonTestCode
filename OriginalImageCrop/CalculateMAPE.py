import numpy as np
import cv2
import os
import math


def CalculateWarpMAPE(GroundTruchPath,WarpMeasuredResultPath):

    suffix = ['jpg', 'png', 'jpeg', 'bmp']
    AllWarpAPEList = []
    AllWarpSPEList = []

    MAPE = 0
    MSPE = 0
    for i in os.listdir(WarpMeasuredResultPath):
        if i.split('.')[-1] not in suffix:
            continue
        print(i)
        WarpResultPath = WarpMeasuredResultPath + "\\" + i.replace('.jpg','.txt')
        TruthWarpResultPath = GroundTruchPath + "\\" + i.replace('.jpg','.txt')

        WarpNumber = []
        with open(WarpResultPath,'r') as R:
            lines = [line.strip() for line in R.readlines()]

            for line in lines:
                if len(line) < 1:
                    continue
                WarpNumber.append(int(line))

        TruthWarpNumber = []
        with open(TruthWarpResultPath,'r') as R:
            lines = [line.strip() for line in R.readlines()]

            for line in lines:
                if len(line) < 1:
                    continue

                if 'Warp' in line:
                    WarpNumberValue = line.split(':')[1]
                    TruthWarpNumber.append(float(WarpNumberValue))

        if len(WarpNumber) !=1 or len(TruthWarpNumber)!=1:
            raise Exception("Contxt of txt is Error...")

        AllWarpAPEList.append(i.replace('.jpg',''))
        AllWarpSPEList.append(i.replace('.jpg',''))

        APE = abs(TruthWarpNumber[0] - WarpNumber[0]/100.0) / TruthWarpNumber[0]
        #APE = abs(TruthWarpNumber[0] - WarpNumber[0]) / TruthWarpNumber[0]
        MAPE = MAPE + APE*100
        AllWarpAPEList.append(APE*100)

        SPE = math.pow(APE,2)
        AllWarpSPEList.append(SPE)
        MSPE = MSPE + SPE

    MAPE = float(MAPE) / len(os.listdir(WarpMeasuredResultPath))
    MSPE = float(MSPE) / len(os.listdir(WarpMeasuredResultPath))
    MSPE = math.sqrt(MSPE)*100

    print(AllWarpAPEList)
    print(AllWarpSPEList)
    print(MAPE)
    print(MSPE)


def CalculateWeftMAPE(GroundTruchPath,WarpMeasuredResultPath):

    suffix = ['jpg', 'png', 'jpeg', 'bmp']
    AllWarpAPEList = []
    AllWarpSPEList = []

    MAPE = 0
    MSPE = 0
    for i in os.listdir(WarpMeasuredResultPath):
        if i.split('.')[-1] not in suffix:
            continue
        print(i)
        WarpResultPath = WarpMeasuredResultPath + "\\" + i.replace('.jpg','.txt')
        TruthWarpResultPath = GroundTruchPath + "\\" + i.replace('.jpg','.txt')

        WarpNumber = []
        with open(WarpResultPath,'r') as R:
            lines = [line.strip() for line in R.readlines()]

            for line in lines:
                if len(line) < 1:
                    continue
                WarpNumber.append(int(line))

        TruthWarpNumber = []
        with open(TruthWarpResultPath,'r') as R:
            lines = [line.strip() for line in R.readlines()]

            for line in lines:
                if len(line) < 1:
                    continue

                if 'Weft' in line:
                    WarpNumberValue = line.split(':')[1]
                    TruthWarpNumber.append(float(WarpNumberValue))

        if len(WarpNumber) !=1 or len(TruthWarpNumber)!=1:
            raise Exception("Contxt of txt is Error...")

        AllWarpAPEList.append(i.replace('.jpg',''))
        AllWarpSPEList.append(i.replace('.jpg',''))

        APE = abs(TruthWarpNumber[0] - WarpNumber[0]/100.0) / TruthWarpNumber[0]
        #APE = abs(TruthWarpNumber[0] - WarpNumber[0]) / TruthWarpNumber[0]
        MAPE = MAPE + APE*100
        AllWarpAPEList.append(APE*100)

        SPE = math.pow(APE,2)
        AllWarpSPEList.append(SPE)
        MSPE = MSPE + SPE

    MAPE = float(MAPE) / len(os.listdir(WarpMeasuredResultPath))
    MSPE = float(MSPE) / len(os.listdir(WarpMeasuredResultPath))
    MSPE = math.sqrt(MSPE)*100

    print(AllWarpAPEList)
    print(AllWarpSPEList)
    print(MAPE)
    print(MSPE)

if __name__ == '__main__':
    CalculateWeftMAPE("E:\\ImageData\\WarpAndWeftSeparation\\PaperTestingImage","E:\\PythonProjects\\SA-Text-Location\\VisualResult\\LinearHeatmapClassical\\WeftHeatmap")







