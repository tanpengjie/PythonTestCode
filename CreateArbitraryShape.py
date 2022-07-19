import random
from PIL import Image
import numpy as np
import cv2


def Unique(points_collection):
    new_points_collection = []
    for i in points_collection:
        if i not in new_points_collection:
            new_points_collection.append(i)
    return new_points_collection

def CreateArbitrary(count):
    image_path = 'E:\\PythonProjects\\A_surface_defect_detection_based_on_positive_samples-master\\22.bmp'
    input_image = Image.open(image_path).convert('L')
    #input_image.show(input_image)
    input_image_array = np.array(input_image)

    point_number = [400, 25, 40, 50, 100]
    random_number = point_number[random.randint(0,len(point_number))-1]

    original_x = random.randint(0,128)
    original_y = random.randint(0,128)

    points_collection = []
    points_collection.append([original_x,original_y])

    temp_points_collections = []
    offset_lists = [[-1,0], [1,0], [0,-1], [0,1]]
    for i in range(1,random_number):
        for j in range(0,4):
            if original_x + offset_lists[j][0] > 0 and original_x + offset_lists[j][0] < 128 and \
                    original_y + offset_lists[j][1] > 0 and original_y + offset_lists[j][1] < 128:
                temp_points_collections.append([original_x + offset_lists[j][0] , original_y + offset_lists[j][1]])
        temp_points_collections = Unique(temp_points_collections)
        if len(temp_points_collections) < 1:
            continue
        original_x,original_y = temp_points_collections[random.randint(1,len(temp_points_collections)) - 1]
        input_image_array[original_x,original_y] = 255
        points_collection.append([original_x,original_y])
        #print(original_x,original_y)
    import cv2
    input_image = Image.fromarray(np.uint8(input_image_array))
    #input_image = input_image.astype(np.uint8)
    #input_image.show(input_image)
    save_image_path = "./ArbitraryShapeDefectImage/"
    cv2.imwrite(save_image_path + str(count) + '.jpg',input_image_array)
    #print("已完成随机点的生成")

def CreateLineDefect(count):
    image_path = 'E:\\PythonProjects\\A_surface_defect_detection_based_on_positive_samples-master\\Image.png'
    input_image = Image.open(image_path).convert('L')
    #input_image.show(input_image)
    input_image_array = np.array(input_image)

    crop_image_path = 'E:\\PythonProjects\\PythonTestCode\\OriginalImageCrop\\1.png'
    crop_image = Image.open(crop_image_path).convert('L')
    crop_image_array = np.array(crop_image)

    image_width = crop_image_array.shape[0]
    image_height = crop_image_array.shape[1]
    pixel_value = crop_image_array[int(image_width / 2),int(image_height / 2)]
    #print(crop_image_array.shape[0],crop_image_array.shape[1])
    point_number = [400,25, 40, 50, 100]
    random_number = point_number[random.randint(0, len(point_number)) - 1]

    original_x = random.randint(0, 128)
    original_y = random.randint(0, 128)

    points_collection = []
    points_collection.append([original_x, original_y])

    temp_points_collections = []
    offset_lists = []
    offset_lists1 = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    #offset_lists2 = [[-1, 0], [1, 0], [-1, 0], [1, 0]]
    #offset_lists2 = [[-2, 0], [-5, 0], [3, 0], [1, 0]]
    offset_lists2 = [[0, 2], [0, 2], [0, 3], [0, 1]]
    #offset_lists2 = [[-2, 1], [-2, 1], [-2, 1], [-2, 1]]
    #offset_lists2 = [[1, -2], [1, -2], [1, -2], [1, -2]]
    #offset_lists2 = [[2, 2], [1, 1], [2, 2], [1, 1]]

    for i in range(1, random_number):
        for j in range(0, 4):
            randon_offset = random.randint(0,10)
            if randon_offset > 9:
                offset_lists = offset_lists1
            else:
                offset_lists = offset_lists2

            if original_x + offset_lists[j][0] > 0 and original_x + offset_lists[j][0] < 128 and \
                    original_y + offset_lists[j][1] > 0 and original_y + offset_lists[j][1] < 128:
                temp_points_collections.append([original_x + offset_lists[j][0], original_y + offset_lists[j][1]])
        temp_points_collections = Unique(temp_points_collections)
        if len(temp_points_collections) < 1:
            continue
        original_x, original_y = temp_points_collections[random.randint(1, len(temp_points_collections)) - 1]
        input_image_array[original_x, original_y] = pixel_value
        points_collection.append([original_x, original_y])
        # print(original_x,original_y)
    import cv2
    input_image = Image.fromarray(np.uint8(input_image_array))
    # input_image = input_image.astype(np.uint8)
    # input_image.show(input_image)
    save_image_path = "./ArbitraryShapeDefectImage/"
    cv2.imwrite(save_image_path + str(count) + '.jpg', input_image_array)
    # print("已完成随机点的生成")


if __name__== "__main__":
    for i in range(0,5000):
        print(i)
        #CreateArbitrary(i)
        CreateLineDefect(i)






