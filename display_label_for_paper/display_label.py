import cv2 as cv

img_path = "./image.png"
label_path = "./label.png"


def display_image(img_path, label_path):
    label = cv.imread(label_path, 0)
    image = cv.imread(img_path)
    for i in range(3):
        image[:, :, i] = cv.bitwise_and(image[:, :, 0], 255 - label)
    image[:, :, 2] += label

    # 0通道蓝色，存放label；2通道红色，存放边缘；1通道黄色，存放全黑用于判断
    # row, col = label.shape
    # for i in range(1, row - 1):
    #     for j in range(1, col - 1):
    #         if label[i, j] != 0:
    #             if image[i - 1, j, 1] != 0 or image[i, j - 1, 1] != 0 or \
    #                     image[i + 1, j, 1] != 0 or image[i, j + 1, 1] != 0:
    #                 if image[i, j, 1] == 0:
    #                     image[i, j, 0] = 255
    return image


if __name__ == "__main__":
    image = display_image(img_path, label_path)
    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()




