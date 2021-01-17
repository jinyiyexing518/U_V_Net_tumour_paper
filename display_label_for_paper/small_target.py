from display_label_for_paper.display_label import display_image
import cv2 as cv
import os


dir_num = "6"
dir_path = os.path.join("./small_target/", str(dir_num))
img_sort = os.listdir(dir_path)
small_target = ""
for img in img_sort:
    if img.split('.')[0] == 'label':
        continue
    elif img.split('.')[0] == 'train':
        continue
    else:
        small_target = img


train = "./small_target/" + dir_num + "/train.png"
label = "./small_target/" + dir_num + "/label.png"
small_target = "./small_target/" + dir_num + "/" + small_target


if __name__ == '__main__':
    # count = len(os.listdir("./small_target"))
    # for i in range(1, count+1):
    #     display_path = "./small_target/" + str(i) + "_display"
    #     if not os.path.isdir(display_path):
    #         os.makedirs(display_path)

    label_predict = display_image(train, label)
    cv.imshow("label", label_predict)
    cv.imwrite("./small_target/" + dir_num + "_display/label_display.png", label_predict)

    small_target_predict = display_image(train, small_target)
    cv.imshow("small_target", small_target_predict)
    cv.imwrite("./small_target/" + dir_num + "_display/small_target_predict.png", small_target_predict)

    cv.waitKey(0)
    cv.destroyAllWindows()







