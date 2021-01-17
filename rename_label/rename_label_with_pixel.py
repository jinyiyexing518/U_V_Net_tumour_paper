import os
import cv2 as cv


label_dir_path = "../u_net/dataset/data_cv_clip/test/label/129"
image_dir_path = "../u_net/dataset/data_cv_clip/test/train/129"

label_save_path = "./image129_label"
image_save_path = "./image129_image"
if not os.path.isdir(label_save_path):
    os.makedirs(label_save_path)
if not os.path.isdir(image_save_path):
    os.makedirs(image_save_path)

label_names = os.listdir(label_dir_path)
image_names = os.listdir(image_dir_path)

for label_name, image_name in zip(label_names, image_names):
    label_path = os.path.join(label_dir_path, label_name)
    image_path = os.path.join(image_dir_path, image_name)

    label = cv.imread(label_path, 0)
    image = cv.imread(image_path, 0)

    white = label == 255
    pixel_num = len(label[white])

    cv.imwrite(os.path.join(label_save_path, "pixel_num_" + str(pixel_num) + ".png"), label)
    cv.imwrite(os.path.join(image_save_path, "pixel_num_" + str(pixel_num) + ".png"), image)
    # cv.imshow("image", image)
    # cv.imshow("label", label)
    # cv.waitKey(0)
    # cv.destroyAllWindows()







