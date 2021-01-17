from display_label_for_paper.display_label import display_image
import cv2 as cv
import os


dir_num = "1"
dir_path = os.path.join("./img/", str(dir_num))
img_sort = os.listdir(dir_path)
unet = ""
vnet = ""
refine = ""
uvnet = ""
multi = ""
u_double = ""
u_residual = ""
v_dense = ""
for img in img_sort:
    if img.split('.')[0] == 'label':
        continue
    elif img.split('.')[0] == 'train':
        continue
    algorithm_name = img.split('.')[-2].split('_')[-2]
    if algorithm_name == 'U':
        unet = img
    elif algorithm_name == 'V':
        vnet = img
    elif algorithm_name == 'Refine':
        refine = img
    elif algorithm_name == 'UV':
        uvnet = img
    elif algorithm_name == 'Multi':
        multi = img
    elif algorithm_name == '++':
        u_double = img
    elif algorithm_name == 'residual':
        u_residual = img
    elif algorithm_name == 'dense':
        v_dense = img


train = "./img/" + dir_num + "/train.png"
label = "./img/" + dir_num + "/label.png"
unet_img = "./img/" + dir_num + "/" + unet
vnet_img = "./img/" + dir_num + "/" + vnet
refine_img = "./img/" + dir_num + "/" + refine
uvnet_img = "./img/" + dir_num + "/" + uvnet
uvnet_multi_img = "./img/" + dir_num + "/" + multi
u_double_img = "./img/" + dir_num + "/" + u_double
u_residual_img = "./img/" + dir_num + "/" + u_residual
v_dense_img = "./img/" + dir_num + "/" + v_dense


if __name__ == '__main__':
    label_predict = display_image(train, label)
    cv.imshow("label", label_predict)
    cv.imwrite("./img/" + dir_num + "_display/label_display.png", label_predict)

    unet_predict = display_image(train, unet_img)
    cv.imshow("unet", unet_predict)
    cv.imwrite("./img/" + dir_num + "_display/unet_display.png", unet_predict)

    vnet_predict = display_image(train, vnet_img)
    cv.imshow("vnet", vnet_predict)
    cv.imwrite("./img/" + dir_num + "_display/vnet_display.png", vnet_predict)

    refine_predict = display_image(train, refine_img)
    cv.imshow("refine", refine_predict)
    cv.imwrite("./img/" + dir_num + "_display/refine_display.png", refine_predict)

    uvnet_predict = display_image(train, uvnet_img)
    cv.imshow("uvnet", uvnet_predict)
    cv.imwrite("./img/" + dir_num + "_display/uvnet_display.png", uvnet_predict)

    uvnet_multi_predict = display_image(train, uvnet_multi_img)
    cv.imshow("uvnet_multi", uvnet_multi_predict)
    cv.imwrite("./img/" + dir_num + "_display/uvnet_multi_display.png", uvnet_multi_predict)

    u_double_predict = display_image(train, u_double_img)
    cv.imshow("u_double", u_double_predict)
    cv.imwrite("./img/" + dir_num + "_display/u_double_display.png", u_double_predict)

    u_residual_predict = display_image(train, u_residual_img)
    cv.imshow("u_residual", u_residual_predict)
    cv.imwrite("./img/" + dir_num + "_display/u_residual_display.png", u_residual_predict)

    v_dense_predict = display_image(train, v_dense_img)
    cv.imshow("v_dense", v_dense_predict)
    cv.imwrite("./img/" + dir_num + "_display/v_dense_display.png", v_dense_predict)

    cv.waitKey(0)
    cv.destroyAllWindows()







