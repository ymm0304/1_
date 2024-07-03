from mode import specseg_with_attention1,specseg
import os
import glob
from matplotlib import cm, pyplot as plt
# 获取所有模型文件的路径
import tensorflow as tf
import cv2
import os
import random

import numpy as np


from tensorflow.keras.utils import normalize
from PIL import Image
from sklearn.model_selection import train_test_split
from natsort import natsorted



from tensorflow.keras.layers import Conv2D, Input, Dropout, BatchNormalization, MaxPooling2D,Add, Conv2DTranspose, concatenate, Activation, Multiply

image_directory = 'yourpath_image_directory'
mask_directory = 'yourpath_mask_directory'

SIZE = 256
image_dataset = []  # Many ways to handle data, you can use pandas. Here, we are using a list format.
mask_dataset = []  # Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.
image_dataset1 = []
print("loading images")
images = natsorted(os.listdir(image_directory))
for i, image_name in enumerate(images):  # Remember enumerate method adds a counter and returns the enumerate object
    if image_name.split('.')[1] == 'png':
        # print(image_directory+image_name)
        image = cv2.imread(image_directory + image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        # q = np.array(image)
        image_dataset.append(np.array(image))

print("loading masks")
masks = natsorted(os.listdir(mask_directory))
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(mask_directory + image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))
for i, image_name in enumerate(images):  # Remember enumerate method adds a counter and returns the enumerate object
    if image_name.split('.')[1] == 'png':
        # print(image_directory+image_name)
        image = cv2.imread(image_directory + image_name,cv2.IMREAD_GRAYSCALE)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        # q = np.array(image)
        image_dataset1.append(np.array(image))
# Normalize images
image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1), 3)
# D not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.

X_test, y_test = image_dataset, mask_dataset

# 获取所有模型文件的路径
# model_files = glob.glob('D:/anconda/test/checkpoints/*.hdf5')
# test_indices = [10, 20, 30, 40, 50]
# model=specseg_with_attention(256,256,1)
# # 循环遍历模型文件并测试
# for model_file in model_files:
#     # 加载模型
#     model.load_weights(model_file)
#     num_images = 5
#     index = 0
#     fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 32))
#     fig.tight_layout()
#     for i, test_img_index in enumerate(test_indices):
#
#         test_img = X_test[test_img_index]
#         ground_truth = y_test[test_img_index]
#         test_img_norm = test_img[:, :, 0][:, :, None]
#         test_img_input = np.expand_dims(test_img_norm, 0)
#         prediction = (model.predict(test_img_input)[0, :, :, 0])
#
#         plt.subplot(num_images, 3, index + 1)
#         plt.title('Test Image')
#         plt.axis('off')
#         plt.imshow(test_img[:, :, 0], cmap='gray')
#         plt.subplot(num_images, 3, index + 2)
#         plt.title('Ground Truth')
#         plt.axis('off')
#         plt.imshow(ground_truth[:, :, 0], cmap='gray')
#         plt.subplot(num_images, 3, index + 3)
#         plt.title('Predicted Specular Highlights')
#         plt.axis('off')
#         plt.imshow(prediction, cmap='gray')
#         index = 3 * (i + 1)
#     file_name = os.path.basename(model_file)
#     result_file_name = f'predictions_{file_name}.png'
#     result_file_path = os.path.join('D:/anconda/test/result', result_file_name)
#     fig.savefig(result_file_path)  # save the figure to file
#     print("Done!")
#     # 进行测试
#     # 这里添加您的测试代码，使用当前加载的模型进行测试
#
#     print(f"Model {model_file} tested successfully.")
model=specseg_with_attention1(256,256,1)
model1=specseg(256,256,1)
model.load_weights('weights_path')  # Trained for 50 epochs and then additional 100
model1.load_weights('weights_path')
num_images = len(X_test)
import cv2
# 创建目录以保存预测结果和对比图像
save_dir = 'your_savepath'
predict_dir = os.path.join(save_dir, 'predict')
predict_dir1 = os.path.join(save_dir, 'predict1')
compare_dir = os.path.join(save_dir, 'compare')
gt_dir = os.path.join(save_dir, 'gt')
os.makedirs(predict_dir, exist_ok=True)
os.makedirs(compare_dir, exist_ok=True)
os.makedirs(predict_dir1, exist_ok=True)
os.makedirs(gt_dir, exist_ok=True)
for i, test_img in enumerate(X_test):
    ground_truth = y_test[i]
    test_img_norm = test_img[:, :, 0][:, :, None]
    test_img_input = np.expand_dims(test_img_norm, 0)
    prediction = (model.predict(test_img_input)[0, :, :, 0])
    prediction1=(model1.predict(test_img_input)[0, :, :, 0])
    # 保存预测结果
    predict_filename = f'predict_{4000+i}.png'
    predict_path = os.path.join(predict_dir, predict_filename)
    cv2.imwrite(predict_path, (prediction * 255).astype(np.uint8))

    gt_filename = f'predict_{4000 + i}.png'
    gt_path = os.path.join(gt_dir, gt_filename)
    test_img_uint8 = np.uint8(test_img * 255)
    test_img_rgb = cv2.cvtColor(test_img_uint8, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(gt_path, image_dataset1[i])



    predict_filename1 = f'predict_{4000 + i}.png'
    predict_path1 = os.path.join(predict_dir1, predict_filename1)
    cv2.imwrite(predict_path1, (prediction1 * 255).astype(np.uint8))
    # 绘制对比图像
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    # plt.title('Test Image')
    plt.axis('off')
    plt.imshow(test_img[:, :, 0], cmap='gray')

    plt.subplot(1, 4, 2)
    # plt.title('Ground Truth')
    plt.axis('off')
    plt.imshow(ground_truth[:, :, 0], cmap='gray')

    plt.subplot(1, 4, 3)
    # plt.title('speSeg')
    plt.axis('off')
    plt.imshow(prediction1, cmap='gray')

    plt.subplot(1, 4, 4)
    # plt.title('Predicted Specular Highlights')
    plt.axis('off')
    plt.imshow(prediction, cmap='gray')



    # 保存对比图像
    compare_filename = f'compare_{4000+i}.png'
    compare_path = os.path.join(compare_dir, compare_filename)
    plt.savefig(compare_path)
    plt.close()

print("Done!")
