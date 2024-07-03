import tensorflow as tf
import cv2
import os
import random

import numpy as np

from tensorflow.keras import mixed_precision

from tensorflow.keras.utils import normalize
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, \
    BatchNormalization, Dropout, Lambda
from tensorflow.keras.initializers import RandomUniform, RandomNormal
from mode import specseg
from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection
from skimage.transform import resize
from skimage.color import rgb2gray
from PIL import Image
from sklearn.model_selection import train_test_split
from natsort import natsorted
import segmentation_models as sm


from tensorflow.keras.layers import Conv2D, Input, Dropout, BatchNormalization, MaxPooling2D,Add, Conv2DTranspose, concatenate, Activation, Multiply

def attention_gate(input, g, inter_channels):
    # Global average pooling
    theta_x = Conv2D(inter_channels, (1, 1), strides=(1, 1), padding='same')(input)
    phi_g = Conv2D(inter_channels, (1, 1), strides=(1, 1), padding='same')(g)
    phi_g = MaxPooling2D(pool_size=(2, 2))(phi_g)  # 添加最大池化以匹配形状
    f = Activation('relu')(Add()([theta_x, phi_g]))
    psi_f = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(f)
    rate = Activation('sigmoid')(psi_f)
    att_x = Multiply()([input, rate])
    return att_x


def specseg_with_attention(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path with attention gates
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(c1)
    c1 = BatchNormalization(axis=-1)(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(c2)
    c2 = BatchNormalization(axis=-1)(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    att1 = attention_gate(p2, p1, 16)
    p2 = concatenate([p2, att1])

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(c3)
    c3 = BatchNormalization(axis=-1)(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    att2 = attention_gate(p3, p2, 32)
    p3 = concatenate([p3, att2])

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(c4)
    c4 = BatchNormalization(axis=-1)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    att3 = attention_gate(p4, p3, 64)
    p4 = concatenate([p4, att3])

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(c5)
    c5 = BatchNormalization(axis=-1)(c5)

    # Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='RandomNormal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 0.5]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    opt = tf.keras.optimizers.Adam(learning_rate=0.00002)

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    model.compile(optimizer=opt, loss=total_loss, metrics=metrics)
    model.summary()

    return model





image_directory = 'yourpath_image_directory'
mask_directory = 'yourpath_mask_directory'

SIZE = 256
image_dataset = []  # Many ways to handle data, you can use pandas. Here, we are using a list format.
mask_dataset = []  # Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

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

# Normalize images
image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1), 3)
# D not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.

X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.10, random_state=0)

IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]
print(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
model = specseg(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('./checkpoints/bigdata_SpecSeg_weights_{epoch:02d}.hdf5',  # 文件名包含 epoch 数字
                             monitor='val_loss',  # 监控验证集上的损失
                             verbose=1,
                             save_weights_only=True,  # 只保存权重而不保存整个模型
                             period=100)  # 每100个 epoch 保存一次权重

# 编译模型
# 训练模型并使用 ModelCheckpoint 回调函数
history = model.fit(X_train, y_train,
                    batch_size=16,
                    verbose=1,
                    epochs=200,
                    validation_split=0.2,
                    validation_data=(X_test, y_test),
                    shuffle=False,
                    callbacks=[checkpoint])  # 传递 ModelCheckp
FScore = model.evaluate(X_test, y_test)
print("SparseCategoricalCrossentropy = ", (FScore * 100), "%")
import matplotlib.pyplot as plt

# 获取训练过程中记录的损失值和验证损失值数据
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# 获取训练过程中的 epoch 数量
epochs = range(1, len(train_loss) + 1)

# 画出训练集和测试集损失值图表
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot_500.png')
plt.show()

#
# from matplotlib import cm, pyplot as plt
#
# model.load_weights('D:/anconda/test/SpecSeg-main/SpecSeg_weights.hdf5')  # Trained for 50 epochs and then additional 100
# test_indices = [10, 20, 30, 40, 50]
# # Enter number of images to test on
# num_images = 5
# index = 0
# fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 32))
# fig.tight_layout()
# for i, test_img_index in enumerate(test_indices):
#     test_img = X_test[test_img_index]
#     ground_truth = y_test[test_img_index]
#     test_img_norm = test_img[:, :, 0][:, :, None]
#     test_img_input = np.expand_dims(test_img_norm, 0)
#     prediction = (model.predict(test_img_input)[0, :, :, 0])
#
#     plt.subplot(num_images, 3, index + 1)
#     plt.title('Test Image')
#     plt.axis('off')
#     plt.imshow(test_img[:, :, 0], cmap='gray')
#     plt.subplot(num_images, 3, index + 2)
#     plt.title('Ground Truth')
#     plt.axis('off')
#     plt.imshow(ground_truth[:, :, 0], cmap='gray')
#     plt.subplot(num_images, 3, index + 3)
#     plt.title('Predicted Specular Highlights')
#     plt.axis('off')
#     plt.imshow(prediction, cmap='gray')
#     index = 3 * (i + 1)
#   # save the figure to file
# print("Done!")

