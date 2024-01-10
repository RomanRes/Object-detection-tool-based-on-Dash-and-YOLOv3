import struct
import numpy as np
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Input, Add, \
                                    ZeroPadding2D, UpSampling2D, Concatenate
from tensorflow.keras import Model
from keras.models import load_model



# darknet_conv,  darknet_residual_block, convolutional_set, yolo_end are the helper function to build YOLOv3 network

def darknet_conv(x, filters, kernel_size, padding, idx,
                 strides=(1, 1), batch_norm=True, leaky=True, use_bias=False):
    """
    This function define darknet convolutional block
    """
    # con2d
    if strides == (1, 1):
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
               padding=padding, use_bias=use_bias, name='conv_' + str(idx))(x)

    # BN
    if batch_norm:
        x = BatchNormalization(momentum=0.99, epsilon=0.001, name='batch_norm' + str(idx))(x)

    # LeakyReLU
    if leaky:
        x = LeakyReLU(alpha=0.1, name='leaky' + str(idx))(x)
    return x


def darknet_residual_block(x, filters, idx):
    shortcut = x
    x = darknet_conv(x, filters=filters, kernel_size=(1, 1), strides=(1, 1), padding="same", idx=idx)
    x = darknet_conv(x, filters=filters * 2, kernel_size=(3, 3), strides=(1, 1), padding="same", idx=idx + 1)
    x = Add()([shortcut, x])
    return x


def convolutional_set(x, filters, idx):
    x = darknet_conv(x, filters=filters, kernel_size=(1, 1), strides=(1, 1), padding="same", idx=idx)
    x = darknet_conv(x, filters=filters * 2, kernel_size=(3, 3), strides=(1, 1), padding="same", idx=idx + 1)
    x = darknet_conv(x, filters=filters, kernel_size=(1, 1), strides=(1, 1), padding="same", idx=idx + 2)
    x = darknet_conv(x, filters=filters * 2, kernel_size=(3, 3), strides=(1, 1), padding="same", idx=idx + 3)
    x = darknet_conv(x, filters=filters, kernel_size=(1, 1), strides=(1, 1), padding="same", idx=idx + 4)
    return x


def yolo_end(x, filters, idx):
    x = darknet_conv(x, filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same", idx=idx)
    x = darknet_conv(x, filters=255, kernel_size=(1, 1), strides=(1, 1),
                     padding="same", batch_norm=False, leaky=False, use_bias=True, idx=idx + 1)
    return x


def yoloV3(name=None):
    inputs = Input(shape=(None, None, 3))  #
    # 0 --> 4
    x = darknet_conv(inputs, filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", idx=0)
    x = darknet_conv(x, filters=64, kernel_size=(3, 3), strides=(2, 2), padding="valid", idx=1)
    x = darknet_residual_block(x, filters=32, idx=2)
    x = darknet_conv(x, filters=128, kernel_size=(3, 3), strides=(2, 2), padding="valid", idx=5)

    # 6 --> 11
    for i in range(2):
        idx = 6 + 3 * i
        x = darknet_residual_block(x, filters=64, idx=idx)

    # 12
    x = darknet_conv(x, filters=256, kernel_size=(3, 3), strides=(2, 2), padding="valid", idx=12)
    # 13 --> 36
    for i in range(8):
        idx = 13 + 3 * i
        x = darknet_residual_block(x, filters=128, idx=idx)

    skip_36 = x

    # 37
    x = darknet_conv(x, filters=512, kernel_size=(3, 3), strides=(2, 2), padding="valid", idx=37)

    # 38 --> 61
    for i in range(8):
        idx = 38 + 3 * i
        x = darknet_residual_block(x, filters=256, idx=idx)

    skip_61 = x
    # 62
    x = darknet_conv(x, filters=1024, kernel_size=(3, 3), strides=(2, 2), padding="valid", idx=62)

    # 63 - 74
    for i in range(4):
        idx = 63 + 3 * i
        x = darknet_residual_block(x, filters=512, idx=idx)

    # 75 --> 79
    x = convolutional_set(x, filters=512, idx=75)
    # 80-81
    yolo_82 = yolo_end(x, filters=1024, idx=80)

    x = darknet_conv(x, filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same", idx=84)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, skip_61])

    x = convolutional_set(x, filters=256, idx=87)
    yolo_94 = yolo_end(x, filters=512, idx=92)

    x = darknet_conv(x, filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same", idx=96)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, skip_36])

    x = convolutional_set(x, filters=128, idx=99)
    yolo_106 = yolo_end(x, filters=256, idx=104)


    return Model(inputs, [yolo_82, yolo_94, yolo_106])


class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major, = struct.unpack('i', w_f.read(4))
            minor, = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))
            if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
                w_f.read(8)
            else:
                w_f.read(4)
            transpose = (major > 1000) or (minor > 1000)
            binary = w_f.read()
        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def load_weights(self, model):
        for i in range(106):
            try:
                conv_layer = model.get_layer('conv_' + str(i))
                # print("loading weights of convolution #" + str(i))
                if i not in [81, 93, 105]:
                    norm_layer = model.get_layer('batch_norm' + str(i))
                    size = np.prod(norm_layer.get_weights()[0].shape)
                    beta = self.read_bytes(size)  # bias
                    gamma = self.read_bytes(size)  # scale
                    mean = self.read_bytes(size)  # mean
                    var = self.read_bytes(size)  # variance
                    weights = norm_layer.set_weights([gamma, beta, mean, var])
                if len(conv_layer.get_weights()) > 1:
                    bias = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2, 3, 1, 0])
                    conv_layer.set_weights([kernel, bias])
                else:
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2, 3, 1, 0])
                    conv_layer.set_weights([kernel])
            except ValueError:
                # print("no convolution #" + str(i))
                pass

    def reset(self):
        self.offset = 0


def get_model(file_name=None):
    if file_name:
        return load_model(file_name)

    model = yoloV3(name="YOLOv3")
    weight_reader = WeightReader("yolov3.weights")
    weight_reader.load_weights(model)
    return model

