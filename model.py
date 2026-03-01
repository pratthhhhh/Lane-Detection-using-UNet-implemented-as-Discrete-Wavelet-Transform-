from imports import *


#  Custom Haar Wavelet Layer (downsampling) 
class HaarWaveletLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(HaarWaveletLayer, self).__init__(**kwargs)

    def call(self, inputs):
        height = tf.shape(inputs)[1]
        width  = tf.shape(inputs)[2]

        # Ensure height and width are divisible by 2
        height = height // 2 * 2
        width  = width  // 2 * 2

        # 2-D Haar wavelet decomposition
        LL = (inputs[:, :height:2, :width:2, :] + inputs[:, 1:height:2, :width:2, :]) / 2
        LH = (inputs[:, :height:2, :width:2, :] - inputs[:, 1:height:2, :width:2, :]) / 2
        HL = (inputs[:, :height:2, :width:2, :] + inputs[:, :height:2, 1:width:2, :]) / 2
        HH = (inputs[:, :height:2, :width:2, :] - inputs[:, :height:2, 1:width:2, :]) / 2

        return LL, LH, HL, HH


#  Custom Inverse Haar Wavelet Layer (upsampling) 
class InverseHaarWaveletLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(InverseHaarWaveletLayer, self).__init__(**kwargs)

    def call(self, inputs):
        LL, LH, HL, HH = tf.split(inputs, 4, axis=-1)

        x1 = (LL + LH) / 2
        x2 = (LL - LH) / 2
        x3 = (HL + HH) / 2
        x4 = (HL - HH) / 2

        # Reconstruct along the spatial axis
        reconstructed = tf.concat([x1, x2, x3, x4], axis=1)
        return reconstructed


#  Convolution block ─
def conv_block(x, filters, kernel_size=30, activation='relu'):
    x = layers.Conv2D(filters, kernel_size, padding='same', activation=activation)(x)
    return x


#  U-Net with Haar DWT 
def unet(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Encoder
    LL1, LH1, HL1, HH1 = HaarWaveletLayer()(x);    x11 = conv_block(LL1, 1)
    LL2, LH2, HL2, HH2 = HaarWaveletLayer()(x11);  x21 = conv_block(LL2, 1)
    LL3, LH3, HL3, HH3 = HaarWaveletLayer()(x21);  x31 = conv_block(LL3, 1)
    LL4, LH4, HL4, HH4 = HaarWaveletLayer()(x31);  x41 = conv_block(LL4, 1)

    # Bottleneck
    bottleneck = conv_block(x41, 1)
    bottleneck = conv_block(bottleneck, 1)

    # Decoder
    concat14 = tf.concat([bottleneck, LH4, HL4, HH4], axis=-1)
    x14 = InverseHaarWaveletLayer()(concat14)
    x14 = tf.image.resize(x14, size=(80, 80),   method=tf.image.ResizeMethod.BILINEAR)
    x14 = conv_block(x14, 1)

    concat13 = tf.concat([x14, LH3, HL3, HH3], axis=-1)
    x13 = InverseHaarWaveletLayer()(concat13)
    x13 = tf.image.resize(x13, size=(160, 160), method=tf.image.ResizeMethod.BILINEAR)
    x13 = conv_block(x13, 1)

    concat12 = tf.concat([x13, LH2, HL2, HH2], axis=-1)
    x12 = InverseHaarWaveletLayer()(concat12)
    x12 = tf.image.resize(x12, size=(320, 320), method=tf.image.ResizeMethod.BILINEAR)
    x12 = conv_block(x12, 1)

    concat11 = tf.concat([x12, LH1, HL1, HH1], axis=-1)
    x1_ = InverseHaarWaveletLayer()(concat11)
    x1_ = tf.image.resize(x1_, size=(640, 640), method=tf.image.ResizeMethod.BILINEAR)
    x1_ = conv_block(x1_, 1)

    output = layers.Conv2D(1, (1, 1), activation='linear')(x1_)

    model = Model(inputs, output)
    return model


if __name__ == "__main__":
    model = unet((640, 640, 1))
    model.summary()
