# !pip uninstall -y tensorflow tensorflow-addons keras
# !pip install tensorflow==2.15 tensorflow-addons==0.21 keras==2.15
# !pip install nibabel
# !pip install opencv-python
# !pip install matplotlib
# !pip install scikit-image
# !pip install scikit-learn
# !pip install pandas
# !pip install seaborn
# !pip install streamlit
import streamlit as st

import numpy as np
import tensorflow as tf
import os
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import math
from scipy import ndimage
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, Conv3D
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate, UpSampling3D, Activation, BatchNormalization
from tensorflow.keras.layers import SpatialDropout3D, MaxPooling3D, Conv3DTranspose
from tensorflow_addons.layers import InstanceNormalization
from skimage.transform import resize

# Classification model imports
def load_images_2d(im, IM_SIZE=224):
    if type(im)!=np.ndarray:
        im = cv2.imread(im) # read the image from the path

    new_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    new_im = cv2.resize(new_im, (IM_SIZE, IM_SIZE))
    new_im = np.array(new_im) / 255.0
    new_im = new_im[np.newaxis, ...]
    return new_im

def get_classification_model(IM_SIZE=224, N_CLASSES=2, INIT_LR=1e-3, MODEL_WEIGHTS="weights/ResNet50_model.hdf5", NETWORK=ResNet50V2):
    # load the base model (ResNet50V2 network) without the head FC layer
    baseModel = NETWORK(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(IM_SIZE, IM_SIZE, 3)))

    # customize the top layers (transfer learning)
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(3, 3))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(N_CLASSES, activation="softmax")(headModel)

    # build the classification model
    model = Model(inputs=baseModel.input, outputs=headModel)

    for layer in baseModel.layers:
        layer.trainable = False

    # compile the classification model
    opt = Adam(learning_rate= INIT_LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    if MODEL_WEIGHTS != None:
        model.load_weights(MODEL_WEIGHTS)

    return model

def get_last_layer(model):
    last_layer = model.layers[-1]
    return last_layer

def get_last_conv_layer(model, DIM="2d"):
    if DIM == "2d":
        final_conv = list(filter(lambda x: isinstance(x, Conv2D),
                                   model.layers))[-1]
    elif DIM == "3d":
        final_conv = list(filter(lambda x: isinstance(x, Conv3D),
                                   model.layers))[-1]
    return final_conv

def get_xai_classification_model(c_model, layer_n = None):

    if layer_n == None:
        xai_layer = get_last_conv_layer(c_model)
    else:
        xai_layer = c_model.get_layer(layer_n)

    model = tf.keras.models.Model([c_model.inputs], [xai_layer.output, c_model.output])

    return model
def crop_image_brats(img, OUT_SHAPE=(192, 224, 160)):
    # manual cropping
    input_shape = np.array(img.shape)
    # center the cropped image
    offset = np.array((input_shape - OUT_SHAPE) / 2).astype(int)
    offset[offset < 0] = 0
    x, y, z = offset
    crop_img = img[x:x + OUT_SHAPE[0], y:y + OUT_SHAPE[1], z:z + OUT_SHAPE[2]]
    # pad the preprocessed image
    padded_img = np.zeros(OUT_SHAPE)
    x, y, z = np.array((OUT_SHAPE - np.array(crop_img.shape)) / 2).astype(int)
    padded_img[x:x + crop_img.shape[0], y:y + crop_img.shape[1], z:z + crop_img.shape[2]] = crop_img
    return padded_img

def postprocess_tumor(seg_data, OUT_SHAPE = (240, 240, 155), POST_ENHANC=False, THRES=200):
    # post-process the enhancing tumor region
    if POST_ENHANC:
        seg_enhancing = (seg_data == 4)
        if np.sum(seg_enhancing) < THRES:
            if np.sum(seg_enhancing) > 0:
                seg_data[seg_enhancing] = 1
                print("\tConverted {} voxels from label 4 to label 1!".format(np.sum(seg_enhancing)))

    input_shape = np.array(seg_data.shape)
    OUT_SHAPE = np.array(OUT_SHAPE)
    offset = np.array((OUT_SHAPE - input_shape)/2).astype(int)
    offset[offset<0] = 0
    x, y, z = offset

    # pad the preprocessed image
    padded_seg = np.zeros(OUT_SHAPE).astype(np.uint8)
    padded_seg[x:x+seg_data.shape[0],y:y+seg_data.shape[1],z:z+seg_data.shape[2]] = seg_data[:,:,2:padded_seg.shape[2]+2]

    return padded_seg #.astype(np.uint8)

def load_images(model, ID, FILE, PATH_DATA='./', DIM=(192, 224, 160), VALID_SET=True, POST_ENHANC=False):
    img1 = os.path.join(PATH_DATA, FILE, ID+'_FLAIR.nii.gz')
    img2 = os.path.join(PATH_DATA, FILE, ID+'_T1.nii.gz')
    img3 = os.path.join(PATH_DATA, FILE, ID+'_T1c.nii.gz')
    img4 = os.path.join(PATH_DATA, FILE, ID+'_T2.nii.gz')

    # combine the four imaging modalities (flair, t1, t1ce, t2)
    imgs_input = nib.concat_images([img1, img2, img3, img4]).get_fdata()

    imgs_preprocess = np.zeros((DIM[0],DIM[1],DIM[2],4)) # (5, 192, 224, 160)
    if VALID_SET:
        for i in range(imgs_preprocess.shape[-1]):
            imgs_preprocess[:, :, :, i] = crop_image_brats(imgs_input[:, :, :, i])
            imgs_preprocess[:, :, :, i] = norm_image(imgs_preprocess[:, :, :, i])

    return imgs_preprocess[np.newaxis, ...]

def create_convolution_block(input_layer, n_filters, BN=True, KERNEL=(3, 3, 3), ACTIV=None,
                             PAD='same', STR=(1, 1, 1), IN=False):
    layer = Conv3D(n_filters, KERNEL, padding=PAD, strides=STR)(input_layer)
    if BN:
        layer = BatchNormalization(axis=-1)(layer)
    elif IN:
        layer = InstanceNormalization(axis=-1)(layer)
    if ACTIV is None:
        return Activation('relu')(layer)
    else:
        return ACTIV()(layer)

def get_up_convolution(n_filters, pool_size, KERNEL=(3, 3, 3), STR=(2, 2, 2),
                       DECONV=False):
    if DECONV:
        return Conv3DTranspose(filters=n_filters, kernal=KERNEL,
                               strides=STR)
    else:
        return UpSampling3D(size=pool_size)

def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters)
    return convolution2

def create_context_module(input_layer, n_level_filters, DROPOUT=0.3):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=DROPOUT)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2

def create_up_sampling_module(input_layer, n_filters, SIZE=(2, 2, 2)):
    convolution1 = create_convolution_block(input_layer, n_filters, KERNEL=(2, 2, 2))
    up_sample = UpSampling3D(size=SIZE)(convolution1)
    return up_sample


def get_deepseg(INP_SHAPE=(192, 224, 160, 4), N_FILTERS=8, DEPTH=5, DROPOUT=0.5,
                      N_SEG=3, N_LABELS=4, OPT=Adam, INIT_LR=1e-4,
                      LOSS="mse", ACTIV="softmax", WEIGHTS=None):
    inputs = Input(INP_SHAPE)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(DEPTH):
        n_level_filters = (2**level_number) * N_FILTERS
        level_filters.append(n_level_filters)

        in_conv = create_convolution_block(current_layer, n_level_filters)
        context_output_layer = create_convolution_block(in_conv, n_level_filters)

        level_output_layers.append(current_layer)
        current_layer = MaxPooling3D(pool_size=(2, 2, 2))(context_output_layer)

    current_layer = SpatialDropout3D(rate=DROPOUT)(current_layer)

    for level_number in reversed(range(DEPTH)):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=-1)

        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output

    output_layer = Conv3D(N_LABELS, (1, 1, 1), name="output_layer")(current_layer)
    activ_block = Activation(ACTIV, name="output_layer_soft")(output_layer)
    model = Model(inputs=inputs, outputs=activ_block)

    model.compile(optimizer=Adam(lr=INIT_LR), loss=LOSS, metrics=["accuracy"])

    if WEIGHTS:
        model.load_weights(WEIGHTS)
    return model

def get_last_layer(model):
    last_layer = model.layers[-1]
    return last_layer

def get_last_conv_layer(model, DIM="3d"):
    if DIM == "2d":
        final_conv = list(filter(lambda x: isinstance(x, Conv2D),
                                   model.layers))[-1]
    elif DIM == "3d":
        final_conv = list(filter(lambda x: isinstance(x, Conv3D),  # Changed Conv2D to Conv3D
                                   model.layers))[-1]
    return final_conv


def get_xai_segmentation_model(s_model, layer_n = None):
    if layer_n == None:
        xai_layer = get_last_conv_layer(s_model)
    else:
        xai_layer = s_model.get_layer(layer_n)

    model = tf.keras.models.Model([s_model.inputs], [xai_layer.output, s_model.output])

    return model
# Grad-CAM
 # for N-dimensional images

def get_grad_cam(model, io_imgs, class_id, LAYER_NAME=None, MODALITY="FLAIR", XAI_MODE="classification", DIMENSION="2d", eps=1e-5):
    modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}
    if LAYER_NAME==None:
        LAYER_NAME = get_last_conv_layer(model).name
    # Sanity check
    layer = model.get_layer(LAYER_NAME)
    assert (isinstance(layer, Conv2D) or isinstance(layer, Conv3D)), "Input layer must be convolutional layer for Grad-CAM"

    io_imgs = tf.convert_to_tensor(io_imgs)
    with tf.GradientTape() as tape:
        tape.watch(io_imgs)
        conv_outputs, predictions = model(io_imgs)
        if XAI_MODE == "classification":
            loss = predictions[:,class_id]
        elif XAI_MODE == "segmentation":
            loss = predictions[:,:,:,:,class_id]

        # Compute gradients with automatic differentiation
        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]

        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))
        if DIMENSION=="2d":
            weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        elif DIMENSION=="3d":
            weights = tf.reduce_mean(norm_grads, axis=(0, 1, 2))

    # Average gradients spatially
    # Build a ponderated map of filters according to gradients importance
    cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)

    # Apply ReLU
    grad_cam = np.maximum(cam, 0)
    # Resize heatmap to be the same size as the input
    if np.max(grad_cam) > 0:
        grad_cam = grad_cam / np.max(grad_cam)

    # Resize to the output layer's shape
    new_shape = io_imgs.shape[1:len(io_imgs.shape)]
    grad_cam = resize(grad_cam, new_shape)
    return grad_cam
# Gudied BackPropagartion
@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad

def get_guided_backprop(model, io_imgs, class_id, LAYER_NAME=None, MODALITY="FLAIR", XAI_MODE="classification", DIMENSION="2d"):
    modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}

    clone_model = tf.keras.models.clone_model(model)
    clone_model.set_weights(model.get_weights())

    if LAYER_NAME==None:
        LAYER_NAME = get_last_layer(clone_model).name
    # Build the GBP model
    gbp_layer = clone_model.get_layer(LAYER_NAME)
    guided_model = tf.keras.models.Model([clone_model.inputs], [gbp_layer.output])
    layer_dict = [layer for layer in guided_model.layers[1:] if hasattr(layer,"activation")]
    for layer in layer_dict:
        if layer.activation == tf.keras.activations.relu:
            layer.activation = guided_relu

    io_imgs = tf.convert_to_tensor(io_imgs)
    with tf.GradientTape() as tape:
        tape.watch(io_imgs)

        if XAI_MODE == "classification":
            output = guided_model(io_imgs)
            output = output[:,class_id]
        elif XAI_MODE == "segmentation":
            output = guided_model(io_imgs)
            output = output[:,:,:,:,class_id]

        # Extract filters and gradients
        guided_backprop = tape.gradient(output, io_imgs)[0] # whole brain

    # Resize to the output layer's shape
    new_shape = io_imgs.shape[1:len(io_imgs.shape)]
    guided_backprop = resize(np.asarray(guided_backprop), new_shape) # convert to 3D instead of 4D

    return guided_backprop
# Guided Grad-CAM
def get_guided_grad_cam(model, io_imgs, class_id, LAYER_NAME=None, MODALITY="FLAIR", XAI_MODE="classification",
                        DIMENSION="2d", CLASS_IDs=[0,1], TUMOR_LABEL="all"):
    if LAYER_NAME==None:
        LAYER_NAME = get_last_conv_layer(model).name
    # Sanity check
    layer = model.get_layer(LAYER_NAME)
    assert (isinstance(layer, Conv2D) or isinstance(layer, Conv3D)), "Input layer must be convolutional layer for Grad-CAM"


    guided_backprop = get_guided_backprop(model, io_imgs, class_id, LAYER_NAME=LAYER_NAME, MODALITY=MODALITY, XAI_MODE=XAI_MODE)
    if XAI_MODE=="segmentation" and TUMOR_LABEL=="all":
        grad_cam = np.zeros_like(guided_backprop)
        for c_id in CLASS_IDs:
            grad_cam += get_grad_cam(model, io_imgs, c_id, LAYER_NAME=LAYER_NAME, MODALITY=MODALITY, XAI_MODE=XAI_MODE, DIMENSION=DIMENSION)
    else:
        grad_cam = get_grad_cam(model, io_imgs, class_id, LAYER_NAME=LAYER_NAME, MODALITY=MODALITY, XAI_MODE=XAI_MODE, DIMENSION=DIMENSION)

    return grad_cam*guided_backprop
# Guided Integrated Gradient
# A very small number for comparing floating point values.
EPSILON = 1e-9

def l1_distance(x1, x2):
    """Returns L1 distance between two points."""
    return np.abs(x1 - x2).sum()

def translate_x_to_alpha(x, x_input, x_baseline):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x_input - x_baseline != 0,
                                        (x - x_baseline) / (x_input - x_baseline), np.nan)

def translate_alpha_to_x(alpha, x_input, x_baseline):
    assert 0 <= alpha <= 1.0
    return x_baseline + (x_input - x_baseline) * alpha

def get_guided_integrated_grads(model, io_imgs, class_id, LAYER_NAME=None, MODALITY="FLAIR", XAI_MODE="classification",
                                                                        STEPS=5, FRAC=0.5, MAX_DIST=1.0):
#                                                                        STEPS=200, FRAC=0.25, MAX_DIST=0.02):
    modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}

    baseline = np.zeros_like(io_imgs)
    x_input = np.asarray(io_imgs, dtype=np.float64)
    x_baseline = np.asarray(baseline, dtype=np.float64)
    x = x_baseline.copy()
    l1_total = l1_distance(x_input, x_baseline)
    attr = np.zeros_like(x_input, dtype=np.float64)

    # If the input is equal to the baseline then the attribution is zero.
    total_diff = x_input - x_baseline
    if np.abs(total_diff).sum() == 0:
        return attr

    # Iterate through every step.
    for step in range(STEPS):
        #print("Step:({}/{})".format(step+1, STEPS))
        # Calculate gradients and make a copy.
        grad_actual = compute_grads(model, x, class_id, LAYER_NAME, MODALITY, XAI_MODE)
        grad = grad_actual.copy()
        # Calculate current step alpha and the ranges of allowed values for this
        # step.
        alpha = (step + 1.0) / STEPS
        alpha_min = max(alpha - MAX_DIST, 0.0)
        alpha_max = min(alpha + MAX_DIST, 1.0)
        x_min = translate_alpha_to_x(alpha_min, x_input, x_baseline)
        x_max = translate_alpha_to_x(alpha_max, x_input, x_baseline)
        # The goal of every step is to reduce L1 distance to the input.
        # `l1_target` is the desired L1 distance after completion of this step.
        l1_target = l1_total * (1 - (step + 1) / STEPS)

        # Iterate until the desired L1 distance has been reached.
        gamma = np.inf
        while gamma > 1.0:
            x_old = x.copy()
            x_alpha = translate_x_to_alpha(x, x_input, x_baseline)
            x_alpha[np.isnan(x_alpha)] = alpha_max
            # All features that fell behind the [alpha_min, alpha_max] interval in
            # terms of alpha, should be assigned the x_min values.
            x[x_alpha < alpha_min] = x_min[x_alpha < alpha_min]

            # Calculate current L1 distance from the input.
            l1_current = l1_distance(x, x_input)
            # If the current L1 distance is close enough to the desired one then
            # update the attribution and proceed to the next step.
            if math.isclose(l1_target, l1_current, rel_tol=EPSILON, abs_tol=EPSILON):
                attr += (x - x_old) * grad_actual
                break

            # Features that reached `x_max` should not be included in the selection.
            # Assign very high gradients to them so they are excluded.
            grad[x == x_max] = np.inf

            # Select features with the lowest absolute gradient.
            threshold = np.quantile(np.abs(grad), FRAC, interpolation='lower')
            s = np.logical_and(np.abs(grad) <= threshold, grad != np.inf)

            # Find by how much the L1 distance can be reduced by changing only the
            # selected features.
            l1_s = (np.abs(x - x_max) * s).sum()

            # Calculate ratio `gamma` that show how much the selected features should
            # be changed toward `x_max` to close the gap between current L1 and target
            # L1.
            if l1_s > 0:
                gamma = (l1_current - l1_target) / l1_s
            else:
                gamma = np.inf

            if gamma > 1.0:
                # Gamma higher than 1.0 means that changing selected features is not
                # enough to close the gap. Therefore change them as much as possible to
                # stay in the valid range.
                x[s] = x_max[s]
            else:
                assert gamma > 0, gamma
                x[s] = translate_alpha_to_x(gamma, x_max, x)[s]
            # Update attribution to reflect changes in `x`.
            attr += (x - x_old) * grad_actual
    return attr[0]
# Integrated Gradient
def compute_grads(model, io_imgs, class_id, LAYER_NAME=None, MODALITY="FLAIR", XAI_MODE="classification"):
    modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}

    io_imgs = tf.convert_to_tensor(io_imgs)
    with tf.GradientTape() as tape:
        tape.watch(io_imgs)

        last_layer = get_last_layer(model)
        if LAYER_NAME == None or LAYER_NAME == last_layer.name:
            _, predictions = model(io_imgs)
            if XAI_MODE == "classification":
                predictions = predictions[:,class_id]
            elif XAI_MODE == "segmentation":
                predictions = predictions[:,:,:,:,class_id]

            gradients = np.array(tape.gradient(predictions, io_imgs))
        else:
            conv_outputs, predictions = model(io_imgs)
            gradients = np.array(tape.gradient(conv_outputs, io_imgs))

    return gradients

def get_integrated_grads(model, io_imgs, class_id, LAYER_NAME=None, MODALITY="FLAIR", XAI_MODE="classification", M_STEPS=25, BS=1):
    modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}

    baseline = np.zeros_like(io_imgs)
    diff = io_imgs - baseline
    total_gradients = np.zeros_like(io_imgs, dtype=np.float32)

    m_step_batched = []
    for alpha in np.linspace(0, 1, M_STEPS):
        m_step = baseline + alpha * diff
        m_step_batched.append(m_step)
        if len(m_step_batched) == BS or alpha == 1:
            m_step_batched = np.asarray(m_step_batched)
            gradients = compute_grads(model, io_imgs, class_id, LAYER_NAME, MODALITY, XAI_MODE)

            total_gradients += gradients.sum(axis=0)
            m_step_batched = []

    integrated_grads = total_gradients * diff / M_STEPS

    return integrated_grads[0]
# SmoothGrad
def get_smoothgrad(model, io_imgs, class_id, LAYER_NAME=None, MODALITY="FLAIR", XAI_MODE="classification",
                   XAI="GBP", DIMENSION="2d", STDEV_SPREAD=.15, N_SAMPLES=5, MAGNITUDE=True):
                   #XAI="GBP", DIMENSION="2d", STDEV_SPREAD=.15, N_SAMPLES=25, MAGNITUDE=True):

    new_shape = io_imgs.shape[1:len(io_imgs.shape)]

    if XAI_MODE == "segmentation" and XAI=="GBP":
        new_shape = io_imgs.shape[1:len(io_imgs.shape)] +(3,)

    total_gradients = np.zeros(new_shape, dtype=np.float32)
    #print("Shape of total_gradients:", total_gradients.shape)
    stdev = STDEV_SPREAD * (np.max(io_imgs) - np.min(io_imgs))
    for _ in range(N_SAMPLES):
        noise = np.random.normal(0, stdev, io_imgs.shape)
        x_plus_noise = io_imgs + noise
        if XAI=="VANILLA":
            grads = get_vanilla_grad(model, x_plus_noise, class_id, LAYER_NAME, MODALITY, XAI_MODE)
        elif XAI=="GBP":
            grads = get_guided_backprop(model, x_plus_noise, class_id, LAYER_NAME, MODALITY, XAI_MODE)
        elif XAI=="IG":
            grads = get_integrated_grads(model, x_plus_noise, class_id, LAYER_NAME, MODALITY, XAI_MODE)
        elif XAI=="GIG":
            grads = get_guided_integrated_grads(model, x_plus_noise, class_id, LAYER_NAME, MODALITY, XAI_MODE)
        elif XAI=="GCAM":
            grads = get_grad_cam(model, x_plus_noise, class_id, LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)
        elif XAI=="GGCAM":
            grads = get_guided_grad_cam(model, x_plus_noise, class_id, LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)

        if MAGNITUDE:
            total_gradients += (grads * grads)
        else:
            total_gradients += grads

    return total_gradients / N_SAMPLES
# Vanilla Gradient
def get_vanilla_grad(model, io_imgs, class_id, LAYER_NAME=None, MODALITY="FLAIR", XAI_MODE="classification", DIMENSION="2d"):
    modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}

    io_imgs = tf.convert_to_tensor(io_imgs)
    with tf.GradientTape() as tape:
        tape.watch(io_imgs)

        last_layer = get_last_layer(model)
        _, predictions = model(io_imgs)
        if LAYER_NAME == None or LAYER_NAME == last_layer.name:
            if XAI_MODE == "classification":
                predictions = predictions[:,class_id]
            elif XAI_MODE == "segmentation":
                predictions = predictions[:,:,:,:,class_id]

        # Extract filters and gradients
        vanilla_grad = tape.gradient(predictions, io_imgs)[0]
        # Resize to the output layer's shape
        new_shape = io_imgs.shape[1:len(io_imgs.shape)]
        vanilla_grad = resize(np.asarray(vanilla_grad), new_shape) # convert to 3D instead of 4D

    return vanilla_grad
# NeuroXAI parameters
XAI_MODES = ["segmentation","classification"]
XAIs = ["VANILLA","GBP","IG","GIG","GCAM","GGCAM","SMOOTH"]
MODALITIES = ["FLAIR","T1","T1CE","T2"]
DIMENSIONS = ["2d","3d"]
TUMOR_LABEL = "all" # for GCAM visualization

# segmentation mode
#DIMENSION = "3d"
#MODALITY = "FLAIR"
#XAI_MODE = "segmentation"
#CLASS_IDs = [1,2,3]

# classification mode
DIMENSION = "2d"
MODALITY = "FLAIR"
XAI_MODE = "classification"
CLASS_IDs = [0,1]

# GPU handling (TF 2.X)
import os
import tensorflow as tf

assert float(tf.__version__[:3]) >= 2.0, "NeuroXAI requires Tensorflow version 2.0 or higher"

gpu_ids = '0' # '0,1' # for multi-gpu environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs\n")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
# Visualization and save of all NeuroXAI methods
def get_neuroxai(ID, model, io_imgs, CLASS_ID=0, SLICE_ID=77, LAYER_NAME=None,
                 MODALITY="FLAIR", XAI_MODE="classification",
                 DIMENSION="2d", CLASS_IDs=[0,1], TUMOR_LABEL="all", SAVE_RESULTS=False, SAVE_PATH=None):

    # sanity checks
    assert DIMENSION in DIMENSIONS, "Input dimension must be in {}".format([d for d in DIMENSIONS])
    assert XAI_MODE in XAI_MODES, "XAI mode must be in {}".format([m for m in XAI_MODES])
    assert MODALITY in MODALITIES, "MRI modality must be in {}".format([m for m in MODALITIES])

    modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}

    if LAYER_NAME==None:
        LAYER_NAME = get_last_layer(model).name
        CONV_LAYER_NAME = get_last_conv_layer(model, DIMENSION).name

    else:
        # A solution for classification visualization of all XAI
        layer = model.get_layer(LAYER_NAME)
        last_layer = get_last_layer(model)
        if layer.name == last_layer.name:
            if (not isinstance(layer, Conv2D)) or (not isinstance(layer, Conv3D)):
                CONV_LAYER_NAME = get_last_conv_layer(model, DIMENSION).name
            else:
                CONV_LAYER_NAME = LAYER_NAME
        else:
            CONV_LAYER_NAME = LAYER_NAME

    print("Visual exaplanations for...\n\t ID: {}, layer: {}".format(ID, LAYER_NAME))
    # Build the model
    if XAI_MODE=="classification": # 2d and 3d
        im_orig = io_imgs[0]
        SmoothXAI="IG"
    elif XAI_MODE=="segmentation": # 3d only
        im_orig = io_imgs[0,:,:,SLICE_ID,modality_dict[MODALITY]] # (1, 192, 224, 160, 4)
        SmoothXAI="GIG"

    # Get XAI heat maps (1 samples for SmoothGrad)
    #vanilla_grads = get_vanilla_grad(model, io_imgs, CLASS_ID, LAYER_NAME=LAYER_NAME, MODALITY=MODALITY, XAI_MODE=XAI_MODE)
    vanilla_grads = compute_grads(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE)[0]

    gbp_grads = get_guided_backprop(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE)
    ig_grads = get_integrated_grads(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE)
    gig_grads = get_guided_integrated_grads(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE)

    if XAI_MODE=="segmentation" and TUMOR_LABEL=="all":
        gcam_grads = get_grad_cam(model, io_imgs, CLASS_IDs[0], CONV_LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)
        for c_id in CLASS_IDs[1:]:
            gcam_grads += get_grad_cam(model, io_imgs, c_id, CONV_LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)
    else:
        gcam_grads = get_grad_cam(model, io_imgs, CLASS_ID, CONV_LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)

    ggcam_grads = get_guided_grad_cam(model, io_imgs, CLASS_ID, CONV_LAYER_NAME, MODALITY, XAI_MODE, DIMENSION, CLASS_IDs, TUMOR_LABEL)
    smooth_grads = get_smoothgrad(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE, XAI=SmoothXAI)

    # Get 2D images for visualizations
    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    guided_backprop = visualize_tensor(gbp_grads)
    vanilla_gradient = visualize_tensor(vanilla_grads)
    integrated_gradient = visualize_tensor(ig_grads)
    guided_integrated_gradient = visualize_tensor(gig_grads)
    gradcam = visualize_tensor(gcam_grads)
    guided_gradcam = visualize_tensor(ggcam_grads)
    smooth_grad = visualize_tensor(smooth_grads)

    if DIMENSION=="3d":
        guided_backprop = guided_backprop[:,:,SLICE_ID]
        vanilla_gradient = vanilla_gradient[:,:,SLICE_ID]
        integrated_gradient = integrated_gradient[:,:,SLICE_ID]
        guided_integrated_gradient = guided_integrated_gradient[:,:,SLICE_ID]
        gradcam = gradcam[:,:,SLICE_ID]
        guided_gradcam = guided_gradcam[:,:,SLICE_ID]
        smooth_grad = smooth_grad[:,:,SLICE_ID]

    # Get overlay images (Over FLAIR MRI)
    gradcam_overlay = overlay_gradcam(im_orig, gradcam, DIMENSION)
    guided_gradcam_overlay = overlay_grad(im_orig, guided_gradcam, DIMENSION)

    # Set up matplot lib figures.
    ROWS = 1
    COLS = 11 if XAI_MODE=="segmentation" else 9
    UPSCALE_FACTOR = 25
    plt.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

    # Render the saliency masks.
    show_gray_image(im_orig, TITLE=MODALITY, AX=plt.subplot(ROWS, COLS, 1))
    show_gray_image(vanilla_gradient, TITLE='Vanilla', AX=plt.subplot(ROWS, COLS, 2))
    show_gray_image(guided_backprop, TITLE='Backprop', AX=plt.subplot(ROWS, COLS, 3))
    show_gray_image(integrated_gradient, TITLE='IG', AX=plt.subplot(ROWS, COLS, 4))
    show_gray_image(guided_integrated_gradient, TITLE='Guided IG', AX=plt.subplot(ROWS, COLS, 5))
    show_gray_image(smooth_grad, TITLE='SmoothGrad', AX=plt.subplot(ROWS, COLS, 6))
    show_heatmap(gradcam, TITLE='Grad-CAM', AX=plt.subplot(ROWS, COLS, 7), cmap="jet")
    show_heatmap(gradcam_overlay, TITLE='Overlay Grad-CAM', AX=plt.subplot(ROWS, COLS, 8))
    show_heatmap(guided_gradcam, TITLE='Guided_Grad-CAM', AX=plt.subplot(ROWS, COLS, 9), cmap="jet")

    if XAI_MODE=="segmentation":
        # Predict the tumor segmentation
        _, prediction_3ch = np.squeeze(model(io_imgs)) # model prediction
        prediction = np.argmax(prediction_3ch, axis=-1)
        prediction = (prediction[:,:,SLICE_ID] == CLASS_ID).astype(np.uint8)
        if CLASS_ID!= 0:
            prediction[prediction>0] = CLASS_ID

        pred_overlay = overlay_pred(io_imgs[0,:,:,SLICE_ID,modality_dict[MODALITY]], prediction)
        show_image(prediction, TITLE='Prediction', AX=plt.subplot(ROWS, COLS, 10))
        show_image(pred_overlay, TITLE='Prediction Overlay', AX=plt.subplot(ROWS, COLS, 11))

    # Save the results
    if SAVE_RESULTS:
        if not os.path.exists(os.path.join(SAVE_PATH, ID)):
            os.makedirs(os.path.join(SAVE_PATH, ID))
        # Save 2D heatmaps
        if XAI_MODE=="classification":
            LAYER_NAME=""
        if TUMOR_LABEL=="all":
            CLASS_ID="all"
        plt.imsave("{}/{}/{}_{}_{}_vanilla.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), vanilla_gradient, CMAP=plt.cm.gray)
        plt.imsave("{}/{}/{}_{}_{}_ig.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), integrated_gradient, CMAP=plt.cm.gray)
        plt.imsave("{}/{}/{}_{}_{}_gig.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), guided_integrated_gradient, CMAP=plt.cm.gray)
        plt.imsave("{}/{}/{}_{}_{}_gbp.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), guided_backprop, CMAP=plt.cm.gray)
        plt.imsave("{}/{}/{}_{}_{}_gcam.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), gradcam, CMAP="jet")
        plt.imsave("{}/{}/{}_{}_{}_ggcam.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), guided_gradcam, CMAP="jet")
        plt.imsave("{}/{}/{}_{}_{}_gcam_overlay.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), gradcam_overlay)
        plt.imsave("{}/{}/{}_{}_{}_MRI.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), im_orig, CMAP=plt.cm.gray)
        plt.imsave("{}/{}/{}_{}_{}_smooth.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), smooth_grad, CMAP=plt.cm.gray)

        if XAI_MODE=="segmentation":
            plt.imsave("{}/{}/{}_{}_{}_pred.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), prediction)
            #plt.imsave("{}/{}/{}_{}_{}_pred_overlay.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), pred_overlay)
            plt.imsave("{}/{}/{}_{}_{}_truth.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID), pred_data_m[:,:,SLICE_ID])


# Visualization and save of a single NeuroXAI method
def visualize_neuroxai_cnn(ID, model, io_imgs, grads, CLASS_ID=0, SLICE_ID=77,
                           LAYER_NAME=None, MODALITY="FLAIR",
                           XAI_MODE="classification", XAI="GCAM",
                           DIMENSION="2d", TUMOR_LABEL="all",
                           SAVE_RESULTS=False, SAVE_PATH=None):

    modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}

    if XAI_MODE=="classification": # 2d and 3d
        im_orig = io_imgs[0]
        SmoothXAI="IG"
        #smooth_grads = get_smoothgrad(model, io_imgs, CLASS_ID, XAI_MODE=XAI_MODE, XAI="IG")
    elif XAI_MODE=="segmentation": # 3d only
        im_orig = io_imgs[0,:,:,SLICE_ID,modality_dict[MODALITY]] # (1, 192, 224, 160, 4)
        SmoothXAI="GIG"

    # Get 2D images for visualizations
    heatmap = visualize_tensor(grads)
    if DIMENSION=="3d":
        heatmap = heatmap[:,:,SLICE_ID]

    # Set up matplot lib figures.
    ROWS = 1
    COLS = 5 if XAI_MODE=="segmentation" else 3
    UPSCALE_FACTOR = 25 if XAI_MODE=="segmentation" else 10
    plt.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

    # Render the saliency masks.
    show_gray_image(im_orig, TITLE=MODALITY, AX=plt.subplot(ROWS, COLS, 1))
    # Get overlay images (Over MRI)
    if XAI=="GCAM" or XAI=="GGCAM":
        overlay = overlay_gradcam(im_orig, heatmap, DIMENSION)
        if LAYER_NAME==None:
            LAYER_NAME = get_last_conv_layer(model, DIMENSION).name
        show_heatmap(heatmap, TITLE=XAI+'-'+LAYER_NAME+'-'+str(CLASS_ID), AX=plt.subplot(ROWS, COLS, 2), cmap="jet")
    else:
        overlay = overlay_grad(im_orig, heatmap, DIMENSION)
        if LAYER_NAME==None:
            LAYER_NAME = get_last_layer(model).name
        show_gray_image(heatmap, TITLE=XAI+'-'+LAYER_NAME+'-'+str(CLASS_ID), AX=plt.subplot(ROWS, COLS, 2)) # cmap=plt.cm.gray
    show_image(overlay, TITLE=XAI+' Overlay', AX=plt.subplot(ROWS, COLS, 3))

    if XAI_MODE=="segmentation":
        # Predict the tumor segmentation
        _, prediction_3ch = np.squeeze(model(io_imgs)) # model prediction
        prediction = np.argmax(prediction_3ch, axis=-1)
        prediction = (prediction[:,:,SLICE_ID] == CLASS_ID).astype(np.uint8)
        if CLASS_ID!= 0:
            prediction[prediction>0] = CLASS_ID

        pred_overlay = overlay_pred(io_imgs[0,:,:,SLICE_ID,modality_dict[MODALITY]], prediction)
        show_image(prediction, TITLE='Prediction', AX=plt.subplot(ROWS, COLS, 4))
        show_image(pred_overlay, TITLE='Prediction Overlay', AX=plt.subplot(ROWS, COLS, 5))

    # Save the results
    if SAVE_RESULTS:
        if not os.path.exists(os.path.join(SAVE_PATH, ID)):
            os.makedirs(os.path.join(SAVE_PATH, ID))

        if XAI_MODE=="classification":
            LAYER_NAME=""
        if TUMOR_LABEL=="all":
            CLASS_ID="all"

        # Save 2D heatmaps
        if XAI=="GCAM" or XAI=="GGCAM":

            plt.imsave("{}/{}/{}_{}_{}_{}_{}.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID, MODALITY, XAI), heatmap, cmap="jet")
        else:
            plt.imsave("{}/{}/{}_{}_{}_{}_{}.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID, MODALITY, XAI), heatmap, cmap=plt.cm.gray)

        plt.imsave("{}/{}/{}_{}_{}_{}_{}_overlay.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID, MODALITY, XAI), overlay)
        plt.imsave("{}/{}/{}_{}_{}_{}_MRI.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID, MODALITY), im_orig, cmap=plt.cm.gray)

        if XAI_MODE=="segmentation":
            plt.imsave("{}/{}/{}_{}_{}_{}_{}_pred.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID, MODALITY, XAI), prediction)
            plt.imsave("{}/{}/{}_{}_{}_{}_{}_pred_overlay.png".format(SAVE_PATH, ID, ID, LAYER_NAME, CLASS_ID, MODALITY, XAI), pred_overlay)

def get_neuroxai_cnn(ID, model, io_imgs, CLASS_ID=0, SLICE_ID=77, LAYER_NAME=None,
                     MODALITY="FLAIR", XAI_MODE="classification", XAI="GCAM",
                     DIMENSION="2d", CLASS_IDs=[0,1], TUMOR_LABEL="all",
                     SAVE_RESULTS=False, SAVE_PATH=None):

    # sanity checks
    assert DIMENSION in DIMENSIONS, "Input dimension must be in {}".format([d for d in DIMENSIONS])
    assert XAI_MODE in XAI_MODES, "XAI mode must be in {}".format([m for m in XAI_MODES])
    assert MODALITY in MODALITIES, "MRI modality must be in {}".format([m for m in MODALITIES])
    assert XAI in XAIs, "XAI method must be in {}".format([m for m in XAIs])

    if LAYER_NAME==None:
        LAYER_NAME = get_last_layer(model).name
        CONV_LAYER_NAME = get_last_conv_layer(model, DIMENSION).name

    else:
        layer = model.get_layer(LAYER_NAME)
        last_layer = get_last_layer(model)
        if layer.name == last_layer.name:
            if (not isinstance(layer, Conv2D)) or (not isinstance(layer, Conv3D)):
                CONV_LAYER_NAME = get_last_conv_layer(model, DIMENSION).name
            else:
                CONV_LAYER_NAME = LAYER_NAME
        else:
            CONV_LAYER_NAME = LAYER_NAME


    print("Visual exaplanations for...\n\t ID: {}, layer: {}".format(ID, LAYER_NAME))
    # Get the gradients
    if XAI=="VANILLA":
        grads = get_vanilla_grad(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE)
    elif XAI=="GBP":
        grads = get_guided_backprop(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE)
    elif XAI=="IG":
        grads = get_integrated_grads(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE)
    elif XAI=="GIG":
        grads = get_guided_integrated_grads(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE)
    elif XAI=="GCAM":
        LAYER_NAME = CONV_LAYER_NAME
        if XAI_MODE=="segmentation" and TUMOR_LABEL=="all":
            grads = get_grad_cam(model, io_imgs, CLASS_IDs[0], LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)
            for c_id in CLASS_IDs[1:]:
                grads += get_grad_cam(model, io_imgs, c_id, LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)
        else:
            grads = get_grad_cam(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)


    elif XAI=="GGCAM":
        LAYER_NAME = CONV_LAYER_NAME
        grads = get_guided_grad_cam(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE, DIMENSION, CLASS_IDs, TUMOR_LABEL)
    elif XAI=="SMOOTH":
        if XAI_MODE=="classification": # 2d and 3d
            SmoothXAI="IG"
        elif XAI_MODE=="segmentation": # 3d only
            SmoothXAI="GIG"
        grads = get_smoothgrad(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE, SmoothXAI, DIMENSION)

    # Visualize the saliency map
    visualize_neuroxai_cnn(ID, model, io_imgs, grads, CLASS_ID, SLICE_ID, LAYER_NAME, MODALITY,
                           XAI_MODE, XAI, DIMENSION, TUMOR_LABEL, SAVE_RESULTS, SAVE_PATH)
# Utils methods


def deprocess_image(img):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    img = np.array(img).copy()
    img -= img.mean()
    img /= (img.std() + K.epsilon())
    img *= 0.25
    # clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)
    # convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def norm_image(img, NORM_TYP="norm"):
    if NORM_TYP == "standard_norm": # standarization, same dataset
        img_mean = img.mean()
        img_std = img.std()
        img_std = 1 if img.std()==0 else img.std()
        img = (img - img_mean) / img_std
    elif NORM_TYP == "norm": # different datasets
        img = (img - np.min(img))/(np.ptp(img)) # (np.max(img) - np.min(img))
    elif NORM_TYP == "norm_slow": # different datasets
        img_ptp = 1 if np.ptp(img)== 0 else np.ptp(img)
        img = (img - np.min(img))/img_ptp
    return img

def get_last_layer(model):
    last_layer = model.layers[-1]
    return last_layer

def get_last_conv_layer(model, DIM="3d"):
    if DIM == "2d":
        final_conv = list(filter(lambda x: isinstance(x, Conv2D),
                                   model.layers))[-1]
    elif DIM == "3d":
        final_conv = list(filter(lambda x: isinstance(x, Conv3D),
                                   model.layers))[-1]
    return final_conv
# Visualization methods
def show_image(im, TITLE='', AX=None):
    if AX is None:
        plt.figure()
    plt.axis('off')
    plt.imshow(im)
    plt.title(TITLE)

def show_gray_image(im, TITLE='', AX=None):
    if AX is None:
        plt.figure()
    plt.axis('off')
    plt.imshow(im, cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.title(TITLE)

def show_heatmap(im, TITLE='', AX=None, cmap="inferno"):
    if AX is None:
        plt.figure()
    plt.axis('off')
    plt.imshow(im, cmap=cmap)
    plt.title(TITLE)

def visualize_tensor(large_image, PERCENTILE=99):
    r"""Returns a 3D tensor as a grayscale 2D tensor.
    This method sums a 3D tensor across the absolute value of axis=-1, and then
    clips values at a given percentile.
    """
    new_image = np.sum(np.abs(large_image), axis=-1)

    vmax = np.percentile(new_image, PERCENTILE)
    vmin = np.min(new_image)

    return np.clip((new_image - vmin) / (vmax - vmin), 0, 1)

def visualize_tensor_negatives(large_image, PERCENTILE=99):
    r"""Returns a 3D tensor as a 2D tensor with positive and negative values.
    """
    new_image = np.sum(large_image, axis=-1)

    span = abs(np.percentile(new_image, PERCENTILE))
    vmin = -span
    vmax = span

    return np.clip((new_image - vmin) / (vmax - vmin), -1, 1)

def overlay_gradcam(img, cam3, DIM=DIMENSION):
    img = np.uint8(255 * norm_image(img))
    cam3 = np.uint8(255 * cam3)

    cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
    cam3 = cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB)

    if DIM=="3d":
        img= img[..., np.newaxis]

    new_img = 0.3 * cam3 + 0.5 * img
    return (new_img * 255.0 / new_img.max()).astype("uint8")

def overlay_grad(img, grad, DIM=DIMENSION):
    img = np.uint8(255 * norm_image(img))
    grad = np.uint8(255 * grad)

    grad = cv2.applyColorMap(grad, cv2.COLORMAP_BONE) #cv2.COLORMAP_JET
    grad = cv2.cvtColor(grad, cv2.COLOR_BGR2RGB)

    if DIM=="3d":
        img= img[..., np.newaxis]

    new_img = 0.5 * grad + 0.3 * img
    return (new_img * 255.0 / new_img.max()).astype("uint8")

def overlay_pred(img, pred, DIM=DIMENSION):
    if DIM=="3d":
        img= img[..., np.newaxis]

    new_img = 0.3 * pred + img
    return (new_img * 255.0 / new_img.max()).astype("uint8")


#**START**
image_path = 'C:/Users/Mohammed A. Mattar/Downloads/Grade 3/UCSF-PDGM-0243_nifti/UCSF-PDGM-0243_FLAIR.nii.gz'
test_load = nib.load(image_path).get_fdata()
test_load.shape
(x, y, z) = test_load.shape
test = test_load[:, :, 130]
plt.imshow(test)
plt.show()

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(test_load[:, :, 95 + i])
    plt.gcf().set_size_inches(10, 10)
plt.show()



IMG_SHAPE = (192, 224, 160)
# NeuroXAI parameters
DIMENSION = "3d"
MODALITY = "FLAIR"
XAI_MODE = "segmentation"
CLASS_IDs = [1, 2, 3]
TUMOR_LABEL = "all" # for GCAM visualization
LAYER_NAME = None #'output_layer'
#LAYER_NAME = 'conv3d_18' #conv3d_10
XAI="GCAM"
# get the segmentation model
s_model = get_deepseg(WEIGHTS="C:/Users/Mohammed A. Mattar/Downloads/DeepSeg_model.hdf5")
model = get_xai_segmentation_model(s_model, LAYER_NAME)

# Sample MRI case
FILE = "UCSF-PDGM-0011_nifti"
ID = "UCSF-PDGM-0011"
SLICE_ID = 100
CLASS_ID = 2 #np.argmax(predictions[0])
TUMOR_LABEL= "all" # for grad-CAM
io_imgs = load_images(model, ID, FILE, PATH_DATA="C:/Users/Mohammed A. Mattar/Downloads/Grade 4")
im_orig = io_imgs[:,:,:,SLICE_ID,0] # 2D FLAIR


get_neuroxai_cnn(ID, model, io_imgs, CLASS_ID=2, SLICE_ID=100,
                      LAYER_NAME=None, MODALITY="FLAIR", XAI_MODE="segmentation",
                      DIMENSION="3d", SAVE_RESULTS=True, SAVE_PATH="C:/Users/Mohammed A. Mattar/Downloads/UCSF-PDGM-0011-Result")

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

# Define base directory for your data
base_dir = 'C:/Users/Mohammed A. Mattar/Downloads'

# Specify the grade folder and patient folder you're processing
grade_folder = 'UCSF-PDGM-0011-Result'  # Example grade folder
patient_folder = 'UCSF-PDGM-0011'  # Example patient folder (update as needed)

# Full directory path to the patient folder
root_dir = os.path.join(base_dir, grade_folder)
patient_folder_path = os.path.join(root_dir, patient_folder)

# Create output directory for visualizations
output_vis_dir = os.path.join(root_dir, 'visualizations')
os.makedirs(output_vis_dir, exist_ok=True)

all_stats = []

# Loop through the files in the patient folder to find the tumor mask
found_mask = False
for file in os.listdir(patient_folder_path):
    if file.endswith("_output_layer_all_FLAIR_GCAM_pred.png"):
        file_path = os.path.join(patient_folder_path, file)
        found_mask = True

        # Read the mask image
        mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Could not read: {file_path}")
            break

        # Binarize the mask
        _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        ys, xs = np.where(binary_mask == 1)
        if len(xs) == 0 or len(ys) == 0:
            print(f"No tumor found in {file_path}")
            break

        # Assume pixel spacing (mm to cm)
        pixel_spacing_mm = 0.5
        pixel_spacing_cm = pixel_spacing_mm / 10

        # Calculate stats
        tumor_area = int(np.sum(binary_mask))
        tumor_area_cm2 = tumor_area * (pixel_spacing_cm ** 2)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        tumor_width_cm = (x_max - x_min) * pixel_spacing_cm
        tumor_height_cm = (y_max - y_min) * pixel_spacing_cm
        centroid = ndimage.center_of_mass(binary_mask)

        # Save the patient ID from the mask file name
        patient_id = os.path.splitext(file)[0]

        all_stats.append({
            "PatientID": patient_id,
            "TumorArea(cm²)": tumor_area_cm2,
            "TumorWidth(cm)": tumor_width_cm,
            "TumorHeight(cm)": tumor_height_cm,
            "CentroidY": centroid[0],
            "CentroidX": centroid[1],
        })

        # Create and save a visualization of the tumor mask
        patient_vis_dir = os.path.join(output_vis_dir, patient_id)
        os.makedirs(patient_vis_dir, exist_ok=True)

        fig, ax = plt.subplots()
        ax.imshow(binary_mask, cmap='inferno')

        # Draw tumor boundary instead of a bounding box
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim == 2:
                ax.plot(contour[:, 0], contour[:, 1], color='red', linewidth=2, label='Tumor Boundary')

        ax.set_title(f"Patient ID: {patient_id[:14]}\nTumor Area: {tumor_area_cm2:.2f} cm²")
        ax.legend(loc='upper right', facecolor='lightgray')
        plt.axis('off')

        vis_path = os.path.join(patient_vis_dir, f"{patient_id}_visualization.png")
        plt.savefig(vis_path, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization for {patient_id}")
        break

# If no mask was found, notify the user
if not found_mask:
    print(f"No matching mask file found in: {patient_folder}")

# Save the stats to a CSV file
df_stats = pd.DataFrame(all_stats)
csv_path = os.path.join(root_dir, 'UCSF-PDGM-0011-Stats.csv')
df_stats.to_csv(csv_path, index=False)
print(f"Saved summary CSV for {grade_folder}")

model.save("C:/Users/Mohammed A. Mattar/Downloads/xploramed.h5")

# base_dir = 'C:/Users/Mohammed A. Mattar/Downloads'

# # List of grade folders to process
# grade_folders = ['Grade 2 Results', 'Grade 3 Results', 'Grade 4 Results']

# # Process each grade folder
# for grade_folder in grade_folders:
#     root_dir = os.path.join(base_dir, grade_folder)
#     output_vis_dir = os.path.join(root_dir, 'visualizations')
#     os.makedirs(output_vis_dir, exist_ok=True)

#     all_stats = []

#     # Loop through each patient folder
#     for patient_folder in sorted(os.listdir(root_dir)):
#         folder_path = os.path.join(root_dir, patient_folder)
#         if not os.path.isdir(folder_path):
#             continue

#         found_mask = False
#         for file in os.listdir(folder_path):
#             if file.endswith("_output_layer_all_FLAIR_GCAM_pred.png"):
#                 file_path = os.path.join(folder_path, file)
#                 found_mask = True

#                 mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#                 if mask is None:
#                     print(f"Could not read: {file_path}")
#                     break

#                 _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
#                 ys, xs = np.where(binary_mask == 1)
#                 if len(xs) == 0 or len(ys) == 0:
#                     print(f"No tumor found in {file_path}")
#                     break

#                 # Assume pixel spacing
#                 pixel_spacing_mm = 0.5
#                 pixel_spacing_cm = pixel_spacing_mm / 10

#                 # Stats
#                 tumor_area = int(np.sum(binary_mask))
#                 tumor_area_cm2 = tumor_area * (pixel_spacing_cm ** 2)
#                 x_min, x_max = xs.min(), xs.max()
#                 y_min, y_max = ys.min(), ys.max()
#                 tumor_width_cm = (x_max - x_min) * pixel_spacing_cm
#                 tumor_height_cm = (y_max - y_min) * pixel_spacing_cm
#                 centroid = ndimage.center_of_mass(binary_mask)

#                 patient_id = os.path.splitext(file)[0]

#                 all_stats.append({
#                     "PatientID": patient_id,
#                     "TumorArea(cm²)": tumor_area_cm2,
#                     "TumorWidth(cm)": tumor_width_cm,
#                     "TumorHeight(cm)": tumor_height_cm,
#                     "CentroidY": centroid[0],
#                     "CentroidX": centroid[1],
#                 })

#                 # Save patient visualization
#                 patient_vis_dir = os.path.join(output_vis_dir, patient_id)
#                 os.makedirs(patient_vis_dir, exist_ok=True)

#                 fig, ax = plt.subplots()
#                 ax.imshow(binary_mask, cmap='inferno')
#                 # ax.scatter(centroid[1], centroid[0], color='cyan', label='Centroid', s=60)

#                 # Draw tumor boundary instead of bounding box
#                 contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 for contour in contours:
#                     contour = contour.squeeze()
#                     if contour.ndim == 2:
#                         ax.plot(contour[:, 0], contour[:, 1], color='red', linewidth=2, label='Tumor Boundary')

#                 ax.set_title(f"Patient ID: {patient_id[:14]}\nTumor Area: {tumor_area_cm2:.2f} cm²")
#                 ax.legend(loc='upper right', facecolor='lightgray')
#                 plt.axis('off')

#                 vis_path = os.path.join(patient_vis_dir, f"{patient_id}_visualization.png")
#                 plt.savefig(vis_path, bbox_inches='tight')
#                 plt.close()
#                 print(f"Saved visualization for {patient_id}")
#                 break

#         if not found_mask:
#             print(f"No matching mask file found in: {patient_folder}")

#     # Save CSV summary for this grade
#     df_stats = pd.DataFrame(all_stats)
#     csv_path = os.path.join(root_dir, 'tumor_stats_summary.csv')
#     df_stats.to_csv(csv_path, index=False)
#     print(f"Saved summary CSV for {grade_folder}")
# # # Code for measuring the actual tumor area from the NII file.

# # nii = nib.load("C:/Users/Mohammed A. Mattar/Downloads/Grade 4/UCSF-PDGM-0011_nifti/UCSF-PDGM-0011_tumor_segmentation.nii.gz")
# # spacing = nii.header.get_zooms()[:2]  # (x_spacing, y_spacing)


# # tumor_area_cm2 = tumor_area * (spacing[0] / 10) * (spacing[1] / 10)
# # print(tumor_area_cm2)
# df = pd.read_csv("C:/Users/Mohammed A. Mattar/Downloads/UCSF-PDGM-metadata_v2.csv")

# selected_columns = [
#     'ID',
#     'Sex',
#     'Age at MRI',
#     'WHO CNS Grade',
#     'Final pathologic diagnosis (WHO 2021)',
#     '1-dead 0-alive',
#     'OS',
#     "MGMT status",
#     "Biopsy prior to imaging",
#     "EOR"
# ]


# filtered_df = df[selected_columns]


# output_path = "C:/Users/Mohammed A. Mattar/Downloads/extracted_csv.csv"

# # Save to the new path
# filtered_df.to_csv(output_path, index=False)

# data = pd.read_csv("C:/Users/Mohammed A. Mattar/Downloads/extracted_csv.csv")
# data



# grade_folders = [
#     "C:/Users/Mohammed A. Mattar/Downloads/Grade 2",
#     "C:/Users/Mohammed A. Mattar/Downloads/Grade 3",
#     "C:/Users/Mohammed A. Mattar/Downloads/Grade 4"
# ]

# # Load CSV
# csv_path = "C:/Users/Mohammed A. Mattar/Downloads/extracted_csv.csv"
# df = pd.read_csv(csv_path)

# # Clean the ID column (remove spaces just in case)
# df["ID"] = df["ID"].str.strip()

# # Extract and normalize IDs from folder names
# all_ids = []
# for folder in grade_folders:
#     for patient_folder in os.listdir(folder):
#         if patient_folder.endswith("_nifti"):
#             raw_id = patient_folder.replace("_nifti", "")
#             # Normalize to match CSV: pad number to 3 digits
#             parts = raw_id.split("-")
#             normalized_id = f"{parts[0]}-{parts[1]}-{int(parts[2]):03d}"
#             all_ids.append(normalized_id)

# # Remove duplicates
# unique_ids = list(set(all_ids))

# # Filter the DataFrame
# filtered_df = df[df["ID"].isin(unique_ids)]

# # Save the result
# filtered_df.to_csv("C:/Users/Mohammed A. Mattar/Downloads/filtered_60_patients.csv", index=False)

# print("Saved:", len(filtered_df), "patients")
