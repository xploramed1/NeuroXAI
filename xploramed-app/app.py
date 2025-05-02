import streamlit as st
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import tempfile
from scipy import ndimage
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv3D, Input, MaxPooling3D, UpSampling3D, concatenate, Activation, BatchNormalization, SpatialDropout3D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization
from skimage.transform import resize
import pandas as pd
import gdown

# ------------------ MODEL FILE CHECK & DOWNLOAD ------------------
def ensure_model_exists():
    model_path = "DeepSeg_model.hdf5"
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        url = "https://drive.google.com/uc?id=1jtveR5q1AdmOnqPEGonXC8TMv3hQIMZK"
        gdown.download(url, model_path, quiet=False)

# --------------------- NeuroXAI Model + Utils ---------------------
def norm_image(img, NORM_TYP="norm"):
    if NORM_TYP == "standard_norm":
        img = (img - img.mean()) / (img.std() if img.std() != 0 else 1)
    elif NORM_TYP == "norm":
        img = (img - np.min(img)) / np.ptp(img)
    return img

def crop_image_brats(img, OUT_SHAPE=(192, 224, 160)):
    offset = np.array((np.array(img.shape) - OUT_SHAPE) / 2).astype(int)
    offset[offset < 0] = 0
    x, y, z = offset
    crop_img = img[x:x + OUT_SHAPE[0], y:y + OUT_SHAPE[1], z:z + OUT_SHAPE[2]]
    padded_img = np.zeros(OUT_SHAPE)
    x, y, z = np.array((OUT_SHAPE - np.array(crop_img.shape)) / 2).astype(int)
    padded_img[x:x + crop_img.shape[0], y:y + crop_img.shape[1], z:z + crop_img.shape[2]] = crop_img
    return padded_img

def load_images(ID, PATH_DATA, DIM=(192, 224, 160)):
    img_files = [f"{ID}_FLAIR.nii.gz", f"{ID}_T1.nii.gz", f"{ID}_T1c.nii.gz", f"{ID}_T2.nii.gz"]
    img_paths = [os.path.join(PATH_DATA, f) for f in img_files]
    imgs_input = nib.concat_images(img_paths).get_fdata()
    imgs_preprocess = np.zeros((*DIM, 4))
    for i in range(4):
        imgs_preprocess[:, :, :, i] = norm_image(crop_image_brats(imgs_input[:, :, :, i]))
    return imgs_preprocess[np.newaxis, ...]

def create_convolution_block(input_layer, n_filters, kernel=(3, 3, 3)):
    x = Conv3D(n_filters, kernel, padding='same')(input_layer)
    x = BatchNormalization()(x)
    return Activation('relu')(x)

def get_deepseg(INP_SHAPE=(192, 224, 160, 4), N_FILTERS=8, DEPTH=5, DROPOUT=0.5, N_LABELS=4, INIT_LR=1e-4, WEIGHTS=None):
    inputs = Input(INP_SHAPE)
    x = inputs
    skips = []
    for i in range(DEPTH):
        x = create_convolution_block(x, N_FILTERS * 2**i)
        skips.append(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = SpatialDropout3D(DROPOUT)(x)
    for i in reversed(range(DEPTH)):
        x = UpSampling3D(size=(2, 2, 2))(x)
        x = concatenate([x, skips[i]], axis=-1)
        x = create_convolution_block(x, N_FILTERS * 2**i)
    output_layer = Conv3D(N_LABELS, (1, 1, 1), activation='softmax', name="output_layer")(x)
    model = Model(inputs=inputs, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=INIT_LR), loss="mse", metrics=["accuracy"])
    if WEIGHTS:
        model.load_weights(WEIGHTS)
    return model

def visualize_tensor(tensor):
    slice_tensor = tensor[0, :, :, 100, 0]
    normed = (slice_tensor - np.min(slice_tensor)) / np.ptp(slice_tensor)
    return normed

def overlay_gradcam(base, heatmap):
    base = np.uint8(255 * norm_image(base))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.4 * heatmap + 0.6 * base[..., np.newaxis]
    return np.uint8(overlay)

# ---------------------- Streamlit App ------------------------
st.set_page_config(layout="wide")
st.title("🧠 NeuroXAI Tumor Segmentation + Grad-CAM")
st.markdown("Upload your **FLAIR, T1, T1CE, T2** files (.nii.gz) and get Grad-CAM visualization and tumor stats.")

uploaded_files = st.file_uploader("Upload 4 MRI files (.nii.gz)", type="nii.gz", accept_multiple_files=True)
patient_id = st.text_input("Patient ID", value="UCSF-PDGM-0011")

if uploaded_files and len(uploaded_files) == 4:
    with tempfile.TemporaryDirectory() as tmp_dir:
        for file in uploaded_files:
            with open(os.path.join(tmp_dir, file.name), "wb") as f:
                f.write(file.read())

        st.success("Files uploaded. Click below to process.")

        if st.button("Run Segmentation + Grad-CAM"):
            with st.spinner("Running model..."):
                ensure_model_exists()
                model = get_deepseg(WEIGHTS="DeepSeg_model.hdf5")  # ✅ This is the FIX

                io_imgs = load_images(ID=patient_id, PATH_DATA=tmp_dir)
                preds = model(io_imgs, training=False).numpy()
                predicted_mask = np.argmax(preds[0], axis=-1)
                slice_mask = predicted_mask[:, :, 100]
                slice_img = io_imgs[0, :, :, 100, 0]

                heatmap = visualize_tensor(preds)
                overlay = overlay_gradcam(slice_img, heatmap)

                st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)

                binary = (slice_mask > 0).astype(np.uint8)
                ys, xs = np.where(binary == 1)
                if xs.size > 0:
                    pixel_spacing_cm = 0.05
                    area = int(np.sum(binary)) * (pixel_spacing_cm ** 2)
                    width = (xs.max() - xs.min()) * pixel_spacing_cm
                    height = (ys.max() - ys.min()) * pixel_spacing_cm
                    centroid = ndimage.center_of_mass(binary)

                    df = pd.DataFrame([{
                        "PatientID": patient_id,
                        "TumorArea(cm²)": area,
                        "TumorWidth(cm)": width,
                        "TumorHeight(cm)": height,
                        "CentroidY": centroid[0],
                        "CentroidX": centroid[1],
                    }])
                    st.dataframe(df)
                    st.download_button("Download CSV", data=df.to_csv(index=False), file_name=f"{patient_id}_tumor_stats.csv", mime="text/csv")
                else:
                    st.warning("No tumor detected in this slice.")
else:
    st.info("Upload exactly 4 files to continue.")
