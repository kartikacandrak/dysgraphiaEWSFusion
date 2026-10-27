import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# =========================================================
# MODEL SETUP
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.DenseNet201_Weights.DEFAULT
        self.base = models.densenet201(weights=weights)
        self.base.classifier = nn.Identity()

    def forward(self, x):
        return self.base(x)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    model = CNNFeatureExtractor().to(device)
    model.eval()
    return model

# =========================================================
# STREAMLIT RUN FUNCTION
# =========================================================
def run():
    st.header("🖼️ Extract Deep Feature)")
    st.write("Upload one or more handwritten images to extract deep embedding features using **DenseNet201**.")

    uploaded_images = st.file_uploader("Upload Handwriting Image", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_images:
        model = load_model()
        feats_list = []

        with st.spinner("Extract image feature..."):
            for img_file in uploaded_images:
                try:
                    img = Image.open(img_file).convert("RGB")
                    tensor_img = transform(img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        feat = model(tensor_img).cpu().numpy().flatten()
                        feats_list.append(feat)
                except Exception as e:
                    st.error(f"fail to process {img_file.name}: {e}")

        if feats_list:
            feats_array = np.array(feats_list, dtype=np.float32)
            st.success(f"Success extract {feats_array.shape[0]} image with {feats_array.shape[1]} feature")
            
            # Tombol Download NPY
            npy_bytes = feats_array.tobytes()
            st.download_button(
                label="💾 Download CNN Feature (.npy)",
                data=npy_bytes,
                file_name="offline_features.npy",
                mime="application/octet-stream"
            )