import streamlit as st
from PIL import Image

st.title("CKD Detection System")
st.write("Upload CBC report to predict CKD")

uploaded_file = st.file_uploader("Upload CBC Report Image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded CBC Report", use_column_width=True)
