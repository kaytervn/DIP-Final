import tensorflow as tf
from tensorflow.keras import datasets, models, optimizers
import numpy as np
import os
import streamlit as st
from utils import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model_path = os.path.join(parent_dir, "models", "digit_model")

# Load model
model = tf.keras.models.load_model(model_path)

OPTIMIZER = tf.keras.optimizers.Adam()
inpWidth, inpHeight = 640, 640
model.compile(
    loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"]
)

(_, _), (X_test, _) = datasets.mnist.load_data()

X_test = X_test.reshape((10000, 28, 28, 1))


def app():
    def tao_anh_ngau_nhien():
        # Tạo 100 số nguyên ngẫu nhiên nằm trong phạm vi [0, 9999]
        index = np.random.randint(0, 9999, 100)
        sample = np.zeros((100, 28, 28, 1))
        for i in range(0, 100):
            sample[i] = X_test[index[i]]

        # Tạo ảnh để xem
        image = np.zeros((10 * 28, 10 * 28), np.uint8)
        k = 0
        for i in range(0, 10):
            for j in range(0, 10):
                image[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = sample[k, :, :, 0]
                k += 1
        frame = center_crop_resize(image, (inpWidth, inpHeight))
        img_container.image(frame)
        data = sample / 255.0
        data = data.astype("float32")
        return data

    def predict(data):
        ket_qua = model.predict(data, verbose=0)
        chu_so = []
        for i in range(0, 100):
            x = np.argmax(ket_qua[i])
            chu_so.append(x)
        s = []
        sub = ""
        dem = 0
        for x in chu_so:
            sub += str(x) + " "
            dem += 1
            if dem % 10 == 0:
                s.append(sub)
                sub = ""
        text_container.markdown(
            "".join(
                [
                    f"<p style='font-size: 30px; font-family: Consolas;'>{element}</p>"
                    for element in s
                ]
            ),
            unsafe_allow_html=True,
        )

    st.set_page_config(
        page_title="Handwritten Digit Detection", page_icon="​📖​​​", layout="wide"
    )
    st.title("🔍 Handwritten Digit Detection")
    st.sidebar.title("MNIST Recognition")
    st.sidebar.write("Click 'Create Image' to generate a random image for detection.")
    create_button = st.sidebar.button("Create Image")
    cols = st.columns(2)
    with cols[0]:
        input_container = st.empty()
        img_container = st.empty()
    with cols[1]:
        result_container = st.empty()
        text_container = st.empty()

    if create_button:
        input_container.subheader("Input")
        result_container.subheader("Result")
        data = tao_anh_ngau_nhien()
        predict(data)


app()
