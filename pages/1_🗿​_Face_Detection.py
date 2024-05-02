import numpy as np
import cv2
import joblib
import streamlit as st
import os
from utils import *
from streamlit_webrtc import webrtc_streamer
import av

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
face_detection_model = os.path.join(
    parent_dir, "models", "face_detection_yunet_2023mar.onnx"
)
face_recognition_model = os.path.join(
    parent_dir, "models", "face_recognition_sface_2021dec.onnx"
)
svc = os.path.join(parent_dir, "models", "svc.pkl")
inpWidth, inpHeight = 640, 640
score_threshold = 0.5
nms_threshold = 0.3
top_k = 5000
recognizer = cv2.FaceRecognizerSF.create(face_recognition_model, "")

detector = cv2.FaceDetectorYN.create(
    face_detection_model,
    "",
    (inpWidth, inpHeight),
    score_threshold,
    nms_threshold,
    top_k,
)
detector.setInputSize([inpWidth, inpHeight])

tm = cv2.TickMeter()

svc = joblib.load(svc)
mydict = ["CongPhan", "DucTrong", "HPCong", "Loki", "ThayDuc"]

st.set_page_config(page_title="Face Detection", page_icon="🗿")
st.title("🗿 Face Detection")
st.write(
    "This program detects 5 specific faces, displays the names in each frame, and if a face is not found in the dataset, it will be displayed as 'Unknown' with a red frame."
)

uploaded_containter = st.empty()
cam_container = st.empty()
cols = st.columns(2)
with cols[0]:
    input_container = st.empty()
    video_container = st.empty()
with cols[1]:
    result_container = st.empty()
    img_container = st.empty()

selected_option = st.sidebar.selectbox(
    "Select an option",
    ["Realtime Video Capture", "Upload Video to Detect"],
    index=0,
)


def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for face in faces[1]:
            if checkValidFace(input, face):
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            coords = face[:-1].astype(np.int32)
            cv2.rectangle(
                input,
                (coords[0], coords[1]),
                (coords[0] + coords[2], coords[1] + coords[3]),
                color,
                thickness,
            )
            cv2.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv2.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv2.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv2.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv2.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv2.putText(
        input,
        "FPS: {:.2f}".format(fps),
        (1, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )


def checkValidFace(frame, face_box):
    face_align = recognizer.alignCrop(frame, face_box)
    face_feature = recognizer.feature(face_align)
    test_predict = svc.predict(face_feature)
    confidence = np.max(np.abs(svc.decision_function(face_feature)))
    if confidence > score_threshold:
        return test_predict
    else:
        return None


def checkValidFace(frame, face_box):
    face_align = recognizer.alignCrop(frame, face_box)
    face_feature = recognizer.feature(face_align)
    test_predict = svc.predict(face_feature)
    confidence = np.max(np.abs(svc.decision_function(face_feature)))
    if confidence > score_threshold:
        return test_predict
    else:
        return None


def video_frame_callback(frame):
    image = frame.to_ndarray(format="bgr24")
    image = standardize_image(image, (inpWidth, inpHeight))
    image = cv2.addWeighted(image, 1.5, np.zeros(image.shape, image.dtype), 0, 0)
    tm.start()
    faces = detector.detect(image)
    tm.stop()
    if faces[1] is not None:
        for face_box in faces[1]:
            test_predict = checkValidFace(image, face_box)
            if test_predict:
                result = mydict[test_predict[0]]
                color = (0, 255, 0)
            else:
                result = "Unknown"
                color = (0, 0, 255)
            cv2.putText(
                image,
                result,
                (int(face_box[0]), int(face_box[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        visualize(image, faces, tm.getFPS())
    return av.VideoFrame.from_ndarray(image, format="bgr24")


def app():
    def process(capture, container, frame_skip=1):
        cap = cv2.VideoCapture(capture)
        detector = cv2.FaceDetectorYN.create(
            face_detection_model,
            "",
            (inpWidth, inpHeight),
            score_threshold,
            nms_threshold,
            top_k,
        )
        detector.setInputSize([inpWidth, inpHeight])
        tm = cv2.TickMeter()
        cur_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = standardize_image(frame, (inpWidth, inpHeight))
            tm.start()
            faces = detector.detect(frame)
            tm.stop()
            if cur_frame % frame_skip == 0:
                if faces[1] is not None:
                    for face_box in faces[1]:
                        test_predict = checkValidFace(frame, face_box)
                        if test_predict:
                            result = mydict[test_predict[0]]
                            color = (0, 255, 0)
                        else:
                            result = "Unknown"
                            color = (0, 0, 255)
                        cv2.putText(
                            frame,
                            result,
                            (int(face_box[0]), int(face_box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )
                visualize(frame, faces, tm.getFPS())
                try:
                    container.image(frame, channels="BGR")
                except Exception as e:
                    cur_frame = 0

            cur_frame += 1
        cap.release()

    def reset_display():
        uploaded_containter.empty()
        cam_container.empty()
        input_container.empty()
        video_container.empty()
        result_container.empty()
        img_container.empty()

    if selected_option == "Realtime Video Capture":
        reset_display()
        webrtc_streamer(
            key="face-detection",
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
        )

    elif selected_option == "Upload Video to Detect":
        reset_display()
        uploaded_video = uploaded_containter.file_uploader(
            "Choose video", type=["mp4", "mov"]
        )

        if uploaded_video:
            input_container.subheader("Input")
            with open(uploaded_video.name, "rb") as v:
                video_container.video(v)
            result_container.subheader("Result")
            process(uploaded_video.name, img_container, 5)


app()
