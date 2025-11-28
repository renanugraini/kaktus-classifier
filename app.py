Microsoft Windows [Version 10.0.26100.7171]
(c) Microsoft Corporation. All rights reserved.

C:\Users\Asus>import streamlit as st
'import' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>import numpy as np
'import' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>import tensorflow as tf
'import' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>from PIL import Image
'from' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>
C:\Users\Asus># Load TFLite model
'#' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>@st.cache_resource
'st.cache_resource' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>def load_model():
'def' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>    interpreter = tf.lite.Interpreter(model_path="model_kaktus.tflite")
'interpreter' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>    interpreter.allocate_tensors()
'interpreter.allocate_tensors' is not recognized as an internalor external command,
operable program or batch file.

C:\Users\Asus>    return interpreter
'return' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>
C:\Users\Asus>interpreter = load_model()
'interpreter' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>
C:\Users\Asus># Daftar nama kelas (sesuaikan sesuai dataset kamu)
'#' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>class_names = ["Astrophytum", "Gymnocalycium", "Mammillaria"]
'class_names' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>
C:\Users\Asus>st.title("🌵 Kaktus Classifier App")
'st.title' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>st.write("Upload gambar kaktus untuk mengidentifikasi jenisnya")
'st.write' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>
C:\Users\Asus>uploaded_file = st.file_uploader("Upload gambar",type=["jpg", "jpeg", "png"])
'uploaded_file' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>
C:\Users\Asus>def predict(image):
'def' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>    input_details = interpreter.get_input_details()
'input_details' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>    output_details = interpreter.get_output_details()
'output_details' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>
C:\Users\Asus>    img = image.resize((150, 150))  # sesuaikan ukuran sesuai model kamu
'img' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>    img_array = np.array(img) / 255.0
'img_array' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
'img_array' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>
C:\Users\Asus>    interpreter.set_tensor(input_details[0]['index'], img_array)
'interpreter.set_tensor' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>    interpreter.invoke()
'interpreter.invoke' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>    prediction = interpreter.get_tensor(output_details[0]['index'])
'prediction' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>    return prediction
'return' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>
C:\Users\Asus>if uploaded_file is not None:
is was unexpected at this time.
C:\Users\Asus>    image = Image.open(uploaded_file)
'image' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>    st.image(image, caption="Gambar yang diupload", use_column_width=True)
'st.image' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>
C:\Users\Asus>    preds = predict(image)
'preds' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>    score = float(np.max(preds))
'score' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>    result = class_names[np.argmax(preds)]
'result' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>
C:\Users\Asus>    st.markdown(f"## 🌵 Jenis Kaktus: **{result}**")
'st.markdown' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>    st.write(f"Confidence: **{score:.2f}**")
'st.write' is not recognized as an internal or external command,
operable program or batch file.

C:\Users\Asus>
