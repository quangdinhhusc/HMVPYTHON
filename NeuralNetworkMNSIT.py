import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_mnist_neural_network(hidden_layer_sizes=(100,), max_iter=10, random_state=42):
    # ... (Mã huấn luyện mô hình như đã cung cấp trước đó) ...
    return mlp, accuracy, classification_report(y_test, y_pred, output_dict=True)

st.title("Huấn luyện Mạng Nơ-ron MNIST")

# Lựa chọn tham số huấn luyện
hidden_layer_size = st.sidebar.slider("Kích thước lớp ẩn", 50, 200, 100)
max_iterations = st.sidebar.slider("Số lần lặp tối đa", 5, 50, 10)

if st.button("Huấn luyện mô hình"):
    with st.spinner("Đang huấn luyện..."):
        model, accuracy, report = train_mnist_neural_network(hidden_layer_sizes=(hidden_layer_size,), max_iter=max_iterations)

    st.success("Huấn luyện hoàn tất!")
    st.write(f"Độ chính xác: {accuracy:.4f}")

    # Hiển thị báo cáo phân loại
    st.subheader("Báo cáo phân loại:")
    st.json(report)

    # (Tùy chọn) Hiển thị một số dự đoán mẫu
    st.subheader("Dự đoán mẫu:")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test = X_test / 255.0
    predictions = model.predict(X_test[:10])

    for i in range(10):
        st.write(f"Dự đoán: {predictions[i]}, Nhãn thực tế: {y_test[i]}")
        image = X_test[i].reshape(28, 28)
        st.image(image, caption=f"Mẫu {i+1}", width=100)