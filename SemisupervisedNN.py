
import time
import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import random
import struct
import mlflow
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from sklearn.model_selection import train_test_split
from PIL import Image
from mlflow.tracking import MlflowClient
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow import keras

def preprocess_canvas_image(canvas_result):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data[:, :, 0].astype(np.uint8))
        img = img.resize((28, 28)).convert("L")  # Resize và chuyển thành grayscale
        img = np.array(img, dtype=np.float32) / 255.0  # Chuẩn hóa về [0, 1]
        return img.reshape(1, -1)  # Chuyển thành vector 1D
    return None


def load_mnist_data():
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    return X, y






def data_preparation():

    # Cho phép người dùng chọn tỷ lệ validation và test
    st.title("Chia dữ liệu Train/Test")
    
    # Tạo các biến để lưu dữ liệu

    test_percent = 0
    train_percent = 0
    indices_percent = 0

    X_train_initial = np.array([]).reshape(0,0)
    X_test_data = np.array([]).reshape(0,0)
    X_indices_data = np.array([]).reshape(0,0)
    y_train_initial = np.array([])

    
    
    # Đọc dữ liệu
    X, y = load_mnist_data()
    total_samples = X.shape[0] 
    
    # Thanh kéo chọn số lượng ảnh để train
    num_samples = st.slider("Chọn số lượng ảnh để huấn luyện:", 100, total_samples, 20000)
    
    if num_samples == total_samples:
        num_samples = num_samples - 10

    # Thanh kéo chọn tỷ lệ Train/Test
    test_size = st.slider("Chọn % dữ liệu Test", 10, 50, 20)
    train_size = 1
    indices_size = 100 - test_size - train_size

    st.write(f"**Tỷ lệ phân chia:** Test={test_size}%, Indices={indices_size}%, Train={train_size}%")

    # Tạo vùng trống để hiển thị kết quả
    result_placeholder = st.empty()
    # Tạo nút "Lưu Dữ Liệu"
    if st.button("Xác Nhận & Lưu Dữ Liệu"):
        
        # Phân chia dữ liệu
        X_selected, _, y_selected, _ = train_test_split(X, y, train_size=num_samples, stratify=y, random_state=42)
        X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_selected, y_selected, test_size=test_size/100, stratify=y_selected, random_state=42)
        
        # Lấy 1% số lượng ảnh cho mỗi class (0-9) để làm tập dữ liệu train ban đầu
        indices = []
        for i in range(10):
            class_indices = np.where(y_train_data == i)[0]
            num_samples = int(0.01 * len(class_indices))
            data_indices_random = np.random.choice(class_indices, num_samples, replace=False)
            indices.extend(data_indices_random)

        X_train_initial = X_train_data[indices]
        y_train_initial = y_train_data[indices]

        # Chuyển 99% còn lại sang tập val
        data_indices = np.setdiff1d(np.arange(len(X_train_data)), indices)
        X_indices_data = X_train_data[data_indices]
        y_indices_data = y_train_data[data_indices]


        # Lưu dữ liệu vào session_state
        st.session_state["X_train"] = X_train_initial
        st.session_state["y_train"] = y_train_initial
        st.session_state["X_indices"] = X_indices_data
        st.session_state["y_indices"] = y_indices_data
        st.session_state["X_test"] = X_test_data
        st.session_state["y_test"] = y_test_data

        summary_df = pd.DataFrame({"Tập dữ liệu": ["Train", "Test", "Indices"], "Số lượng mẫu": [X_train_initial.shape[0], X_test_data.shape[0], X_indices_data.shape[0]]})
        st.success("✅ Dữ liệu đã được chia thành công!")
        st.table(summary_df)
        
        # # Ghi log cho quá trình phân chia dữ liệu
        # mlflow.log_param("test_size", test_size)
        # mlflow.log_metric("test_percent", test_percent)
        # mlflow.log_metric("train_percent", train_percent)
        # mlflow.log_metric("val_percent", val_percent)
        # with result_placeholder:
        # Hiển thị kết quả
        
        


def learning_model():
    if "X_train" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return
    if "X_indices" not in st.session_state:
        st.error("⚠️ Dữ liệu X_indices không tồn tại! Vui lòng kiểm tra bước chuẩn bị dữ liệu.")
        return
        
    # Lấy dữ liệu từ session_state
    X_train = st.session_state["X_train"]
    X_indices = st.session_state["X_indices"]
    X_test = st.session_state["X_test"]
    y_train = st.session_state["y_train"]
    y_indices = st.session_state["y_indices"]
    y_test = st.session_state["y_test"]

    # Các tham số từ giao diện
    k_folds = st.slider("Số fold cho Cross-Validation:", 3, 10, 5)
    num_layers = st.slider("Số lớp ẩn:", 1, 5, 2)
    num_neurons = st.slider("Số neuron mỗi lớp:", 32, 512, 128, 32)
    epochs = st.slider("Số epochs:", min_value=1, max_value=50, value=20, step=1)
    learning_rate = st.slider("Tốc độ học (Learning Rate):", min_value=1e-4, max_value=1e-1, value=1e-3, step=1e-4, format="%.4f")
    activation = st.selectbox("Hàm kích hoạt:", ["relu", "sigmoid", "tanh"])
    optimizer = st.selectbox("Optimizer:", ["adam", "sgd", "rmsprop"])
    max_iterations = st.slider("Số vòng lặp tối đa cho pseudo-labeling:", 1, 10, 3)
    threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01)

    loss_fn = "sparse_categorical_crossentropy"
    run_name = st.text_input("Nhập tên Run:", "Default_Run")
    st.session_state['run_name'] = run_name
    
    if st.button("Huấn luyện mô hình"):
        with st.spinner("Đang huấn luyện..."):
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "num_layers": num_layers,
                "num_neurons": num_neurons,
                "activation": activation,
                "optimizer": optimizer,
                "learning_rate": learning_rate,
                "k_folds": k_folds,
                "epochs": epochs,
                "max_iterations": max_iterations,
                "threshold": threshold
            })

            X_unlabeled = X_indices.copy()
            iteration = 0
            overall_progress = st.progress(0)
            total_start_time = time.time()

            while len(X_unlabeled) > 0 and iteration < max_iterations:
                iteration += 1
                st.write(f"🔄 Vòng lặp pseudo-labeling thứ {iteration}")

                kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                accuracies, losses = [], []
                training_progress = st.progress(0)
                training_status = st.empty()

                # Cross-validation
                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
                    X_k_train, X_k_val = X_train[train_idx], X_train[val_idx]
                    y_k_train, y_k_val = y_train[train_idx], y_train[val_idx]

                    X_k_train_flat = X_k_train.reshape(-1, 28 * 28).astype('float32') / 255.0
                    X_k_val_flat = X_k_val.reshape(-1, 28 * 28).astype('float32') / 255.0

                    model = keras.Sequential([
                        layers.Input(shape=(28 * 28,))
                    ] + [layers.Dense(num_neurons, activation=activation) for _ in range(num_layers)
                    ] + [layers.Dense(10, activation="softmax")])

                    if optimizer == "adam":
                        opt = keras.optimizers.Adam(learning_rate=learning_rate)
                    elif optimizer == "sgd":
                        opt = keras.optimizers.SGD(learning_rate=learning_rate)
                    else:
                        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)

                    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

                    try:
                        start_time = time.time()
                        history = model.fit(X_k_train_flat, y_k_train, epochs=epochs, 
                                          validation_data=(X_k_val_flat, y_k_val), verbose=0)
                        accuracies.append(history.history["val_accuracy"][-1])
                        losses.append(history.history["val_loss"][-1])
                        elapsed_time = time.time() - start_time
                    except Exception as e:
                        st.error(f"Training failed in fold {fold_idx + 1}: {str(e)}")
                        mlflow.end_run()
                        return

                    progress_percent = int(((fold_idx + 1) / k_folds) * 100)
                    training_progress.progress(progress_percent)
                    training_status.text(f"⏳ Đang huấn luyện... {progress_percent}%")

                avg_val_accuracy = np.mean(accuracies)
                mlflow.log_metrics({
                    f"iter_{iteration}_avg_val_accuracy": avg_val_accuracy,
                    f"iter_{iteration}_avg_val_loss": np.mean(losses)
                })

                # Gán nhãn giả
                X_unlabeled_flat = X_unlabeled.reshape(-1, 28 * 28).astype('float32') / 255.0
                predictions = model.predict(X_unlabeled_flat, verbose=0)
                confidence_scores = np.max(predictions, axis=1)
                pseudo_labels = np.argmax(predictions, axis=1)

                confident_mask = confidence_scores >= threshold
                if np.sum(confident_mask) > 0:
                    X_confident = X_unlabeled[confident_mask]
                    y_confident = pseudo_labels[confident_mask]
                    X_train = np.concatenate([X_train, X_confident])
                    y_train = np.concatenate([y_train, y_confident])
                    X_unlabeled = X_unlabeled[~confident_mask]
                    st.write(f"✅ Đã thêm {np.sum(confident_mask)} mẫu vào tập huấn luyện")
                else:
                    st.write(f"⚠️ Không có mẫu nào đạt ngưỡng tin cậy {threshold}. Kết thúc sớm.")
                    break  # Thoát vòng lặp nếu không có mẫu nào được gán nhãn

                overall_progress.progress(min(iteration / max_iterations, 1.0))

            # Huấn luyện lại trên toàn bộ dữ liệu để có mô hình cuối cùng
            X_train_flat = X_train.reshape(-1, 28 * 28).astype('float32') / 255.0
            model.fit(X_train_flat, y_train, epochs=epochs, verbose=0)

            total_elapsed_time = time.time() - total_start_time
            mlflow.log_metrics({"total_elapsed_time": total_elapsed_time})
            mlflow.end_run()

            # Lưu mô hình
            st.session_state["trained_model"] = model

            st.success(f"✅ Quá trình huấn luyện và gán nhãn giả hoàn tất!")
            st.write(f"📊 **Độ chính xác trung bình cuối cùng trên tập validation:** {avg_val_accuracy:.4f}")
            st.write(f"⏱️ **Tổng thời gian huấn luyện:** {total_elapsed_time:.2f} giây")
            st.write(f"📈 **Số mẫu trong tập huấn luyện cuối cùng:** {len(X_train)}")



# def learning_model():
#     num=0
#     if "X_train" not in st.session_state:
#         st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
#         return
#     # Kiểm tra và lấy dữ liệu từ session_state
#     if "X_indices" not in st.session_state:
#         st.error("⚠️ Dữ liệu X_indices không tồn tại! Vui lòng kiểm tra bước chuẩn bị dữ liệu.")
#         return
        
#     # Lấy dữ liệu từ session_state
#     X_train = st.session_state["X_train"]
#     X_indices = st.session_state["X_indices"]
#     X_test = st.session_state["X_test"]
#     y_train = st.session_state["y_train"]
#     y_indices = st.session_state["y_indices"]
#     y_test = st.session_state["y_test"]

#     # Các tham số từ giao diện
#     k_folds = st.slider("Số fold cho Cross-Validation:", 3, 10, 5)
#     num_layers = st.slider("Số lớp ẩn:", 1, 5, 2)
#     num_neurons = st.slider("Số neuron mỗi lớp:", 32, 512, 128, 32)
#     activation = st.selectbox("Hàm kích hoạt:", ["relu", "sigmoid", "tanh"])
#     optimizer = st.selectbox("Optimizer:", ["adam", "sgd", "rmsprop"])
#     epochs = st.slider("🕰 Số epochs:", min_value=1, max_value=50, value=20, step=1)
#     learning_rate = st.slider("⚡ Tốc độ học (Learning Rate):", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-5, format="%.5f")
#     max_iterations = st.slider("Số vòng lặp tối đa cho pseudo-labeling:", 1, 10, 3)  # Thêm tham số mới
#     threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01)

#     loss_fn = "sparse_categorical_crossentropy"
#     run_name = st.text_input("🔹 Nhập tên Run:", "Default_Run")
#     st.session_state['run_name'] = run_name
    
#     if st.button("🚀 Huấn luyện mô hình"):
#         with st.spinner("Đang huấn luyện..."):
#             mlflow.start_run(run_name=run_name)
#             mlflow.log_params({
#                 "num_layers": num_layers,
#                 "num_neurons": num_neurons,
#                 "activation": activation,
#                 "optimizer": optimizer,
#                 "learning_rate": learning_rate,
#                 "k_folds": k_folds,
#                 "epochs": epochs,
#                 "max_iterations": max_iterations
#             })

#             X_unlabeled = X_indices.copy()  # Sao chép dữ liệu chưa gán nhãn
#             iteration = 0
#             overall_progress = st.progress(0)
            
#             while len(X_unlabeled) > 0 and iteration < max_iterations:
#                 iteration += 1
#                 st.write(f"🔄 Vòng lặp pseudo-labeling thứ {iteration}")

#                 # Huấn luyện mô hình với dữ liệu hiện tại
#                 kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
#                 accuracies, losses = [], []
#                 training_progress = st.progress(0)
#                 training_status = st.empty()

#                 for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
#                     X_k_train, X_k_val = X_train[train_idx], X_train[val_idx]
#                     y_k_train, y_k_val = y_train[train_idx], y_train[val_idx]

#                     # Chuẩn bị dữ liệu
#                     X_k_train_flat = X_k_train.reshape(-1, 28 * 28)
#                     X_k_val_flat = X_k_val.reshape(-1, 28 * 28)
#                     X_k_train_flat = X_k_train_flat.astype('float32') / 255.0
#                     X_k_val_flat = X_k_val_flat.astype('float32') / 255.0

#                     # Xây dựng mô hình
#                     model = keras.Sequential([
#                         layers.Input(shape=(28 * 28,))
#                     ] + [layers.Dense(num_neurons, activation=activation) for _ in range(num_layers)
#                     ] + [layers.Dense(10, activation="softmax")])

#                     # Chọn optimizer
#                     if optimizer == "adam":
#                         opt = keras.optimizers.Adam(learning_rate=learning_rate)
#                     elif optimizer == "sgd":
#                         opt = keras.optimizers.SGD(learning_rate=learning_rate)
#                     else:
#                         opt = keras.optimizers.RMSprop(learning_rate=learning_rate)

#                     model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

#                     try:
#                         start_time = time.time()
#                         history = model.fit(X_k_train_flat, y_k_train, epochs=epochs, 
#                                        validation_data=(X_k_val_flat, y_k_val), verbose=0)
#                         accuracies.append(history.history["val_accuracy"][-1])
#                         losses.append(history.history["val_loss"][-1])
#                         elapsed_time = time.time() - start_time
#                     except Exception as e:
#                         st.error(f"Training failed in fold {fold_idx + 1}: {str(e)}")
#                         return

#                     progress_percent = int(((fold_idx + 1) / k_folds) * 100)
#                     training_progress.progress(progress_percent)
#                     training_status.text(f"⏳ Đang huấn luyện... {progress_percent}%")

#                 avg_val_accuracy = np.mean(accuracies)
#                 mlflow.log_metrics({
#                     f"iter_{iteration}_avg_val_accuracy": avg_val_accuracy,
#                     f"iter_{iteration}_avg_val_loss": np.mean(losses)
#                 })

#                 # Gán nhãn giả cho X_unlabeled
#                 X_unlabeled_flat = X_unlabeled.reshape(-1, 28 * 28).astype('float32') / 255.0
#                 predictions = model.predict(X_unlabeled_flat)
#                 confidence_scores = np.max(predictions, axis=1)
#                 pseudo_labels = np.argmax(predictions, axis=1)

#                 # Lọc các mẫu có độ tin cậy >= threhold
#                 confident_mask = confidence_scores >= threshold
#                 if np.sum(confident_mask) > 0:
#                     X_confident = X_unlabeled[confident_mask]
#                     y_confident = pseudo_labels[confident_mask]

#                     # Thêm vào tập huấn luyện
#                     X_train = np.concatenate([X_train, X_confident])
#                     y_train = np.concatenate([y_train, y_confident])

#                     # Loại bỏ các mẫu đã được gán nhãn khỏi X_unlabeled
#                     X_unlabeled = X_unlabeled[~confident_mask]

#                     st.write(f"✅ Đã thêm {np.sum(confident_mask)} mẫu vào tập huấn luyện")
#                 else:
#                     st.write("⚠️ Không có mẫu nào đạt ngưỡng tin cậy 0.5")
#                     break

#                 overall_progress.progress(min(iteration / max_iterations, 1.0))

#             mlflow.log_metrics({"elapsed_time": elapsed_time})
#             mlflow.end_run()
#             st.session_state["trained_model"] = model

#             st.success(f"✅ Quá trình huấn luyện và gán nhãn giả hoàn tất!")
#             st.write(f"📊 **Độ chính xác trung bình cuối cùng trên tập validation:** {avg_val_accuracy:.4f}")
#             st.write(f"⏱️ **Tổng thời gian huấn luyện:** {elapsed_time:.2f} giây")
#             st.write(f"📈 **Số mẫu trong tập huấn luyện cuối cùng:** {len(X_train)}")


def run_PseudoLabelling_app():

    # Cấu hình Streamlit    
    # st.set_page_config(page_title="Phân loại ảnh", layout="wide")
    # Định nghĩa hàm để đọc file .idx
    

    

    mlflow_tracking_uri = st.secrets["MLFLOW_TRACKING_URI"]
    mlflow_username = st.secrets["MLFLOW_TRACKING_USERNAME"]
    mlflow_password = st.secrets["MLFLOW_TRACKING_PASSWORD"]
    
    # Thiết lập biến môi trường
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
    
    # Thiết lập MLflow (Đặt sau khi mlflow_tracking_uri đã có giá trị)
    mlflow.set_tracking_uri(mlflow_tracking_uri)




    # Giao diện Streamlit
    st.title("📸 Phân loại ảnh MNIST với Streamlit")
    tabs = st.tabs([
        "Thông tin dữ liệu",
        "Thông tin",
        "Xử lí dữ liệu",
        "Huấn luyện mô hình",
        "Demo dự đoán file ảnh",
        "Demo dự đoán Viết Tay",
        "Thông tin & Mlflow",
    ])
    # tab_info, tab_load, tab_preprocess, tab_split,  tab_demo, tab_log_info = tabs
    tab_info,tab_note,tab_load, tab_preprocess,  tab_demo, tab_demo_2 ,tab_mlflow= tabs

    # with st.expander("🖼️ Dữ liệu ban đầu", expanded=True):
    with tab_info:
        with st.expander("**Thông tin dữ liệu**", expanded=True):
            st.markdown(
                '''
                **MNIST** là phiên bản được chỉnh sửa từ bộ dữ liệu NIST gốc của Viện Tiêu chuẩn và Công nghệ Quốc gia Hoa Kỳ.  
                Bộ dữ liệu ban đầu gồm các chữ số viết tay từ nhân viên bưu điện và học sinh trung học.  

                Các nhà nghiên cứu **Yann LeCun, Corinna Cortes, và Christopher Burges** đã xử lý, chuẩn hóa và chuyển đổi bộ dữ liệu này thành **MNIST** để dễ dàng sử dụng hơn cho các bài toán nhận dạng chữ số viết tay.
                '''
            )
            # image = Image.open(r'C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App\image.png')
            X_Information, y_Information = load_mnist_data()
            # Gắn ảnh vào Streamlit và chỉnh kích thước
            # st.image(image, caption='Mô tả ảnh', width=600) 
            # Đặc điểm của bộ dữ liệu
        with st.expander("**Đặc điểm của bộ dữ liệu**", expanded=True):
            st.markdown(
                '''
                - **Số lượng ảnh:** 70.000 ảnh chữ số viết tay  
                - **Kích thước ảnh:** Mỗi ảnh có kích thước 28x28 pixel  
                - **Cường độ điểm ảnh:** Từ 0 (màu đen) đến 255 (màu trắng)  
                - **Dữ liệu nhãn:** Mỗi ảnh đi kèm với một nhãn số từ 0 đến 9  
                '''
            )
            st.write(f"🔍 Số lượng ảnh huấn luyện: `{X_Information.shape[0]}`")
            st.write(f"🔍 Số lượng ảnh kiểm tra: `{y_Information.shape[0]}`")

        with st.expander("**Hiển thị số lượng mẫu của từng chữ số từ 0 đến 9 trong tập huấn luyện**", expanded=True):
            label_counts = pd.Series(y_Information).value_counts().sort_index()

            # # Hiển thị biểu đồ cột
            st.subheader("📊 Biểu đồ số lượng mẫu của từng chữ số")
            st.bar_chart(label_counts)

            # Hiển thị bảng dữ liệu dưới biểu đồ
            st.subheader("📋 Số lượng mẫu cho từng chữ số")
            df_counts = pd.DataFrame({"Chữ số": label_counts.index, "Số lượng mẫu": label_counts.values})
            st.dataframe(df_counts)


            st.subheader("Chọn ngẫu nhiên 10 ảnh từ tập huấn luyện để hiển thị")
            num_images = 10
            random_indices = random.sample(range(len(y_Information)), num_images)
            fig, axes = plt.subplots(1, num_images, figsize=(10, 5))

            for ax, idx in zip(axes, random_indices):
                ax.imshow(X_Information[idx], cmap='gray')
                ax.axis("off")
                ax.set_title(f"Label: {y_Information[idx]}")

            st.pyplot(fig)
        with st.expander("**Kiểm tra hình dạng của tập dữ liệu**", expanded=True):    
            # Kiểm tra hình dạng của tập dữ liệu
            st.write("🔍 Hình dạng tập huấn luyện:", X_Information.shape)
            st.write("**Chuẩn hóa dữ liệu (đưa giá trị pixel về khoảng 0-1)**")



    with tab_note:
        with st.expander("**Thông tin mô hình**", expanded=True):
        # Assume model_option1 is selected from somewhere in the app
            st.markdown("""
                    ### Neural Network (NN)
                    """) 
            st.markdown("---")        
            st.markdown("""            
            ### Khái Niệm:  
            **Neural Network (NN)**:
            - Là một mô hình tính toán lấy cảm hứng từ cấu trúc và chức năng của mạng lưới thần kinh sinh học. Nó được tạo thành từ các nút kết nối với nhau, hay còn gọi là nơ-ron nhân tạo, được sắp xếp thành các lớp.
            - Ý tưởng chính của **Neural Network** là tạo ra một mô hình tính toán có khả năng học hỏi và xử lý thông tin giống như bộ não con người.
            """)

            st.markdown("---")        
                       
            st.write("### Mô Hình Tổng Quát:")   
            st.image("imgnn/modelnn.png",use_container_width ="auto")
            st.markdown(""" 
            - Layer đầu tiên là input layer, các layer ở giữa được gọi là hidden layer, layer cuối cùng được gọi là output layer. Các hình tròn được gọi là node.
            - Mỗi mô hình luôn có 1 input layer, 1 output layer, có thể có hoặc không các hidden layer. Tổng số layer trong mô hình được quy ước là số layer - 1 (Không tính input layer).
            - Mỗi node trong hidden layer và output layer :
                - Liên kết với tất cả các node ở layer trước đó với các hệ số w riêng.
                - Mỗi node có 1 hệ số bias b riêng.
                - Diễn ra 2 bước: tính tổng linear và áp dụng activation function.
            """)

            st.markdown("---")          
            st.markdown("""
            ### Nguyên lý hoạt động:  
            - Dữ liệu đầu vào được đưa vào lớp đầu vào.
            - Mỗi nơ-ron trong lớp ẩn nhận tín hiệu từ các nơ-ron ở lớp trước đó, xử lý tín hiệu và chuyển tiếp kết quả đến các nơ-ron ở lớp tiếp theo.
            - Quá trình này tiếp tục cho đến khi dữ liệu đến lớp đầu ra.
            - Kết quả đầu ra được tạo ra dựa trên các tín hiệu nhận được từ lớp ẩn cuối cùng.
            """)           
            st.markdown("---")
            st.markdown("""  
            ### Áp dụng vào ngữ cảnh Neural Network với MNIST:  
            - **MNIST (Modified National Institute of Standards and Technology database)** là một bộ dữ liệu kinh điển trong lĩnh vực học máy, đặc biệt là trong việc áp dụng mạng nơ-ron. Nó bao gồm 70.000 ảnh xám của chữ số viết tay (từ 0 đến 9), được chia thành 60.000 ảnh huấn luyện và 10.000 ảnh kiểm tra.
            - Mục tiêu của bài toán là phân loại chính xác chữ số từ 0 đến 9 dựa trên ảnh đầu vào.
            - Có nhiều cách để áp dụng mạng nơ-ron cho bài toán phân loại chữ số viết tay trên MNIST. Dưới đây là một số phương pháp phổ biến:
                - **Multi-Layer Perceptron (MLP)**: Một mô hình mạng nơ-ron sâu với nhiều lớp ẩn.
                - **Convolutional Neural Network (CNN)**: Một mô hình mạng nơ-ron sâu được thiết kế đặc biệt cho việc xử lý ảnh.
                - **Recurrent Neural Network (RNN)**: Một mô hình mạng nơ-ron sâu được thiết kế cho dữ liệu chuỗi.
            """)
    
    with tab_load:
        with st.expander("**Tải dữ liệu**", expanded=True):
            
            data_preparation()


    # 3️⃣ HUẤN LUYỆN MÔ HÌNH
    with tab_preprocess:
        with st.expander("**Huấn luyện Neural Network**", expanded=True):

            learning_model()

    with tab_demo:   
        with st.expander("**Dự đoán kết quả**", expanded=True):
            st.write("**Dự đoán trên ảnh do người dùng tải lên**")

            # Kiểm tra xem mô hình đã được huấn luyện và lưu chưa
            if "trained_model" not in st.session_state:
                st.warning("⚠️ Chưa có mô hình nào được huấn luyện. Vui lòng huấn luyện mô hình trước khi dự đoán.")
            else:
                best_model = st.session_state["trained_model"]
                st.write(f"Mô hình đang sử dụng: Mô hình đã huấn luyện từ `learning_model()`")

                # Cho phép người dùng tải lên ảnh
                uploaded_file = st.file_uploader("📂 Chọn một ảnh để dự đoán (28x28)", type=["png", "jpg", "jpeg"])

                if uploaded_file is not None:
                    # Đọc và tiền xử lý ảnh
                    image = Image.open(uploaded_file).convert("L")  # Chuyển sang ảnh xám
                    image = np.array(image)

                    # Resize ảnh về kích thước 28x28
                    image = cv2.resize(image, (28, 28))

                    # Phẳng hóa và chuẩn hóa giống dữ liệu huấn luyện
                    image_flat = image.reshape(1, 28 * 28).astype('float32') / 255.0

                    # Dự đoán với mô hình đã lưu
                    try:
                        prediction = best_model.predict(image_flat, verbose=0)[0]
                        predicted_class = np.argmax(prediction)
                        confidence = np.max(prediction)

                        # Hiển thị ảnh và kết quả dự đoán
                        st.image(uploaded_file, caption="📷 Ảnh bạn đã tải lên", width=200)
                        st.success(f"Dự đoán: **{predicted_class}** với độ tin cậy: **{confidence:.4f}**")

                        # (Tùy chọn) Hiển thị phân phối xác suất
                        st.write("Phân phối xác suất:")
                        st.bar_chart(prediction)

                    except Exception as e:
                        st.error(f"Lỗi khi dự đoán: {str(e)}. Hãy kiểm tra định dạng ảnh và mô hình!")

                else:
                    st.info("Vui lòng tải lên một ảnh để dự đoán.")

    with tab_demo_2:   
        st.header("✍️ Vẽ số để dự đoán")

        # 📥 Load mô hình đã huấn luyện
        if "trained_model" in st.session_state:
            model = st.session_state["trained_model"]
            st.success("✅ Đã sử dụng mô hình vừa huấn luyện!")
        else:
            st.error("⚠️ Chưa có mô hình! Hãy huấn luyện trước.")


        # 🆕 Cập nhật key cho canvas khi nhấn "Tải lại"
        if "key_value" not in st.session_state:
            st.session_state.key_value = str(random.randint(0, 1000000))  

        if st.button("🔄 Tải lại nếu không thấy canvas"):
            st.session_state.key_value = str(random.randint(0, 1000000))  

        # ✍️ Vẽ số
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            height=150,
            width=150,
            drawing_mode="freedraw",
            key=st.session_state.key_value,
            update_streamlit=True
        )

        if st.button("Dự đoán số"):
            img = preprocess_canvas_image(canvas_result)

            if img is not None:
                st.image(Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)

                # Dự đoán số
                prediction = model.predict(img)
                predicted_number = np.argmax(prediction, axis=1)[0]
                max_confidence = np.max(prediction)

                st.subheader(f"🔢 Dự đoán: {predicted_number}")
                st.write(f"📊 Mức độ tin cậy: {max_confidence:.2%}")

                # Hiển thị bảng confidence score
                prob_df = pd.DataFrame(prediction.reshape(1, -1), columns=[str(i) for i in range(10)]).T
                prob_df.columns = ["Mức độ tin cậy"]
                st.bar_chart(prob_df)

            else:
                st.error("⚠️ Hãy vẽ một số trước khi bấm Dự đoán!")

    

    with tab_mlflow:
        st.header("Thông tin Huấn luyện & MLflow UI")
        try:
            client = MlflowClient()
            experiment_name = "Classification"
    
            # Kiểm tra nếu experiment đã tồn tại
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = client.create_experiment(experiment_name)
                st.success(f"Experiment mới được tạo với ID: {experiment_id}")
            else:
                experiment_id = experiment.experiment_id
                st.info(f"Đang sử dụng experiment ID: {experiment_id}")
    
            mlflow.set_experiment(experiment_name)
    
            # Truy vấn các run trong experiment
            runs = client.search_runs(experiment_ids=[experiment_id])
    
            # 1) Chọn và đổi tên Run Name
            st.subheader("Đổi tên Run")
            if runs:
                run_options = {run.info.run_id: f"{run.data.tags.get('mlflow.runName', 'Unnamed')} - {run.info.run_id}"
                               for run in runs}
                selected_run_id_for_rename = st.selectbox("Chọn Run để đổi tên:", 
                                                          options=list(run_options.keys()), 
                                                          format_func=lambda x: run_options[x])
                new_run_name = st.text_input("Nhập tên mới cho Run:", 
                                             value=run_options[selected_run_id_for_rename].split(" - ")[0])
                if st.button("Cập nhật tên Run"):
                    if new_run_name.strip():
                        client.set_tag(selected_run_id_for_rename, "mlflow.runName", new_run_name.strip())
                        st.success(f"Đã cập nhật tên Run thành: {new_run_name.strip()}")
                    else:
                        st.warning("Vui lòng nhập tên mới cho Run.")
            else:
                st.info("Chưa có Run nào được log.")
    
            # 2) Xóa Run
            st.subheader("Danh sách Run")
            if runs:
                selected_run_id_to_delete = st.selectbox("", 
                                                         options=list(run_options.keys()), 
                                                         format_func=lambda x: run_options[x])
                if st.button("Xóa Run", key="delete_run"):
                    client.delete_run(selected_run_id_to_delete)
                    st.success(f"Đã xóa Run {run_options[selected_run_id_to_delete]} thành công!")
                    st.experimental_rerun()  # Tự động làm mới giao diện
            else:
                st.info("Chưa có Run nào để xóa.")
    
            # 3) Danh sách các thí nghiệm
            st.subheader("Danh sách các Run đã log")
            if runs:
                selected_run_id = st.selectbox("Chọn Run để xem chi tiết:", 
                                               options=list(run_options.keys()), 
                                               format_func=lambda x: run_options[x])
    
                # 4) Hiển thị thông tin chi tiết của Run được chọn
                selected_run = client.get_run(selected_run_id)
                st.write(f"**Run ID:** {selected_run_id}")
                st.write(f"**Run Name:** {selected_run.data.tags.get('mlflow.runName', 'Unnamed')}")
    
                st.markdown("### Tham số đã log")
                st.json(selected_run.data.params)
    
                st.markdown("### Chỉ số đã log")
                metrics = {
                    "mean_cv_accuracy": selected_run.data.metrics.get("mean_cv_accuracy", "N/A"),
                    "std_cv_accuracy": selected_run.data.metrics.get("std_cv_accuracy", "N/A"),
                    "accuracy": selected_run.data.metrics.get("accuracy", "N/A"),
                    "model_type": selected_run.data.metrics.get("model_type", "N/A"),
                    "kernel": selected_run.data.metrics.get("kernel", "N/A"),
                    "C_value": selected_run.data.metrics.get("C_value", "N/A")
                

                }
                st.json(metrics)
    
                # 5) Nút bấm mở MLflow UI
                st.subheader("Truy cập MLflow UI")
                mlflow_url = "https://dagshub.com/quangdinhhusc/HMVPYTHON.mlflow"
                if st.button("Mở MLflow UI"):
                    st.markdown(f'**[Click để mở MLflow UI]({mlflow_url})**')
            else:
                st.info("Chưa có Run nào được log. Vui lòng huấn luyện mô hình trước.")
    
        except Exception as e:
            st.error(f"Không thể kết nối với MLflow: {e}")

    


if __name__ == "__main__":
    run_PseudoLabelling_app()