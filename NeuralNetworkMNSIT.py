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
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow import keras

def preprocess_canvas_image(canvas_result):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data[:, :, 0].astype(np.uint8))
        img = img.resize((28, 28)).convert("L")  # Resize và chuyển thành grayscale
        img = np.array(img, dtype=np.float32) / 255.0  # Chuẩn hóa về [0, 1]
        return img.reshape(1, -1)  # Chuyển thành vector 1D
    return None

def run_NeuralNetwork_app():
    

    # Cấu hình Streamlit    
    # st.set_page_config(page_title="Phân loại ảnh", layout="wide")
    # Định nghĩa hàm để đọc file .idx
    def load_mnist_images(filename):
        with open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        return images

    def load_mnist_labels(filename):
        with open(filename, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)
        return labels

    mlflow_tracking_uri = st.secrets["MLFLOW_TRACKING_URI"]
    mlflow_username = st.secrets["MLFLOW_TRACKING_USERNAME"]
    mlflow_password = st.secrets["MLFLOW_TRACKING_PASSWORD"]
    
    # Thiết lập biến môi trường
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
    
    # Thiết lập MLflow (Đặt sau khi mlflow_tracking_uri đã có giá trị)
    mlflow.set_tracking_uri(mlflow_tracking_uri)



    # Định nghĩa đường dẫn đến các file MNIST
    dataset_path = os.path.dirname(os.path.abspath(__file__)) 
    train_images_path = os.path.join(dataset_path, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(dataset_path, "train-labels.idx1-ubyte")
    test_images_path = os.path.join(dataset_path, "t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(dataset_path, "t10k-labels.idx1-ubyte")

    # Tải dữ liệu
    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)
    test_images = load_mnist_images(test_images_path)
    test_labels = load_mnist_labels(test_labels_path)

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
            st.write(f"🔍 Số lượng ảnh huấn luyện: `{train_images.shape[0]}`")
            st.write(f"🔍 Số lượng ảnh kiểm tra: `{test_images.shape[0]}`")

        with st.expander("**Hiển thị số lượng mẫu của từng chữ số từ 0 đến 9 trong tập huấn luyện**", expanded=True):
            label_counts = pd.Series(train_labels).value_counts().sort_index()

            # # Hiển thị biểu đồ cột
            # st.subheader("📊 Biểu đồ số lượng mẫu của từng chữ số")
            # st.bar_chart(label_counts)

            # Hiển thị bảng dữ liệu dưới biểu đồ
            st.subheader("📋 Số lượng mẫu cho từng chữ số")
            df_counts = pd.DataFrame({"Chữ số": label_counts.index, "Số lượng mẫu": label_counts.values})
            st.dataframe(df_counts)


            st.subheader("Chọn ngẫu nhiên 10 ảnh từ tập huấn luyện để hiển thị")
            num_images = 10
            random_indices = random.sample(range(len(train_images)), num_images)
            fig, axes = plt.subplots(1, num_images, figsize=(10, 5))

            for ax, idx in zip(axes, random_indices):
                ax.imshow(train_images[idx], cmap='gray')
                ax.axis("off")
                ax.set_title(f"Label: {train_labels[idx]}")

            st.pyplot(fig)
        with st.expander("**Kiểm tra hình dạng của tập dữ liệu**", expanded=True):    
            # Kiểm tra hình dạng của tập dữ liệu
            st.write("🔍 Hình dạng tập huấn luyện:", train_images.shape)
            st.write("🔍 Hình dạng tập kiểm tra:", test_images.shape)
            st.write("**Chuẩn hóa dữ liệu (đưa giá trị pixel về khoảng 0-1)**")
            # Chuẩn hóa dữ liệu
            train_images = train_images.astype("float32") / 255.0
            test_images = test_images.astype("float32") / 255.0

            # Hiển thị thông báo sau khi chuẩn hóa
            st.success("✅ Dữ liệu đã được chuẩn hóa về khoảng [0,1].") 

            # Hiển thị bảng dữ liệu đã chuẩn hóa (dạng số)
            num_samples = 5  # Số lượng mẫu hiển thị
            df_normalized = pd.DataFrame(train_images[:num_samples].reshape(num_samples, -1))  

            
            sample_size = 10_000  
            pixel_sample = np.random.choice(train_images.flatten(), sample_size, replace=False)
            if "train_images" not in st.session_state:
                st.session_state.train_images = train_images
                st.session_state.train_labels = train_labels
                st.session_state.test_images = test_images
                st.session_state.test_labels = test_labels


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
        with st.expander("**Phân chia dữ liệu**", expanded=True):    

            # Kiểm tra nếu dữ liệu đã được load
            if "train_images" in st.session_state:
                # Lấy dữ liệu từ session_state
                train_images = st.session_state.train_images
                train_labels = st.session_state.train_labels
                test_images = st.session_state.test_images
                test_labels = st.session_state.test_labels

                # Chuyển đổi dữ liệu thành vector 1 chiều
                X = np.concatenate((train_images, test_images), axis=0)  # Gộp toàn bộ dữ liệu
                y = np.concatenate((train_labels, test_labels), axis=0)
                X = X.reshape(X.shape[0], -1)  # Chuyển thành vector 1 chiều

                with mlflow.start_run():

                    # Cho phép người dùng chọn tỷ lệ validation và test
                    test_size = st.slider("🔹 Chọn % tỷ lệ tập test", min_value=10, max_value=50, value=20, step=5) / 100
                    val_size = st.slider("🔹 Chọn % tỷ lệ tập validation (trong phần train)", min_value=10, max_value=50, value=20, step=5) / 100

                    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    val_size_adjusted = val_size / (1 - test_size)  # Điều chỉnh tỷ lệ val cho phần còn lại
                    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)

                    # Tính tỷ lệ thực tế của từng tập
                    total_samples = X.shape[0]
                    test_percent = (X_test.shape[0] / total_samples) * 100
                    val_percent = (X_val.shape[0] / total_samples) * 100
                    train_percent = (X_train.shape[0] / total_samples) * 100
                st.write(f"📊 **Tỷ lệ phân chia**: Test={test_percent:.0f}%, Validation={val_percent:.0f}%, Train={train_percent:.0f}%")
                st.write("✅ Dữ liệu đã được xử lý và chia tách.")
                st.write(f"🔹 Kích thước tập huấn luyện: `{X_train.shape}`")
                st.write(f"🔹 Kích thước tập validation: `{X_val.shape}`")
                st.write(f"🔹 Kích thước tập kiểm tra: `{X_test.shape}`")
            else:
                st.error("🚨 Dữ liệu chưa được nạp. Hãy đảm bảo `train_images`, `train_labels` và `test_images` đã được tải trước khi chạy.")



    # 3️⃣ HUẤN LUYỆN MÔ HÌNH
    with tab_preprocess:
        with st.expander("**Huấn luyện Neural Network**", expanded=True):
            
            train_data_size = st.slider("Số lượng dữ liệu dùng để train model", 100, len(X_train), len(X_train), step=100)
            X_train = X_train[:train_data_size]
            y_train = y_train[:train_data_size]

            # Chuẩn hóa dữ liệu
            X_train = X_train / 255.0
            X_val = X_val / 255.0
            X_test = X_test / 255.0
            
            # Lựa chọn tham số huấn luyện
            k_folds = st.slider("Số fold cho Cross-Validation:", 3, 10, 5)
            
            num_layers = st.slider("Số lớp ẩn:", 1, 5, 2)

            epochs = st.slider("Số lần lặp tối đa", 2, 50, 5)

            learning_rate_init = st.slider("Tốc độ học", 0.001, 0.1, 0.01, step = 0.001, format="%.3f")

            activation = st.selectbox("Hàm kích hoạt:", ["relu", "sigmoid", "tanh"])

            num_neurons = st.selectbox("Số neuron mỗi lớp:", [32, 64, 128, 256], index=0)

            optimizer = st.selectbox("Chọn hàm tối ưu", ["adam", "sgd", "lbfgs"])

            loss_fn = "sparse_categorical_crossentropy"
            run_name = st.text_input("Nhập tên Run:", "Default_Run")
            st.session_state['run_name'] = run_name

            # Huấn luyện mô hình
            if st.button("⏹️ Huấn luyện mô hình"):
                with st.spinner("🔄 Đang huấn luyện..."):
                    

                    # Bắt đầu MLflow run với tên do người dùng chỉ định
                    with mlflow.start_run(run_name=run_name):
                        kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                        accuracies, losses = [], []

                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        total_folds = k_folds

                        # Khởi tạo mô hình một lần duy nhất trước vòng lặp K-fold
                        cnn = keras.Sequential([
                            layers.Input(shape=(X_train.shape[1],))
                        ] + [layers.Dense(num_neurons, activation=activation) for _ in range(num_layers)] + [
                            layers.Dense(10, activation="softmax")
                        ])
                        cnn.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

                        # Biến để lưu lịch sử huấn luyện tổng hợp qua tất cả các fold
                        full_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
                        total_time = 0  # Theo dõi tổng thời gian huấn luyện

                        for i, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
                            X_k_train, X_k_val = X_train[train_idx], X_train[val_idx]
                            y_k_train, y_k_val = y_train[train_idx], y_train[val_idx]
                            
                            progress_bar_epoch = st.progress(0)
                            
                            class EpochCallback(keras.callbacks.Callback):
                                def on_epoch_end(self, epoch, logs=None):
                                    progress_epoch = (epoch + 1) / epochs * 100
                                    progress_bar_epoch.progress(int(progress_epoch))
                                    st.write(f"Fold {i+1}/{k_folds}: Epoch {epoch+1}/{epochs}: Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")

                            start_time = time.time()
                            # Tiếp tục huấn luyện mô hình từ trạng thái của fold trước
                            history = cnn.fit(X_k_train, y_k_train, epochs=epochs, validation_data=(X_k_val, y_k_val), verbose=2, callbacks=[EpochCallback()])
                            elapsed_time = time.time() - start_time
                            total_time += elapsed_time

                            # Gộp lịch sử huấn luyện từ fold hiện tại vào full_history
                            full_history['loss'].extend(history.history['loss'])
                            full_history['accuracy'].extend(history.history['accuracy'])
                            full_history['val_loss'].extend(history.history['val_loss'])
                            full_history['val_accuracy'].extend(history.history['val_accuracy'])

                            # Lưu trữ độ chính xác và loss của fold hiện tại để tính trung bình
                            accuracies.append(history.history["val_accuracy"][-1])
                            losses.append(history.history["val_loss"][-1])

                            progress = (i + 1) / total_folds
                            progress_bar.progress(progress)
                            progress_text.text(f"️🎯Tiến trình huấn luyện: {int(progress * 100)}%")
                            
                        # Tính toán các chỉ số trung bình
                        avg_val_accuracy = np.mean(accuracies)
                        avg_val_loss = np.mean(losses)
                        
                        # Ghi log các chỉ số vào MLflow
                        mlflow.log_metrics({"avg_val_accuracy": avg_val_accuracy, "avg_val_loss": avg_val_loss, "total_time": total_time})
                        
                        test_loss, test_accuracy = cnn.evaluate(X_test, y_test, verbose=0)
                        mlflow.log_metrics({"test_accuracy": test_accuracy, "test_loss": test_loss})

                        # Lưu thông tin thí nghiệm và mô hình vào session_state
                        run_id = mlflow.active_run().info.run_id
                        st.session_state["trained_model"] = cnn
                        st.session_state["run_id"] = run_id
                        st.session_state["run_name"] = run_name
                        st.session_state["history"] = full_history
                        st.success(f"✅ Huấn luyện hoàn tất! Run ID: {run_id}")

                        # Ghi log các tham số
                        mlflow.log_param("train_data_size", train_data_size)
                        mlflow.log_param("k_folds", k_folds)
                        mlflow.log_param("num_layers", num_layers)
                        mlflow.log_param("epochs", epochs)
                        mlflow.log_param("learning_rate_init", learning_rate_init)
                        mlflow.log_param("activation", activation)
                        mlflow.log_param("num_neurons", num_neurons)
                        mlflow.log_param("optimizer", optimizer)
                        mlflow.log_param("loss_function", loss_fn)
                        mlflow.log_metric("train_accuracy", full_history['accuracy'][-1])
                        mlflow.log_metric("val_accuracy", full_history['val_accuracy'][-1])
                        mlflow.log_metric("final_train_loss", full_history['loss'][-1])
                        mlflow.log_metric("final_val_loss", full_history['val_loss'][-1])

                        st.write(f"Tổng thời gian huấn luyện: {total_time:.2f} giây")
                        st.write(f"Độ chính xác trung bình trên tập validation: {avg_val_accuracy:.4f}")
                        st.write(f"Độ chính xác trên tập test: {test_accuracy:.4f}")

                        # Vẽ biểu đồ Loss và Accuracy
                        st.markdown("---")
                        st.markdown("#### 📈**Biểu đồ Accuracy và Loss**")
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                        
                        ax1.plot(full_history['loss'], label='Train Loss', color='blue')
                        ax1.plot(full_history['val_loss'], label='Val Loss', color='orange')
                        ax1.set_title('Loss')
                        ax1.set_xlabel('Epoch')
                        ax1.set_ylabel('Loss')
                        ax1.legend()
                        
                        ax2.plot(full_history['accuracy'], label='Train Accuracy', color='blue')
                        ax2.plot(full_history['val_accuracy'], label='Val Accuracy', color='orange')
                        ax2.set_title('Accuracy')
                        ax2.set_xlabel('Epoch')
                        ax2.set_ylabel('Accuracy')
                        ax2.legend()

                        st.pyplot(fig)

    with tab_demo:   
        with st.expander("**Dự đoán kết quả**", expanded=True):
            st.write("**Dự đoán trên ảnh do người dùng tải lên**")

            # Kiểm tra xem mô hình đã được huấn luyện và lưu chưa
            if "trained_model" not in st.session_state:
                st.warning("⚠️ Chưa có mô hình nào được huấn luyện. Vui lòng huấn luyện mô hình trước khi dự đoán.")
            else:
                best_model = st.session_state["trained_model"]
                run_name = st.session_state["run_name"]
                st.write(f"Mô hình đang sử dụng: Mô hình đã huấn luyện `{run_name}`!")

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
            run_name = st.session_state["run_name"]
            st.success("✅ Mô hình đang sử dụng: Mô hình đã huấn luyện từ `{run_name}`!")
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
                # Preprocess canvas image
                img = Image.fromarray(canvas_result.image_data[:, :, 0].astype(np.uint8))
                img = img.resize((28, 28)).convert("L")  # Resize to 28x28 and convert to grayscale
                img_array = np.array(img, dtype=np.float32)

                # Normalize pixel values to [0, 1] (same as training data)
                img_normalized = img_array / 255.0

                # Reshape to 1D vector (1, 784) to match training input
                img_flat = img_normalized.reshape(1, -1)
                # Dự đoán số
                prediction = model.predict(img_flat)
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
            experiment_name = "NeuralNetwork"
    
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

                st.json(selected_run.data.metrics)
    
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
    run_NeuralNetwork_app()
    # st.write(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    # print("🎯 Kiểm tra trên DagsHub: https://dagshub.com/Dung2204/MINST.mlflow/")
    # # # cd "C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App"
    # ClassificationMinst.
    



    ## thay vì decision tree là gini và entropy thì -> chỉ còn entropy với chọn độ sâu của cây
    ## bổ sung thêm Chọn số folds (KFold Cross-Validation) ở cả 2 phần decsion tree và svms
    ## cập nhật lại phần demo , vì nó đang không sử dụng dữ liệu ở phần huấn luyện
  
