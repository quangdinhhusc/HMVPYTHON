from sklearn.neural_network import MLPClassifier
import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import random
import struct
from sklearn.datasets import fetch_openml, load_iris
import mlflow
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from PIL import Image
from sklearn.model_selection import KFold
from collections import Counter
from mlflow.tracking import MlflowClient

def run_NeuralNetwork_app():
    @st.cache_data  # Lưu cache để tránh load lại dữ liệu mỗi lần chạy lại Streamlit
    def get_sampled_pixels(images, sample_size=100_000):
        return np.random.choice(images.flatten(), sample_size, replace=False)

    @st.cache_data  # Cache danh sách ảnh ngẫu nhiên
    def get_random_indices(num_images, total_images):
        return np.random.randint(0, total_images, size=num_images)

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
        "Demo dự đoán",
        "Thông tin & Mlflow",
    ])
    # tab_info, tab_load, tab_preprocess, tab_split,  tab_demo, tab_log_info = tabs
    tab_info,tab_note,tab_load, tab_preprocess,  tab_demo ,tab_mlflow= tabs

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
        st.expander("**Thông tin mô hình**", expanded=True)
        # Assume model_option1 is selected from somewhere in the app
            


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
            # Lựa chọn tham số huấn luyện
            hidden_layer_size = st.slider("Kích thước lớp ẩn", 50, 200, 100)
            max_iterations = st.slider("Số lần lặp tối đa", 5, 50, 10)

            if st.button("Huấn luyện mô hình"):
                with st.spinner("Đang huấn luyện..."):
                    cnn= MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), max_iter=max_iterations)
                    cnn.fit(X_train, y_train)
                    y_pred = cnn.predict(X_test)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    accuracy = accuracy_score(y_test, y_pred)
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
                predictions = cnn.predict(X_test[:10])

                for i in range(10):
                    st.write(f"Dự đoán: {predictions[i]}, Nhãn thực tế: {y_test[i]}")
                    image = X_test[i].reshape(28, 28)
                    st.image(image, caption=f"Mẫu {i+1}", width=100)

    with tab_demo:   
        with st.expander("**Dự đoán kết quả**", expanded=True):
            st.write("**Dự đoán trên ảnh do người dùng tải lên**")

            # Kiểm tra xem mô hình đã được huấn luyện và lưu kết quả chưa
            if "selected_model_type" not in st.session_state or "trained_model" not in st.session_state:
                st.warning("⚠️ Chưa có mô hình nào được huấn luyện. Vui lòng huấn luyện mô hình trước khi dự đoán.")
            else:
                best_model_name = st.session_state.selected_model_type
                best_model = st.session_state.trained_model

                st.write(f"🎯 Mô hình đang sử dụng: `{best_model_name}`")
                # st.write(f"✅ Độ chính xác trên tập kiểm tra: `{st.session_state.get('test_accuracy', 'N/A'):.4f}`")

                # Lấy các tham số từ session_state để hiển thị
                if best_model_name == "Decision Tree":
                    criterion = st.session_state.get("dt_criterion", "entropy")
                    max_depth = st.session_state.get("dt_max_depth", 5)  # Giá trị mặc định là 5
                    n_folds = st.session_state.get("n_folds", 5)  # Giá trị mặc định là 5
                    st.write("🔹 **Tham số mô hình Decision Tree:**")
                    st.write(f"- **Tiêu chí phân nhánh**: `{criterion}`")
                    st.write(f"- **Độ sâu tối đa**: `{max_depth}`")
                    st.write(f"- **Số folds trong Cross-Validation**: `{n_folds}`")
                elif best_model_name == "SVM":
                    kernel = st.session_state.get("svm_kernel", "linear")
                    C = st.session_state.get("svm_C", 1.0)
                    n_folds = st.session_state.get("n_folds", 5)  # Giá trị mặc định là 5
                    st.write("🔹 **Tham số mô hình SVM:**")
                    st.write(f"- **Kernel**: `{kernel}`")
                    st.write(f"- **C (Regularization)**: `{C}`")
                    st.write(f"- **Số folds trong Cross-Validation**: `{n_folds}`")

                # Cho phép người dùng tải lên ảnh
                uploaded_file = st.file_uploader("📂 Chọn một ảnh để dự đoán", type=["png", "jpg", "jpeg"])

                if uploaded_file is not None:
                    # Đọc ảnh từ tệp tải lên
                    image = Image.open(uploaded_file).convert("L")  # Chuyển sang ảnh xám
                    image = np.array(image)

                    # Kiểm tra xem dữ liệu huấn luyện đã lưu trong session_state hay chưa
                    if "X_train" in st.session_state:
                        X_train_shape = st.session_state["X_train"].shape[1]  # Lấy số đặc trưng từ tập huấn luyện

                        # Resize ảnh về kích thước phù hợp với mô hình đã huấn luyện
                        image = cv2.resize(image, (28, 28))  # Cập nhật kích thước theo dữ liệu ban đầu
                        image = image.reshape(1, -1)  # Chuyển về vector 1 chiều

                        # Đảm bảo số chiều đúng với dữ liệu huấn luyện
                        if image.shape[1] == X_train_shape:
                            prediction = best_model.predict(image)[0]

                            # Hiển thị ảnh và kết quả dự đoán
                            st.image(uploaded_file, caption="📷 Ảnh bạn đã tải lên", use_container_width=True)
                            st.success(f"✅ **Dự đoán:** {prediction}")
                        else:
                            st.error(f"🚨 Ảnh không có số đặc trưng đúng ({image.shape[1]} thay vì {X_train_shape}). Hãy kiểm tra lại dữ liệu đầu vào!")
                    else:
                        st.error("🚨 Dữ liệu huấn luyện không tìm thấy. Hãy huấn luyện mô hình trước khi dự đoán.")

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
    run_NeuralNetwork_app()
    # st.write(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    # print("🎯 Kiểm tra trên DagsHub: https://dagshub.com/Dung2204/MINST.mlflow/")
    # # # cd "C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App"
    # ClassificationMinst.
    



    ## thay vì decision tree là gini và entropy thì -> chỉ còn entropy với chọn độ sâu của cây
    ## bổ sung thêm Chọn số folds (KFold Cross-Validation) ở cả 2 phần decsion tree và svms
    ## cập nhật lại phần demo , vì nó đang không sử dụng dữ liệu ở phần huấn luyện
  
