import os
import time
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from libsvm.svmutil import *

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="GNR602: SVM Classifier", layout="wide")
st.title("🛰️ GNR602: EuroSAT SVM Classifier")
st.markdown("Evaluate One-vs-One (OvO) and One-vs-Rest (OvR) SVM performance on satellite imagery.")

# --- Sidebar UI for Parameter Tuning ---
st.sidebar.header("⚙️ SVM Parameter Tuning")
kernel_options = ["Linear", "Polynomial", "RBF", "Sigmoid"]
kernel_var = st.sidebar.selectbox("Kernel:", kernel_options, index=2)
c_param = st.sidebar.number_input("Slack Parameter (C):", value=1000.0, step=10.0)
g_param = st.sidebar.number_input("Gamma (g):", value=0.1, step=0.01)
limit_param = st.sidebar.number_input("Images per Class:", min_value=1, value=500, step=50)

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Dataset Configuration")
# Defaults to the folder inside your GitHub repository
dataset_path = st.sidebar.text_input("Dataset Repository Path:", value="EuroSAT/")

# --- Core Functions ---
@st.cache_data(show_spinner=False)
def extract_features(img):
    """Extracts color means, std devs, HSV histograms, and edge magnitude."""
    img_f = img.astype(np.float32) / 255.0
    (means, stds) = cv2.meanStdDev(img_f)
    
    # Color Histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [4, 4, 4], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # Texture/Edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.mean(np.sqrt(sobelx**2 + sobely**2))
    
    return np.concatenate([means.flatten(), stds.flatten(), hist, [edge_mag]])

@st.cache_data(show_spinner=False)
def load_data(path, limit):
    features, labels, images = [], [], []
    classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    
    for idx, c_name in enumerate(classes):
        c_dir = os.path.join(path, c_name)
        for img_name in os.listdir(c_dir)[:limit]:
            img_path = os.path.join(c_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                features.append(extract_features(img))
                labels.append(float(idx))
    return labels, features, images, classes

def generate_pixel_acc_map(img_rgb, model, train_min, train_max):
    """Generates a spatial classification map for an image."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    res = 40 
    working_img = cv2.resize(img_bgr, (res, res))
    
    win = 7
    pad = win // 2
    padded = cv2.copyMakeBorder(working_img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    
    feats = []
    for y in range(res):
        for x in range(res):
            patch = padded[y:y+win, x:x+win]
            feats.append(extract_features(patch))
    
    # Scaling and converting to pure Python types for libsvm
    fs_np = (np.array(feats) - train_min) / (train_max - train_min + 1e-6)
    fs_list = [list(map(float, row)) for row in fs_np]
    
    preds, _, _ = svm_predict([0.0]*len(fs_list), fs_list, model, '-q')
    
    data = np.array([int(p) for p in preds])
    return cv2.resize(data.reshape((res, res)).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

# --- Execution Logic ---
if st.sidebar.button("🚀 Start Classification", type="primary"):
    if not os.path.exists(dataset_path):
        st.error(f"Dataset folder '{dataset_path}' not found. Ensure the EuroSAT folder is pushed to your GitHub repository.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            k_idx = kernel_options.index(kernel_var)
            
            # 1. Load Data
            status_text.info("Extracting features from satellite imagery...")
            y, x, raw_images, class_names = load_data(dataset_path, int(limit_param))
            progress_bar.progress(20)
            
            # 2. Scale Data
            x_np = np.array(x)
            train_min, train_max = x_np.min(axis=0), x_np.max(axis=0)
            x_scaled = (x_np - train_min) / (train_max - train_min + 1e-6)
            progress_bar.progress(30)
            
            # 3. Split
            x_train, x_test, y_train, y_test, _, img_test = train_test_split(
                x_scaled.tolist(), y, raw_images, test_size=0.2, random_state=42
            )

            x_train = [list(map(float, row)) for row in x_train]
            y_train = [float(i) for i in y_train]
            x_test = [list(map(float, row)) for row in x_test]
            y_test = [float(i) for i in y_test]

            param_str = f'-t {k_idx} -c {c_param} -g {g_param} -q'

            # 4. One-vs-One (OvO)
            status_text.info("Training One-vs-One (OvO) Model...")
            t0 = time.time()
            model_ovo = svm_train(y_train, x_train, param_str)
            p_labs_ovo, p_acc_ovo, _ = svm_predict(y_test, x_test, model_ovo, '-q')
            time_ovo = time.time() - t0
            progress_bar.progress(60)

            # 5. One-vs-Rest (OvR)
            status_text.info("Training One-vs-Rest (OvR) Models...")
            t1 = time.time()
            u_labs = np.unique(y)
            ovr_scores = []
            
            for lab in u_labs:
                binary_y = [1.0 if item == lab else -1.0 for item in y_train]
                m = svm_train(binary_y, x_train, param_str)
                _, _, p_vals = svm_predict(y_test, x_test, m, '-q')
                lbl_idx = 0 if m.get_labels()[0] == 1.0 else 1
                ovr_scores.append([v[lbl_idx] for v in p_vals])
            
            p_labs_ovr = u_labs[np.argmax(ovr_scores, axis=0)]
            acc_ovr = accuracy_score(y_test, p_labs_ovr) * 100
            time_ovr = time.time() - t1
            progress_bar.progress(85)

            # 6. Generate Classified Maps
            status_text.info("Generating Spatial Classification Maps...")
            indices = np.random.choice(len(img_test), 3, replace=False)
            spatial_maps = [generate_pixel_acc_map(img_test[idx], model_ovo, train_min, train_max) for idx in indices]
            progress_bar.progress(100)
            status_text.success("Classification Complete!")

            # --- Display Results ---
            st.markdown("### 📊 Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("OvO Accuracy", f"{p_acc_ovo[0]:.2f}%")
            col2.metric("OvR Accuracy", f"{acc_ovr:.2f}%")
            col3.metric("Time Difference", f"OvO: {time_ovo:.2f}s | OvR: {time_ovr:.2f}s")

            st.markdown("---")
            
            # Plotting Metrics
            fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.bar(['OvO', 'OvR'], [p_acc_ovo[0], acc_ovr], color=['#2ecc71', '#3498db'])
            ax1.set_title("Accuracy (%)")
            ax1.set_ylim(0, 100)
            
            ax2.bar(['OvO', 'OvR'], [time_ovo, time_ovr], color=['#e67e22', '#e74c3c'])
            ax2.set_title("Execution Time (sec)")
            
            cm = confusion_matrix(y_test, p_labs_ovo)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(class_names)))
            disp.plot(ax=ax3, cmap='Greens', colorbar=False)
            ax3.set_title("Confusion Matrix (OvO)")
            
            plt.tight_layout()
            st.pyplot(fig1)

            # Plotting Spatial Maps
            st.markdown("### 🗺️ Sample Classified Maps")
            fig2, axes = plt.subplots(3, 2, figsize=(8, 12))
            num_classes = len(class_names)
            cmap_classes = plt.get_cmap('tab10', num_classes) 
            
            for i, idx in enumerate(indices):
                t_idx, p_idx = int(y_test[idx]), int(p_labs_ovo[idx])
                color = 'green' if t_idx == p_idx else 'red'
                
                axes[i, 0].imshow(img_test[idx])
                axes[i, 0].set_title(f"True: {class_names[t_idx]}\nPred: {class_names[p_idx]}", color=color)
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(spatial_maps[i], cmap=cmap_classes, vmin=0, vmax=num_classes-1)
                axes[i, 1].set_title("Classification Map")
                axes[i, 1].axis('off')
                
            legend_elements = [Line2D([0], [0], marker='s', color='w', label=class_names[j],
                                     markerfacecolor=cmap_classes(j), markersize=10) 
                               for j in range(num_classes)]
            
            fig2.legend(handles=legend_elements, loc='center right', title="Legend", bbox_to_anchor=(1.15, 0.5))
            fig2.subplots_adjust(right=0.85, hspace=0.4) 
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
