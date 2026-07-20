import os
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from libsvm.svmutil import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D

class EuroSAT_Ultimate_SVM:
    def __init__(self, root):
        self.root = root
        self.root.title("GNR602: SVM Classifier")
        self.root.geometry("450x600")

        tk.Label(root, text="SVM Parameter Tuning", font=('Arial', 12, 'bold')).pack(pady=10)

        # Kernel Selection
        tk.Label(root, text="Kernel:").pack()
        self.kernel_var = tk.StringVar(value="RBF")
        ttk.Combobox(root, textvariable=self.kernel_var, values=["Linear", "Polynomial", "RBF", "Sigmoid"], state="readonly").pack(pady=5)

        # C Parameter
        tk.Label(root, text="Slack Parameter (C):").pack()
        self.c_param = tk.Entry(root)
        self.c_param.insert(0, "1000")
        self.c_param.pack(pady=5)

        # Gamma Parameter 
        tk.Label(root, text="Gamma (g):").pack()
        self.g_param = tk.Entry(root)
        self.g_param.insert(0, "0.1") 
        self.g_param.pack(pady=5)

        # Images per Class
        tk.Label(root, text="Images per Class:").pack()
        self.limit_param = tk.Entry(root)
        self.limit_param.insert(0, "500") 
        self.limit_param.pack(pady=5)

        self.status_var = tk.StringVar(value="Status: Ready")
        tk.Label(root, textvariable=self.status_var, fg="darkblue").pack(pady=20)

        self.btn = tk.Button(root, text="Select Folder & Start Classification", command=self.start_thread, bg="#28a745", fg="white", font=('Arial', 10, 'bold'), height=2)
        self.btn.pack(pady=10)

        # Storage for scaling and model
        self.train_min = None
        self.train_max = None

    def extract_features(self, img):
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

    def generate_pixel_acc_map(self, img_rgb, model):
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
                feats.append(self.extract_features(patch))
        
        # Scaling and converting to pure Python types for libsvm
        fs_np = (np.array(feats) - self.train_min) / (self.train_max - self.train_min + 1e-6)
        fs_list = [list(map(float, row)) for row in fs_np]
        
        preds, _, _ = svm_predict([0.0]*len(fs_list), fs_list, model, '-q')
        
        data = np.array([int(p) for p in preds])
        return cv2.resize(data.reshape((res, res)).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

    def start_thread(self):
        folder = filedialog.askdirectory()
        if not folder: return
        self.btn.config(state="disabled")
        threading.Thread(target=self.run_analysis, args=(folder,), daemon=True).start()

    def run_analysis(self, folder):
        try:
            c = float(self.c_param.get())
            g = float(self.g_param.get())
            limit = int(self.limit_param.get())
            k_idx = {"Linear": 0, "Polynomial": 1, "RBF": 2, "Sigmoid": 3}[self.kernel_var.get()]

            # 1. Load Data
            y, x, raw_images, class_names = self.load_data(folder, limit)
            
            # 2. Scale Data
            x_np = np.array(x)
            self.train_min, self.train_max = x_np.min(axis=0), x_np.max(axis=0)
            x_scaled = (x_np - self.train_min) / (self.train_max - self.train_min + 1e-6)
            
            # 3. Split
            x_train, x_test, y_train, y_test, _, img_test = train_test_split(
                x_scaled.tolist(), y, raw_images, test_size=0.2, random_state=42
            )

            # Ensure native Python types for libsvm
            x_train = [list(map(float, row)) for row in x_train]
            y_train = [float(i) for i in y_train]
            x_test = [list(map(float, row)) for row in x_test]
            y_test = [float(i) for i in y_test]

            param_str = f'-t {k_idx} -c {c} -g {g} -q'

            # 4. One-vs-One (OvO) - Default libsvm behavior
            self.status_var.set("Training One-vs-One...")
            t0 = time.time()
            model_ovo = svm_train(y_train, x_train, param_str)
            p_labs_ovo, p_acc_ovo, _ = svm_predict(y_test, x_test, model_ovo)
            time_ovo = time.time() - t0

            # 5. One-vs-Rest (OvR) - Manual implementation
            self.status_var.set("Training One-vs-Rest...")
            t1 = time.time()
            u_labs = np.unique(y)
            ovr_scores = []
            
            for lab in u_labs:
                # Binary labels: 1 for current class, -1 for all others
                binary_y = [1.0 if item == lab else -1.0 for item in y_train]
                m = svm_train(binary_y, x_train, param_str)
                
                # Get decision values
                _, _, p_vals = svm_predict(y_test, x_test, m, '-q')
                
                # Check model label order to ensure we pick the correct column for class '1.0'
                lbl_idx = 0 if m.get_labels()[0] == 1.0 else 1
                ovr_scores.append([v[lbl_idx] for v in p_vals])
            
            p_labs_ovr = u_labs[np.argmax(ovr_scores, axis=0)]
            acc_ovr = accuracy_score(y_test, p_labs_ovr) * 100
            time_ovr = time.time() - t1

            # 6. Generate Classified Maps for 3 Random Samples 
            self.status_var.set("Generating Classified Sample Maps...")
            indices = np.random.choice(len(img_test), 3, replace=False)
            spatial_maps = [self.generate_pixel_acc_map(img_test[idx], model_ovo) for idx in indices]

            self.status_var.set("Success!")
            self.root.after(0, lambda: self.show_results(
                p_acc_ovo[0], acc_ovr, time_ovo, time_ovr, 
                y_test, p_labs_ovo, img_test, class_names, indices, spatial_maps
            ))
            print(f"\n--- Final Comparison ---")
            print(f"One-vs-One Accuracy: {p_acc_ovo[0]:.2f}%")
            print(f"One-vs-Rest Accuracy: {acc_ovr:.2f}%")
            print(f"OvO Time: {time_ovo:.4f}s | OvR Time: {time_ovr:.4f}s\n")

        except Exception as e:
            self.status_var.set("Error!")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
        finally:
            self.btn.config(state="normal")

    def load_data(self, path, limit):
        features, labels, images = [], [], []
        classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
        for idx, c_name in enumerate(classes):
            self.status_var.set(f"Extracting: {c_name}")
            self.root.update_idletasks()
            c_dir = os.path.join(path, c_name)
            for img_name in os.listdir(c_dir)[:limit]:
                img_path = os.path.join(c_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    features.append(self.extract_features(img))
                    labels.append(float(idx))
        return labels, features, images, classes

    def show_results(self, acc_ovo, acc_ovr, t_ovo, t_ovr, y_true, pred_ovo, img_test, class_names, indices, spatial_maps):
        # Result Window 1: Metrics
        res_win = tk.Toplevel(self.root)
        res_win.title("Performance Metrics")
        
        fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.bar(['OvO', 'OvR'], [acc_ovo, acc_ovr], color=['#2ecc71', '#3498db'])
        ax1.set_title("Accuracy (%)")
        ax1.set_ylim(0, 100)
        
        ax2.bar(['OvO', 'OvR'], [t_ovo, t_ovr], color=['#e67e22', '#e74c3c'])
        ax2.set_title("Execution Time (sec)")
        
        cm = confusion_matrix(y_true, pred_ovo)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(class_names)))
        disp.plot(ax=ax3, cmap='Greens', colorbar=False)
        ax3.set_title("Confusion Matrix (OvO)")
        
        plt.tight_layout()
        FigureCanvasTkAgg(fig1, master=res_win).get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Result Window 2: Spatial Analysis
        pred_win = tk.Toplevel(self.root)
        pred_win.title("Sample Classified Maps")
        
        fig2, axes = plt.subplots(3, 2, figsize=(10, 10))
        num_classes = len(class_names)
        cmap_classes = plt.get_cmap('tab10', num_classes) 
        
        for i, idx in enumerate(indices):
            t_idx, p_idx = int(y_true[idx]), int(pred_ovo[idx])
            color = 'green' if t_idx == p_idx else 'red'
            
            axes[i, 0].imshow(img_test[idx])
            axes[i, 0].set_title(f"True: {class_names[t_idx]}\nPred: {class_names[p_idx]}", color=color)
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(spatial_maps[i], cmap=cmap_classes, vmin=0, vmax=num_classes-1)
            axes[i, 1].set_title("Classification Map")
            axes[i, 1].axis('off')
            
        legend_elements = [Line2D([0], [0], marker='s', color='w', label=class_names[j],
                                 markerfacecolor=cmap_classes(j), markersize=12) 
                           for j in range(num_classes)]
        
        fig2.legend(handles=legend_elements, loc='center right', title="Legend")
        fig2.subplots_adjust(right=0.8, hspace=0.4) 
        FigureCanvasTkAgg(fig2, master=pred_win).get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = EuroSAT_Ultimate_SVM(root)
    root.mainloop()