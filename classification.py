import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import wilcoxon

# Helper: Entropi Sampel
def compute_entropy(prob_matrix):
    return -np.sum(prob_matrix * np.log(prob_matrix + 1e-12), axis=1, keepdims=True)

def run():
    # Judul Utama Web
    st.title("🧠 Dysgraphia Fusion Analysis Dashboard")
    st.write("Dysgraphia screening application based on Entropy Weighted Stacking (EWS) Fusion using sensor and image modality.")

    # Form Upload File di Web
    st.sidebar.header("Upload File (.npy)")
    uploaded_online = st.sidebar.file_uploader("File Online Feature (onlineDB2.npy)", type=["npy"])
    uploaded_offline = st.sidebar.file_uploader("File Offline Feature (offlineDB2_dense201.npy)", type=["npy"])
    uploaded_label = st.sidebar.file_uploader("File Label (yDB2.npy)", type=["npy"])

    # Tombol Eksekusi
    if st.sidebar.button("Run"):
        if uploaded_online and uploaded_offline and uploaded_label:
            # Load Data
            Xo = np.load(uploaded_online)
            Xoff = np.load(uploaded_offline)
            y = np.load(uploaded_label)
            
            st.info("Awaiting for cross-validation (10-Fold) ...")
            
            # Hyperparameter SVM
            params_online = {"C": 4, "gamma": 0.015625, "kernel": "rbf"}
            params_offline = {"C": 100, "gamma": 0.0002752087, "kernel": "rbf"}

            acc_online, acc_offline, acc_fusion = [], [], []
            pre_online, pre_offline, pre_fusion = [], [], []
            rec_online, rec_offline, rec_fusion = [], [], []
            f1_online, f1_offline, f1_fusion = [], [], []

            classes = np.unique(y)
            n_classes = len(classes)
            h_max = np.log(n_classes) if n_classes > 1 else 1.0

            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

            for fold, (train_idx, test_idx) in enumerate(skf.split(Xo, y), 1):
                Xo_train, Xo_test = Xo[train_idx], Xo[test_idx]
                Xoff_train, Xoff_test = Xoff[train_idx], Xoff[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                sc_online = StandardScaler()
                Xo_train = sc_online.fit_transform(Xo_train)
                Xo_test = sc_online.transform(Xo_test)

                sc_offline = StandardScaler()
                Xoff_train = sc_offline.fit_transform(Xoff_train)
                Xoff_test = sc_offline.transform(Xoff_test)

                svm_online = SVC(**params_online, probability=True, random_state=42)
                svm_offline = SVC(**params_offline, probability=True, random_state=42)

                svm_online.fit(Xo_train, y_train)
                svm_offline.fit(Xoff_train, y_train)

                pred_online = svm_online.predict(Xo_test)
                pred_offline = svm_offline.predict(Xoff_test)

                acc_online.append(accuracy_score(y_test, pred_online))
                acc_offline.append(accuracy_score(y_test, pred_offline))
                pre_online.append(precision_score(y_test, pred_online, average="macro", zero_division=0))
                pre_offline.append(precision_score(y_test, pred_offline, average="macro", zero_division=0))
                rec_online.append(recall_score(y_test, pred_online, average="macro", zero_division=0))
                rec_offline.append(recall_score(y_test, pred_offline, average="macro", zero_division=0))
                f1_online.append(f1_score(y_test, pred_online, average="macro", zero_division=0))
                f1_offline.append(f1_score(y_test, pred_offline, average="macro", zero_division=0))

                prob_online_train = svm_online.predict_proba(Xo_train)
                prob_online_test = svm_online.predict_proba(Xo_test)
                prob_offline_train = svm_offline.predict_proba(Xoff_train)
                prob_offline_test = svm_offline.predict_proba(Xoff_test)

                e_on_tr = compute_entropy(prob_online_train) / h_max
                e_off_tr = compute_entropy(prob_offline_train) / h_max
                e_on_te = compute_entropy(prob_online_test) / h_max
                e_off_te = compute_entropy(prob_offline_test) / h_max

                inv_on_tr = 1.0 - e_on_tr
                inv_off_tr = 1.0 - e_off_tr
                inv_on_te = 1.0 - e_on_te
                inv_off_te = 1.0 - e_off_te

                s_tr = inv_on_tr + inv_off_tr + 1e-2
                w_ent_on_tr = inv_on_tr / s_tr
                w_ent_off_tr = inv_off_tr / s_tr

                s_te = inv_on_te + inv_off_te + 1e-2
                w_ent_on_te = inv_on_te / s_te
                w_ent_off_te = inv_off_te / s_te

                meta_train_ent = np.hstack([prob_online_train * w_ent_on_tr, prob_offline_train * w_ent_off_tr])
                meta_test_ent = np.hstack([prob_online_test * w_ent_on_te, prob_offline_test * w_ent_off_te])

                meta_clf_ent = SVC(kernel="poly", degree=3, coef0=2.0, C=0.01, probability=True, random_state=42)
                meta_clf_ent.fit(meta_train_ent, y_train)

                pred_ent = meta_clf_ent.predict(meta_test_ent)

                acc_fusion.append(accuracy_score(y_test, pred_ent))
                pre_fusion.append(precision_score(y_test, pred_ent, average="macro", zero_division=0))
                rec_fusion.append(recall_score(y_test, pred_ent, average="macro", zero_division=0))
                f1_fusion.append(f1_score(y_test, pred_ent, average="macro", zero_division=0))

            # Tampilkan Tabel Hasil Performa
            st.subheader("Report")
            summary_data = {
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Sensor-Derived (Online)': [np.mean(acc_online), np.mean(pre_online), np.mean(rec_online), np.mean(f1_online)],
                'Deep Embedding (Offline)': [np.mean(acc_offline), np.mean(pre_offline), np.mean(rec_offline), np.mean(f1_offline)],
                'Fusion Framework': [np.mean(acc_fusion), np.mean(pre_fusion), np.mean(rec_fusion), np.mean(f1_fusion)]
            }
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary)

            # Tampilkan Uji Statistik Wilcoxon
            st.subheader("📈 Statistical Analysis (Wilcoxon Signed-Rank Test)")
            metrics_dict = {
                'Accuracy': (acc_fusion, acc_online, acc_offline),
                'Precision': (pre_fusion, pre_online, pre_offline),
                'Recall': (rec_fusion, rec_online, rec_offline),
                'F1-Score': (f1_fusion, f1_online, f1_offline)
            }

            results_stat = []
            for metric_name, (fusion_vec, online_vec, offline_vec) in metrics_dict.items():
                stat_on, p_val_on = wilcoxon(fusion_vec, online_vec)
                stat_off, p_val_off = wilcoxon(fusion_vec, offline_vec)

                results_stat.append({
                    'Metric': metric_name,
                    'Stat (vs Online)': stat_on,
                    'p-value (vs Online)': p_val_on,
                    'Sig. (vs Online)': 'Ya (p<0.05)' if p_val_on < 0.05 else 'Tidak',
                    'Stat (vs Offline)': stat_off,
                    'p-value (vs Offline)': p_val_off,
                    'Sig. (vs Offline)': 'Ya (p<0.05)' if p_val_off < 0.05 else 'Tidak'
                })

            df_stat = pd.DataFrame(results_stat)
            st.dataframe(df_stat)
        else:
            st.warning("Upload ketiga file `.npy` terlebih dahulu di bilah samping (sidebar).")

# Baris ini memungkinkan file dijalankan secara mandiri
if __name__ == "__main__":
    run()