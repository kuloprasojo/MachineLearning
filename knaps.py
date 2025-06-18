import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from streamlit_option_menu import option_menu
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def naive_bayes_predict(df):
    train_size = int(0.8 * len(df))
    train_data = df[:train_size]
    test_data = df[train_size:]

    features = df.columns[:-1]
    labels = df['type'].unique()

    prior_prob = train_data['type'].value_counts() / len(train_data)

    likelihoods = {}
    for label in labels:
        likelihoods[label] = {}
        subset = train_data[train_data['type'] == label]
        for feature in features:
            likelihoods[label][feature] = {
                0: (subset[feature] == 0).sum() / len(subset),
                1: (subset[feature] == 1).sum() / len(subset)
            }

    def predict(row):
        probabilities = {}
        for label in labels:
            prob = prior_prob[label]
            for feature in features:
                prob *= likelihoods[label][feature].get(row[feature], 1e-6)
            probabilities[label] = prob
        return max(probabilities, key=probabilities.get)

    correct = 0
    total = len(test_data)
    predictions = []

    for _, row in test_data.iterrows():
        pred = predict(row)
        predictions.append(pred)
        if pred == row['type']:
            correct += 1

    accuracy = correct / total
    test_data = test_data.copy()
    test_data['Prediksi'] = predictions

    return accuracy, test_data, predict, prior_prob, likelihoods, labels, features

def main():
    st.set_page_config(page_title="Naive Bayes Classifier", layout="wide")

    with st.sidebar:
        selected = option_menu(
            "Main Menu",
            ["Home", "Informasi Dataset", "Klasifikasi Naive Bayes", "Uji Coba"], 
            icons=['house', 'table', 'boxes', 'check2-circle'], 
            menu_icon="cast",
            default_index=0,
            orientation='vertical'
        )
        upload_file = st.file_uploader("Masukkan file CSV di sini", key="fileUploader")

    if upload_file is not None:
        df = pd.read_csv(upload_file)

        if selected == "Home":
            st.title("📊 Naive Bayes Classifier")
            st.markdown("Selamat datang di aplikasi klasifikasi **Naive Bayes** berbasis gejala penyakit.")
            st.success("Dataset berhasil dimuat!")
            st.dataframe(df.head())

        elif selected == "Informasi Dataset":
            st.header("📁 Informasi Dataset")
            st.write(f"Dataset berisi **{df.shape[0]} baris** dan **{df.shape[1]} kolom**.")
            st.write("Jumlah data tiap kelas:")
            st.write(df['type'].value_counts())
            st.write("Jumlah nilai hilang per kolom:")
            st.write(df.isnull().sum())

            st.subheader("Struktur Dataset")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.code(buffer.getvalue(), language='text')

        elif selected == "Klasifikasi Naive Bayes":
            st.header("📦 Klasifikasi dengan Naive Bayes")
            with st.spinner("Sedang menghitung..."):
                accuracy, prediction_df, _, _, _, _, _ = naive_bayes_predict(df)

            st.success(f"Akurasi Model: {accuracy:.2%}")

            st.subheader("📊 Distribusi Hasil Prediksi")
            pred_count = prediction_df['Prediksi'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=pred_count.index, y=pred_count.values, ax=ax, palette="Set2")
            ax.set_title("Jumlah Prediksi per Kategori")
            ax.set_ylabel("Jumlah")
            ax.set_xlabel("Label Prediksi")
            st.pyplot(fig)

            st.subheader("🧮 Confusion Matrix")
            cm = confusion_matrix(prediction_df['type'], prediction_df['Prediksi'], labels=prediction_df['type'].unique())
            fig_cm, ax_cm = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=prediction_df['type'].unique())
            disp.plot(ax=ax_cm, cmap='Blues')
            st.pyplot(fig_cm)

            st.subheader("Contoh Hasil Prediksi")
            st.dataframe(prediction_df.head(10))

            # Tombol unduh hasil
            csv = prediction_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(f'<a href="data:file/csv;base64,{b64}" download="hasil_prediksi.csv">📥 Unduh Hasil sebagai CSV</a>', unsafe_allow_html=True)

        elif selected == "Uji Coba":
            st.header("🧪 Uji Coba Prediksi Manual")
            accuracy, _, predict_func, prior_prob, likelihoods, labels, features = naive_bayes_predict(df)

            input_data = {}
            with st.form("form_uji_coba"):
                for feature in features:
                    pilihan = st.radio(f"Apakah Anda mengalami '{feature}'?", ["Tidak", "Ya"], horizontal=True)
                    input_data[feature] = 1 if pilihan == "Ya" else 0
                submitted = st.form_submit_button("🔍 Submit Prediksi")

            if submitted:
                input_row = pd.DataFrame([input_data])

                # Hitung probabilitas semua label
                probabilities = {}
                for label in labels:
                    prob = prior_prob[label]
                    for feature in features:
                        prob *= likelihoods[label][feature].get(input_row.iloc[0][feature], 1e-6)
                    probabilities[label] = prob

                sorted_probs = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))
                result = max(sorted_probs, key=sorted_probs.get)
                st.success(f"Hasil Prediksi Berdasarkan Input Anda: **{result}**")

                st.subheader("📊 Probabilitas Tiap Label")
                prob_df = pd.DataFrame(sorted_probs.items(), columns=["Label", "Probabilitas"])
                st.dataframe(prob_df)

                fig_prob, ax_prob = plt.subplots()
                sns.barplot(x="Probabilitas", y="Label", data=prob_df, palette="viridis", ax=ax_prob)
                ax_prob.set_title("Distribusi Probabilitas Prediksi")
                st.pyplot(fig_prob)

    else:
        st.warning("Silakan unggah file CSV terlebih dahulu melalui sidebar.")

if __name__ == "__main__":
    main()
