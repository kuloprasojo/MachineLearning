import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

def naive_bayes_predict(df):
    # Split dataset
    train_size = int(0.8 * len(df))
    train_data = df[:train_size]
    test_data = df[train_size:]

    # Feature and label
    features = df.columns[:-1]
    labels = df['type'].unique()

    # Prior probability
    prior_prob = train_data['type'].value_counts() / len(train_data)

    # Likelihood
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

    # Evaluasi akurasi
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

    return accuracy, test_data

def main():
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
            st.write("Jumlah data tiap kelas:")
            st.write(df['type'].value_counts())
            st.write("Jumlah nilai hilang per kolom:")
            st.write(df.isnull().sum())

            st.subheader("Struktur Dataset")
            buffer = []
            df.info(buf=buffer.append)
            st.text('\n'.join(buffer))

        elif selected == "Uji Coba":
            st.header("🧪 Uji Coba Prediksi Manual")
            features = df.columns[:-1]
            input_data = {}
        
            with st.form("form_uji_coba"):
                for feature in features:
                    pilihan = st.radio(f"Apakah Anda mengalami '{feature}'?", ["Tidak", "Ya"], horizontal=True)
                    input_data[feature] = 1 if pilihan == "Ya" else 0
        
                submitted = st.form_submit_button("🔍 Submit Prediksi")
        
            if submitted:
                # Siapkan model kembali
                train_size = int(0.8 * len(df))
                train_data = df[:train_size]
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
        
                input_row = pd.DataFrame([input_data])
                result = predict(input_row.iloc[0])
                st.success(f"Hasil Prediksi Berdasarkan Input Anda: **{result}**")
        
    else:
        st.warning("Silakan unggah file CSV terlebih dahulu melalui sidebar.")

if __name__ == "__main__":
    main()
