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
            st.write("Jumlah data tiap kelas:")
            st.write(df['type'].value_counts())
            st.write("Jumlah nilai hilang per kolom:")
            st.write(df.isnull().sum())

            st.subheader("Struktur Dataset")
            buffer = []
            df.info(buf=buffer.append)
            st.text('\n'.join(buffer))

        elif selected == "Klasifikasi Naive Bayes":
            st.header("📦 Klasifikasi dengan Naive Bayes")
            with st.spinner("Sedang menghitung..."):
                accuracy, prediction_df = naive_bayes_predict(df)
            st.success(f"Akurasi Model: {accuracy:.2%}")
            st.dataframe(prediction_df.head(10))

        elif selected == "Uji Coba":
            st.header("🧪 Uji Coba Prediksi Manual")
            features = df.columns[:-1]
            input_data = {}
            for feature in features:
                input_data[feature] = st.selectbox(f"{feature}", [0, 1])

            _, prediction_df = naive_bayes_predict(df)

            # Gunakan satu baris untuk prediksi
            input_row = pd.DataFrame([input_data])
            prediction = prediction_df.columns[-1]  # Ambil kolom "Prediksi" sebagai label
            st.info(f"Prediksi (simulasi berdasarkan data latih): **{prediction_df['Prediksi'].mode()[0]}**")
    else:
        st.warning("Silakan unggah file CSV terlebih dahulu melalui sidebar.")

if __name__ == "__main__":
    main()
