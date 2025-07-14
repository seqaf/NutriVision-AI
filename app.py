import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Fungsi untuk memuat model dan vectorizer ---
@st.cache_resource
def load_ml_assets():
    """Memuat TF-IDF Vectorizer dan model Linear Regression."""
    try:
        tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        model_calories = joblib.load('models/linear_regression_calories.pkl')
        model_proteins = joblib.load('models/linear_regression_proteins.pkl')
        model_fat = joblib.load('models/linear_regression_fat.pkl')
        model_carbohydrate = joblib.load('models/linear_regression_carbohydrate.pkl')
        return tfidf_vectorizer, model_calories, model_proteins, model_fat, model_carbohydrate
    except FileNotFoundError:
        st.error("Model atau TF-IDF Vectorizer tidak ditemukan. Pastikan Anda telah menjalankan 'train_models.py' dan file-file .pkl ada di folder 'models'.")
        return None, None, None, None, None

# --- Fungsi untuk memuat dataset (untuk menampilkan daftar makanan) ---
@st.cache_data
def load_food_data(file_path):
    """Memuat dataset Food_Nutrition.csv."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Clean column names
    df['name'] = df['name'].str.lower()  # Convert names to lowercase
    return df

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="NutriVision AI", page_icon="🍎", layout="wide")

# --- Header Aplikasi ---
st.markdown("""
<style>
.header {
    background-color: #43cea2;
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.header h1 {
    margin: 0;
    font-size: 2.5rem;
}
.header h3 {
    margin: 0;
    font-size: 1.5rem;
}
</style>
<div class="header">
    <h1>🍎 NutriVision AI</h1>
    <h3>Prediksi Kandungan Gizi Makanan dengan Machine Learning</h3>
</div>
""", unsafe_allow_html=True)

# --- Deskripsi Aplikasi ---
st.write("""
Aplikasi ini memprediksi kandungan **Kalori, Protein, Lemak, dan Karbohidrat** pada makanan
berdasarkan nama makanan yang Anda masukkan. Prediksi dilakukan menggunakan model **Regresi Linier**
yang dilatih dengan fitur TF-IDF dari nama makanan pada dataset Food Nutrition.
""")

# --- Memuat aset ML ---
tfidf_vectorizer, model_calories, model_proteins, model_fat, model_carbohydrate = load_ml_assets()

# Memuat data makanan untuk referensi
df_food = load_food_data("Food_Nutrition.csv")

if tfidf_vectorizer and model_calories:  # Check if models are loaded successfully
    # --- Input Pengguna untuk Prediksi ---
    st.header("🔍 Masukkan Nama Makanan untuk Prediksi")
    col1, col2 = st.columns([3, 1])  # Bagi layout menjadi dua kolom

    with col1:
        food_name_input = st.text_input("Nama Makanan (contoh: Nasi, Ayam Goreng, Apel)", key="food_input")
    with col2:
        # Input berat makanan, default 100 gram
        food_weight_input = st.number_input("Berat (gram)", min_value=1, value=100, step=1, key="food_weight")

    # --- Proses Prediksi ---
    if st.button("Prediksi Kandungan Gizi"):
        if food_name_input:
            # Pra-pemrosesan input
            processed_food_name = food_name_input.lower()

            # Transformasi input menggunakan TF-IDF Vectorizer yang sudah dilatih
            try:
                input_vector = tfidf_vectorizer.transform([processed_food_name])
            except ValueError as e:
                st.error(f"Kesalahan transformasi input: {e}. Mungkin nama makanan terlalu pendek atau tidak relevan dengan vocabulary model.")
                st.info("Coba masukkan nama makanan yang lebih umum atau periksa ejaan.")
                input_vector = None  # Set to None to skip prediction

            if input_vector is not None:
                # Lakukan prediksi untuk setiap kandungan gizi
                pred_calories_per_100g = model_calories.predict(input_vector)[0]
                pred_proteins_per_100g = model_proteins.predict(input_vector)[0]
                pred_fat_per_100g = model_fat.predict(input_vector)[0]
                pred_carbohydrate_per_100g = model_carbohydrate.predict(input_vector)[0]

                # Pastikan prediksi tidak negatif
                pred_calories_per_100g = np.maximum(0, pred_calories_per_100g)
                pred_proteins_per_100g = np.maximum(0, pred_proteins_per_100g)
                pred_fat_per_100g = np.maximum(0, pred_fat_per_100g)
                pred_carbohydrate_per_100g = np.maximum(0, pred_carbohydrate_per_100g)

                # Hitung total kandungan gizi berdasarkan berat yang dimasukkan
                factor = food_weight_input / 100.0
                total_calories = pred_calories_per_100g * factor
                total_proteins = pred_proteins_per_100g * factor
                total_fat = pred_fat_per_100g * factor
                total_carbohydrate = pred_carbohydrate_per_100g * factor

                st.success(f"Prediksi Kandungan Gizi untuk '{food_name_input}' ({food_weight_input} gram):")
                st.write(f"**Kalori:** {total_calories:.2f} kcal")
                st.write(f"**Protein:** {total_proteins:.2f} g")
                st.write(f"**Lemak:** {total_fat:.2f} g")
                st.write(f"**Karbohidrat:** {total_carbohydrate:.2f} g")

                # Cari data aktual jika makanan ada di dataset
                actual_data = df_food[df_food['name'] == processed_food_name]
                if not actual_data.empty:
                    st.markdown("---")
                    st.subheader(f"Detail Makanan '{food_name_input.title()}'")
                    
                    # Tampilkan gambar dalam ukuran lebih besar (400px)
                    if 'image' in actual_data.columns and pd.notna(actual_data['image'].iloc[0]):
                        st.image(actual_data['image'].iloc[0], width=400)
                    
                    # Tampilkan informasi nutrisi dalam bentuk tab untuk tampilan lebih rapi
                    tab1, tab2 = st.tabs(["Prediksi Model", "Data Aktual"])
                    
                    with tab1:
                        st.write("**Prediksi Model per 100 gram:**")
                        st.write(f"- Kalori: {pred_calories_per_100g:.2f} kcal")
                        st.write(f"- Protein: {pred_proteins_per_100g:.2f} g")
                        st.write(f"- Lemak: {pred_fat_per_100g:.2f} g")
                        st.write(f"- Karbohidrat: {pred_carbohydrate_per_100g:.2f} g")
                    
                    with tab2:
                        st.write("**Data Aktual:**")
                        st.write(f"- Kalori: {actual_data['calories'].iloc[0]} kcal")
                        st.write(f"- Protein: {actual_data['proteins'].iloc[0]} g")
                        st.write(f"- Lemak: {actual_data['fat'].iloc[0]} g")
                        st.write(f"- Karbohidrat: {actual_data['carbohydrate'].iloc[0]} g")

        else:
            st.warning("Harap masukkan nama makanan terlebih dahulu.")

    # --- Daftar Makanan dengan Navigasi yang Lebih Baik ---
    st.header("📋 Daftar Makanan")
    
    # Tambahkan fitur pencarian dan filter
    search_col, filter_col = st.columns(2)
    
    with search_col:
        search_query = st.text_input("Cari nama makanan...", key="search_food").lower()
    
    with filter_col:
        # Filter berdasarkan range kalori
        min_cal, max_cal = st.slider(
            "Filter berdasarkan kalori (per 100g)",
            min_value=int(df_food['calories'].min()),
            max_value=int(df_food['calories'].max()),
            value=(0, int(df_food['calories'].max()))
        )
    
    # Terapkan filter
    filtered_df = df_food.copy()
    if search_query:
        filtered_df = filtered_df[filtered_df['name'].str.contains(search_query, na=False)]
    
    filtered_df = filtered_df[
        (filtered_df['calories'] >= min_cal) & 
        (filtered_df['calories'] <= max_cal)
    ]
    
    # Tampilkan jumlah hasil
    st.write(f"Menampilkan {len(filtered_df)} dari {len(df_food)} makanan")
    
    # Tampilkan hasil dalam tabel tanpa kolom image
    if not filtered_df.empty:
        st.dataframe(
            filtered_df[['name', 'calories', 'proteins', 'fat', 'carbohydrate']],
            column_config={
                "name": "Nama Makanan",
                "calories": "Kalori (kcal)",
                "proteins": "Protein (g)",
                "fat": "Lemak (g)",
                "carbohydrate": "Karbohidrat (g)"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("Tidak ada makanan yang cocok dengan kriteria pencarian")

    # --- Tab untuk Makanan Tertinggi ---
    st.header("🏆 Makanan Tertinggi Berdasarkan Nutrisi")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Kalori Tertinggi", "Protein Tertinggi", "Lemak Tertinggi", "Karbohidrat Tertinggi"])
    
    with tab1:
        top_calories = df_food.sort_values('calories', ascending=False).head(10)
        st.write(top_calories[['name', 'calories']].set_index('name'))

    with tab2:
        top_proteins = df_food.sort_values('proteins', ascending=False).head(10)
        st.write(top_proteins[['name', 'proteins']].set_index('name'))

    with tab3:
        top_fats = df_food.sort_values('fat', ascending=False).head(10)
        st.write(top_fats[['name', 'fat']].set_index('name'))

    with tab4:
        top_carbs = df_food.sort_values('carbohydrate', ascending=False).head(10)
        st.write(top_carbs[['name', 'carbohydrate']].set_index('name'))

else:
    st.warning("Aplikasi tidak dapat berjalan karena model atau vectorizer tidak berhasil dimuat. Harap periksa pesan error di atas.")
