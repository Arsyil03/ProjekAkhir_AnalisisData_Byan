import streamlit as st
import pandas as pd
import plotly.express as px 
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(__file__)
data_path = os.path.join(BASE_DIR, "main_data.csv")
df_air_quality = pd.read_csv(data_path)

df_air_quality["year"] = pd.to_datetime(df_air_quality["year"], format='%Y').dt.year  # Convert to int
df_air_quality.fillna(df_air_quality.select_dtypes(include=['number']).mean(numeric_only=True), inplace=True)

st.set_page_config(
    page_title="Dashboard Suhu & Polutan Kota",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Flag_of_the_People%27s_Republic_of_China.svg/200px-Flag_of_the_People%27s_Republic_of_China.svg.png",
                 use_column_width=True)

st.sidebar.title("\U0001F3D9️ Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["\U0001F4CA Data", "\U0001F4C8 Visualisasi", "\U0001F52C Analisis Lanjutan"])

st.sidebar.subheader("Filter Tahun")
selected_year = st.sidebar.selectbox("Pilih Tahun", sorted(df_air_quality["year"].unique()))
df_filtered = df_air_quality[df_air_quality["year"] == selected_year]

city_stats = df_filtered.groupby("City")[["TEMP", "PM2.5"]].mean().reset_index()
city_stats = city_stats.sort_values(by="TEMP", ascending=False)

st.title("\U0001F30D Dashboard Suhu & Polutan Kota di Negara China (2013-2017)")
st.markdown("---")

if not city_stats.empty:
    hottest_city = city_stats.iloc[0]
    coldest_city = city_stats.iloc[-1]

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="\U0001F321️ Kota Terpanas", 
                  value=hottest_city["City"], 
                  delta=f"{hottest_city['TEMP']:.2f}°C")
    with col2:
        st.metric(label="\u2744️ Kota Terdingin", 
                  value=coldest_city["City"], 
                  delta=f"{coldest_city['TEMP']:.2f}°C")
    
    st.markdown("---")

    if page == "\U0001F4CA Data":
        st.subheader("Data Suhu dan Polutan Rata-rata Tiap Kota")
        st.dataframe(city_stats.style.format({"TEMP": "{:.2f}°C", "PM2.5": "{:.2f} µg/m³"}))
    
    elif page == "\U0001F4C8 Visualisasi":
        st.subheader("Grafik Suhu Rata-rata per Kota")
        fig = px.bar(city_stats, x="TEMP", y="City", 
                     title=f"Urutan Kota dengan Suhu Rata-rata ({selected_year})",
                     labels={"TEMP": "Suhu Rata-rata (°C)", "City": "Kota"},
                     color="TEMP", color_continuous_scale="Blues",
                     orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Distribusi Polutan PM2.5")
        top_polluted_cities = city_stats.head(12)
        
        fig, ax = plt.subplots(figsize=(4, 4)) 
        wedges, texts, autotexts = ax.pie(
            top_polluted_cities["PM2.5"],
            labels=top_polluted_cities["City"],
            autopct="%1.1f%%",
            colors=sns.color_palette("coolwarm", len(top_polluted_cities)),
            pctdistance=0.5, 
            labeldistance=1.2 
        )
        for autotext in autotexts:
            autotext.set_color('white') 
        ax.set_title("Distribusi Polutan PM2.5 di 12 Kota dengan Polusi Tertinggi (2013-2017)")
        st.pyplot(fig)
        
        st.subheader("Kota dengan Polutan PM2.5 Tertinggi Tiap Tahun")
        city_max_pm = df_air_quality.loc[df_air_quality.groupby("year")["PM2.5"].idxmax()]
        city_max_pm = city_max_pm.sort_values(by="year")
        
        fig_pm = px.bar(city_max_pm, x="year", y="PM2.5", color="City",
                        title="Kota dengan Polutan PM2.5 Tertinggi Tiap Tahun",
                        labels={"PM2.5": "Konsentrasi PM2.5 (µg/m³)", "year": "Tahun"},
                        color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig_pm, use_container_width=True)
        
        col1, col2 = st.columns(2)
        df_2013_2017 = df_air_quality[(df_air_quality["year"] >= 2013) & (df_air_quality["year"] <= 2017)]
        city_stats_all_years = df_2013_2017.groupby("City")["TEMP"].mean().reset_index()
        
        with col1:
            fig_max_temp = px.bar(city_stats_all_years.nlargest(12, "TEMP"), x="TEMP", y="City", 
                                  title="Top 12 Kota dengan Suhu Tertinggi Sepanjang Tahun",
                                  labels={"TEMP": "Suhu Maksimum (°C)", "City": "Kota"},
                                  color="TEMP", color_continuous_scale="Reds",
                                  orientation="h")
            fig_max_temp.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig_max_temp, use_container_width=True)

        with col2:
            fig_min_temp = px.bar(city_stats_all_years.nsmallest(12, "TEMP"), x="TEMP", y="City", 
                                  title="Top 12 Kota dengan Suhu Terendah Sepanjang Tahun",
                                  labels={"TEMP": "Suhu Minimum (°C)", "City": "Kota"},
                                  color="TEMP", color_continuous_scale="Blues",
                                  orientation="h")
            fig_min_temp.update_yaxes(categoryorder="total descending")
            st.plotly_chart(fig_min_temp, use_container_width=True)

if page == "\U0001F52C Analisis Lanjutan":
    st.title("\U0001F52C Analisis Lanjutan - Clustering")
    
    # Pilih fitur suhu dan polutan
    features = ["TEMP", "PM2.5", "PM10", "NO2"]
    df_cluster = df_air_quality[features].dropna()
    
    # Normalisasi data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cluster)
    
    # Tentukan jumlah cluster dengan Elbow Method
    inertia = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)
    
    st.subheader("Elbow Method untuk Menentukan Jumlah Cluster")
    fig, ax = plt.subplots()
    ax.plot(range(1, 10), inertia, marker='o')
    ax.set_xlabel('Jumlah Cluster')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method')
    st.pyplot(fig)
    
    # Gunakan jumlah cluster optimal (misal k=3 berdasarkan elbow method)
    k_optimal = 3  # Sesuaikan dengan hasil elbow method
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    df_air_quality["Cluster"] = kmeans.fit_predict(df_scaled)
    
    st.subheader("Visualisasi Hasil Clustering")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_air_quality, x="TEMP", y="PM2.5", hue="Cluster", palette="viridis", ax=ax)
    ax.set_title("Clustering berdasarkan Suhu dan Polutan")
    st.pyplot(fig)