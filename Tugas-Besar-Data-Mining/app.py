import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# --- Judul Dashboard ---
st.set_page_config(page_title="Analisis Kendaraan Listrik", layout="wide")
st.title("🚗📊 Dashboard Analisis Data Kendaraan Listrik")

# --- Load Dataset ---
st.header("1. Data Understanding")
df = pd.read_csv("data/customer_experience_data.csv")
st.write("Contoh 5 data teratas:")
st.dataframe(df.head())
st.write("Statistik Deskriptif:")
st.dataframe(df.describe())

# --- Histogram Distribusi ---
st.header("2. Histogram Distribusi Data")
num_cols = df.select_dtypes(include='number').columns.tolist()
col_hist = st.selectbox("Pilih kolom untuk histogram:", num_cols)
fig1, ax1 = plt.subplots()
sns.histplot(df[col_hist], kde=True, ax=ax1)
ax1.set_title(f"Distribusi: {col_hist}")
st.pyplot(fig1)

# --- Boxplot ---
st.header("3. Boxplot untuk Deteksi Outlier")
col_box = st.selectbox("Pilih kolom untuk boxplot:", num_cols, key="box")
fig2, ax2 = plt.subplots()
sns.boxplot(x=df[col_box], ax=ax2)
ax2.set_title(f"Boxplot: {col_box}")
st.pyplot(fig2)

# --- K-Means Clustering ---
st.header("4. K-Means Clustering")
cluster_cols = st.multiselect("Pilih kolom numerik untuk clustering:", num_cols, default=num_cols[:2])
n_cluster = st.slider("Jumlah cluster:", 2, 10, 3)

if len(cluster_cols) >= 2:
    X = df[cluster_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    df_cluster = X.copy()
    df_cluster['Cluster'] = labels

    st.subheader("Hasil Clustering (contoh data):")
    st.dataframe(df_cluster.head())

    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels, palette='Set2', ax=ax3)
    ax3.set_xlabel(cluster_cols[0])
    ax3.set_ylabel(cluster_cols[1])
    ax3.set_title("Visualisasi 2D Cluster")
    st.pyplot(fig3)

# --- Logistic Regression ---
st.header("5. Klasifikasi Logistic Regression")
target = st.selectbox("Pilih kolom target (biner):", df.columns)
features = st.multiselect("Pilih fitur numerik:", num_cols)

if target and features:
    X = df[features].dropna()
    y = df[target].loc[X.index]  # cocokkan index y dengan X

    if len(y.unique()) == 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Hasil Evaluasi Model")
        st.text(classification_report(y_test, y_pred))
    else:
        st.warning("Kolom target harus berupa 2 kelas (biner)!")
