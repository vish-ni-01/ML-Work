# imports
import streamlit as st
import numpy as np
import joblib
import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# page config
st.set_page_config(page_title="Iris ML Comparison", layout="wide")

# load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Iris.csv")

df = load_data()

# load models
perceptron_model = joblib.load("perceptron_model.pkl")
ann_model = load_model("ann_model.h5")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

with open("results.json") as f:
    results = json.load(f)

# create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Make Prediction", "Model Comparison", "About"])


# =====================================================
# ======================= HOME ========================
# =====================================================
with tab1:

    st.markdown("## 📊 Project Overview")
    st.write("""
    This project compares a traditional Machine Learning model (Perceptron)
    with a Deep Learning model (Artificial Neural Network)
    on the Iris dataset.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", "150")
    col2.metric("Features", "4")
    col3.metric("Classes", "3")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔹 Perceptron")
        st.write("""
        - Linear classifier  
        - Implemented using Scikit-learn  
        - Suitable for linearly separable data  
        """)

    with col2:
        st.subheader("🔹 Artificial Neural Network")
        st.write("""
        - Fully Connected Dense Network  
        - ReLU activation (Hidden Layer)  
        - Softmax activation (Output Layer)  
        - Optimizer: Adam  
        - Loss: Categorical Crossentropy  
        """)

    st.markdown("---")
    st.info("🎯 Objective: Compare classical ML vs Deep Learning on structured data.")


# =====================================================
# ================== MAKE PREDICTION ==================
# =====================================================
with tab2:

    st.header("🔮 Make a Prediction")

    input_mode = st.radio(
        "Choose Input Method",
        ["Manual Input", "Upload CSV File"],
        horizontal=True
    )

    if input_mode == "Manual Input":

        col1, col2 = st.columns(2)

        with col1:
            sl = st.slider("Sepal Length", 4.0, 8.0, 5.1)
            sw = st.slider("Sepal Width", 2.0, 4.5, 3.5)

        with col2:
            pl = st.slider("Petal Length", 1.0, 7.0, 1.4)
            pw = st.slider("Petal Width", 0.1, 2.5, 0.2)

        model_choice = st.radio("Select Model", ["Perceptron", "ANN"], horizontal=True)

        input_data = np.array([[sl, sw, pl, pw]])
        input_scaled = scaler.transform(input_data)

        if st.button("🚀 Predict", use_container_width=True):

            if model_choice == "Perceptron":
                pred = perceptron_model.predict(input_scaled)
            else:
                pred_probs = ann_model.predict(input_scaled)
                pred = np.argmax(pred_probs, axis=1)

            class_name = label_encoder.inverse_transform(pred)
            st.success(f"🌼 Predicted Flower: **{class_name[0]}**")

    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        model_choice = st.radio("Select Model", ["Perceptron", "ANN"], horizontal=True)

        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)

            st.write("### 📄 Uploaded Data Preview")
            st.dataframe(df_upload.head(), use_container_width=True)

            try:
                input_scaled = scaler.transform(df_upload)

                if model_choice == "Perceptron":
                    predictions = perceptron_model.predict(input_scaled)
                else:
                    pred_probs = ann_model.predict(input_scaled)
                    predictions = np.argmax(pred_probs, axis=1)

                class_names = label_encoder.inverse_transform(predictions)
                df_upload["Predicted_Class"] = class_names

                st.write("### 📊 Predictions")
                st.dataframe(df_upload, use_container_width=True)

                csv = df_upload.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "⬇ Download Predictions",
                    csv,
                    "iris_predictions.csv",
                    "text/csv",
                    use_container_width=True
                )

            except:
                st.error("CSV must contain exactly 4 feature columns in correct order.")


# =====================================================
# ================= MODEL COMPARISON ==================
# =====================================================
with tab3:

    st.header("📊 Model Performance Comparison")

    col1, col2 = st.columns(2)

    col1.metric(
        "Perceptron Accuracy",
        f"{results['Perceptron Accuracy']*100:.2f}%"
    )

    col2.metric(
        "ANN Accuracy",
        f"{results['ANN Accuracy']*100:.2f}%"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Perceptron Confusion Matrix")
        st.image("confusion_perceptron.png", use_container_width=True)

    with col2:
        st.subheader("ANN Confusion Matrix")
        st.image("confusion_ann.png", use_container_width=True)

    st.info("ANN performs better because it captures non-linear patterns.")


# =====================================================
# ======================= ABOUT =======================
# =====================================================
with tab4:

    st.header("📘 About This Project")

    # ================= DATASET OVERVIEW =================
    st.markdown("## 🌸 Dataset Overview")
    st.write("""
    The Iris dataset is a classic multi-class classification dataset.
    It contains 150 flower samples divided into three species:
    Setosa, Versicolor, and Virginica.
    """)

    st.markdown("## 🗂 Dataset Details")
    st.write("""
    - Total Samples: 150  
    - Features:
        • Sepal Length  
        • Sepal Width  
        • Petal Length  
        • Petal Width  
    - Classes:
        • Setosa  
        • Versicolor  
        • Virginica  
    """)

    # ================= DATA PREVIEW =================
    st.markdown("### 📄 First 5 Rows of Dataset")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### 📊 Dataset Shape")
    st.write(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

    # ================= SUMMARY STATS =================
    st.markdown("### 📈 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    # ================= PREPROCESSING =================
    st.markdown("## ⚙️ Data Preprocessing")
    st.write("""
    - Train-Test Split performed  
    - StandardScaler used for feature scaling  
    - Label Encoding applied  
    - One-Hot Encoding used for ANN training  
    """)

    # ================= MODEL INFO =================
    st.markdown("## 🤖 Models Used")

    st.subheader("🔹 Perceptron (Scikit-learn)")
    st.write("""
    - Type: Linear Classifier  
    - Library: Scikit-learn  
    - Learns a linear decision boundary  
    - Suitable for linearly separable data  
    """)

    st.subheader("🔹 Artificial Neural Network (TensorFlow/Keras)")
    st.write("""
    - Type: Deep Learning Model  
    - Architecture: Fully Connected Dense Network  
    - Hidden Layer Activation: ReLU  
    - Output Layer Activation: Softmax  
    - Optimizer Used: Adam  
    - Loss Function: Categorical Crossentropy  
    - Target Encoding: One-Hot Encoding  
    """)

    st.markdown("### 🏗 ANN Architecture Summary")
    st.code("""
Input Layer: 4 neurons
Hidden Layer: Dense (ReLU activation)
Output Layer: 3 neurons (Softmax)
Optimizer: Adam
Loss Function: Categorical Crossentropy
""")

    # ================= PAIRPLOT =================
    st.markdown("## 📊 Pairplot Visualization")

    @st.cache_data 
    def generate_pairplot(data): 
        return sns.pairplot(data, hue="Species") 
    if st.button("Show Pairplot"): 
        with st.spinner("Generating Pairplot..."): 
            pairplot = generate_pairplot(df)
        st.pyplot(pairplot.figure)

    # ================= HEATMAP =================
    st.markdown("## 🔥 Correlation Heatmap")

    if st.button("Show Heatmap"):
        with st.spinner("Generating Heatmap..."):
            plt.figure(figsize=(8,6))
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
            st.pyplot(plt)

    # ================= LEARNING OUTCOME =================
    st.markdown("## 🎯 Learning Outcome")
    st.write("""
    This project demonstrates:

    • Difference between classical ML and Deep Learning  
    • Importance of preprocessing and encoding  
    • Model evaluation techniques  
    • How non-linear models can outperform linear models  
    • Deployment using Streamlit  
    """)

    # ================= CREATOR =================
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center;">
            <h3>👩‍💻 Created by Avni Singh</h3>
            <p>
                🔗 <a href="https://github.com/avnisinngh" target="_blank">
                Visit My GitHub</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ================= GLOBAL FOOTER =================
st.markdown("---")
st.caption("© 2026 Avni Singh | Iris Classification using ANN & Perceptron")
