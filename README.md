# Superstore Intelligence Engine

This repository contains the source code for a comprehensive, end-to-end machine learning platform designed to provide customer intelligence and operational insights for an e-commerce business. The project demonstrates a full MLOps workflow, from automated EDA and data preparation to advanced modeling, synthetic data generation, and final deployment in an interactive web application.

---

## 🚀 Live Demo

**[https://superstore-intelligence-v2.streamlit.app/](https://superstore-intelligence-v2.streamlit.app/)**

---

## 🏛️ Project Architecture

This platform is architected as a modular, script-based MLOps project, emphasizing reproducibility, efficiency, and clear separation of concerns.


**Key Components:**

1.  **Automated Data Pipelines (`src/`):**
    *   **`comprehensive_eda_report.py`**: A command-line tool that generates a full diagnostic and business intelligence report on the raw data.
    *   **`data_preparation.py`**: A production pipeline that cleans, validates, and enriches the raw data, outputting a high-integrity `.parquet` file.
    *   **`generate_synthetic_data.py`**: A GPU-accelerated pipeline that trains a `CTGAN` model to generate a large, statistically validated synthetic dataset for data augmentation.
    *   **`train_final_models.py`**: The master training script that engineers advanced features on the augmented dataset and produces the final, tuned predictive models.
    *   **`train_advanced_segmentation.py`**: A script to train the Autoencoder model for advanced customer segmentation.
    *   **`train_predictive_mba.py`**: A script to train the innovative feature-based classifier for market basket analysis.

2.  **ML Model Suite (`models/`):**
    *   **Profitability Classifier:** An XGBoost model that predicts if a transaction will be profitable with **95.8% accuracy**.
    *   **Profit Forecaster:** An XGBoost model that predicts the profit of a transaction with **83.2% R-squared**.
    *   **Predictive MBA Classifier:** An innovative XGBoost model to predict product co-purchase likelihood, solving the "cold start" problem.
    *   **Advanced Segmentation Encoder:** A Keras/TensorFlow Autoencoder for intelligent customer embedding.

3.  **Interactive Application (`app.py`):**
    *   A multi-page Streamlit dashboard that serves as the front-end for the entire platform, providing access to AI-generated personas, the "Deal Desk" forecaster, and the "AI Strategy Co-Pilot."

---

## 🔧 Key Technologies & Libraries

*   **Data Processing:** Pandas, NumPy
*   **Machine Learning:** Scikit-learn, XGBoost, TensorFlow/Keras
*   **Synthetic Data:** SDV (`CTGAN`)
*   **Interpretability:** SHAP
*   **Generative AI:** Google Generative AI (Gemini)
*   **Application:** Streamlit
*   **Environment:** `uv` for package management

---

## ⚙️ Setup and Execution

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # macOS/Linux
    # .venv\Scripts\activate  # Windows
    ```

3.  **Install Dependencies:**
    This project uses `uv` for fast package installation.
    ```bash
    pip install uv
    uv pip install -r requirements.txt
    ```

4.  **Run the MLOps Pipeline (Optional):**
    The pre-trained models are included, but you can re-run the entire pipeline.
    ```bash
    # 1. Generate clean data
    python src/data_preparation.py
    
    # 2. Generate synthetic data (requires GPU environment)
    python src/generate_synthetic_data.py
    
    # 3. Train all final models
    python src/train_final_models.py
    python src/train_advanced_segmentation.py
    python src/train_predictive_mba.py
    ```

5.  **Run the Streamlit Application:**
    Ensure you have your Google AI API key set in `.streamlit/secrets.toml`.
    ```bash
    streamlit run app.py
    ```

---

## ❓ Architectural FAQ

For a deep dive into the key architectural decisions, strategic pivots, and technical justifications made during this project, please see the detailed **[Architectural FAQ](./docs/FAQ.md)**.