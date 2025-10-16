import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import google.generativeai as genai
from itertools import combinations
from pathlib import Path

st.set_page_config(
    page_title="Superstore Intelligence Engine v2.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ARCHITECTURAL NOTE: Cached Asset Loading ---
@st.cache_data
def load_data():
    df = pd.read_parquet('data/processed/cleaned_superstore_data.parquet')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    return df

@st.cache_resource
def load_models():
    models_path = Path('models')
    models = {
        "v3_classifier": joblib.load(models_path / "final_v3_profitability_classifier.joblib"),
        "v3_forecaster": joblib.load(models_path / "final_v3_profit_forecaster.joblib"),
        "mba_model": joblib.load(models_path / "final_predictive_mba_classifier.joblib")
    }
    return models

@st.cache_data
def create_feature_lookups(_df):
    df_with_margin = _df.copy()
    if 'Profit_Margin' not in df_with_margin.columns:
        df_with_margin['Profit_Margin'] = np.where(df_with_margin['Total Sales'] > 0, df_with_margin['Total Profit'] / df_with_margin['Total Sales'], 0)

    lookups = {
        "postal_code_profitability": df_with_margin.groupby('Postal Code')['Profit_Margin'].mean().to_dict(),
        "state_sales_volume": df_with_margin.groupby('State')['Total Sales'].sum().to_dict(),
        "subcategory_avg_margin": df_with_margin.groupby('Sub-Category')['Profit_Margin'].mean().to_dict(),
        "category_avg_discount": df_with_margin.groupby('Category')['Discount'].mean().to_dict(),
        "customer_avg_order_size": df_with_margin.groupby('Customer ID')['Total Sales'].mean().to_dict(),
        "product_features": df_with_margin.groupby('Product Name').agg(
            Category=('Category', 'first'),
            SubCategory=('Sub-Category', 'first'),
            AvgSalesPrice=('Sales Price', 'mean')
        ).reset_index()
    }
    return lookups

# --- Generative AI Configuration ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    GEMINI_MODEL = genai.GenerativeModel('gemma-3-27b-it')
except Exception:
    GEMINI_MODEL = None

# --- Persona Content ---
PERSONAS = {
    "🏆 Super Loyalists (Top 33%)": {
        "stats": {"Recency": "116.0 days", "Frequency": "8.4 orders", "Monetary Value": "$6,585.2"},
        "narrative": """ ... """ # Paste your full persona markdown here
    },
    "💰 High Spenders (Middle 33%)": {
        "stats": {"Recency": "134.1 days", "Frequency": "5.8 orders", "Monetary Value": "$3,334.4"},
        "narrative": """ ... """ # Paste your full persona markdown here
    },
    "😴 Dormant Customers (Bottom 33%)": {
        "stats": {"Recency": "234.8 days", "Frequency": "3.4 orders", "Monetary Value": "$1,676.9"},
        "narrative": """ ... """ # Paste your full persona markdown here
    }
}

# --- On-the-fly Feature Engineering Function ---
def create_live_features(df_input, lookups, base_df):
    df = df_input.copy()
    df['month_of_year'] = df['Order Date'].dt.month

    customer_id = df['Customer ID'].iloc[0]
    if customer_id == "New Customer":
        df['days_since_first_order'] = 0
        df['customer_avg_order_size'] = base_df['Total Sales'].mean()
    else:
        first_order_date = base_df[base_df['Customer ID'] == customer_id]['Order Date'].min()
        df['days_since_first_order'] = (df['Order Date'] - first_order_date).dt.days.iloc[0]
        df['customer_avg_order_size'] = lookups['customer_avg_order_size'].get(customer_id, base_df['Total Sales'].mean())

    df['postal_code_profitability'] = lookups['postal_code_profitability'].get(df['Postal Code'].iloc[0], base_df['Profit_Margin'].mean())
    df['state_sales_volume'] = lookups['state_sales_volume'].get(df['State'].iloc[0], base_df['Total Sales'].mean())
    df['subcategory_avg_margin'] = lookups['subcategory_avg_margin'].get(df['Sub-Category'].iloc[0], base_df['Profit_Margin'].mean())
    df['category_avg_discount'] = lookups['category_avg_discount'].get(df['Category'].iloc[0], base_df['Discount'].mean())
    
    return df

# --- PAGE RENDERING FUNCTIONS ---
def render_home_page(df):
    st.title("Superstore Intelligence Engine v2.0")
    st.markdown("An integrated MLOps platform for customer intelligence, operational forecasting, and strategic decision support.")
    
    st.subheader("Key Performance Indicators")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"${df['Total Sales'].sum():,.0f}")
    col2.metric("Total Profit", f"${df['Total Profit'].sum():,.0f}")
    col3.metric("Total Unique Customers", f"{df['Customer ID'].nunique()}")

    st.subheader("Profitability by Category")
    fig, ax = plt.subplots(); df.groupby('Category')['Total Profit'].sum().sort_values().plot(kind='barh', ax=ax); st.pyplot(fig)

def render_customer_segmentation_page():
    st.title("Customer Segment Explorer")
    st.markdown("Explore AI-generated personas for customer segments discovered by our advanced deep learning model.")
    
    selected_persona = st.selectbox("Select a persona:", list(PERSONAS.keys()))
    if selected_persona:
        p_data = PERSONAS[selected_persona]
        c1, c2, c3 = st.columns(3); c1.metric("Avg. Recency", p_data["stats"]["Recency"]); c2.metric("Avg. Frequency", p_data["stats"]["Frequency"]); c3.metric("Avg. Monetary Value", p_data["stats"]["Monetary Value"])
        st.markdown("---"); st.markdown(p_data["narrative"])

def render_prediction_suite_page(df, models, lookups):
    st.title("Prediction Suite: The Deal Desk Forecaster")
    st.markdown("Use our definitive V3 models to analyze transaction profitability. These models are trained on an augmented dataset with advanced features for superior accuracy.")

    c1, c2 = st.columns(2)
    with c1:
        segment = st.selectbox("Customer Segment", df['Segment'].unique()); region = st.selectbox("Region", df['Region'].unique()); state = st.selectbox("State", sorted(df[df['Region'] == region]['State'].unique())); category = st.selectbox("Product Category", df['Category'].unique()); sub_category = st.selectbox("Product Sub-Category", sorted(df[df['Category'] == category]['Sub-Category'].unique()))
    with c2:
        price_stats = df[df['Sub-Category'] == sub_category]['Sales Price'].describe()
        sales_price = st.number_input(f"Sales Price ($)", min_value=0.0, value=round(price_stats['mean'], 2), step=0.01, format="%.2f")
        quantity = st.number_input("Quantity", min_value=1, value=2, step=1); discount = st.slider("Discount (%)", min_value=0, max_value=100, value=10) / 100.0
        postal_code = st.selectbox("Postal Code", sorted(df[df['State'] == state]['Postal Code'].unique()))
        customer_id = st.selectbox("Select Existing Customer (for historical context)", ["New Customer"] + sorted(list(df['Customer ID'].unique())))

    if st.button("Analyze Profitability"):
        raw_input = {'Segment': segment, 'Region': region, 'State': state, 'Category': category, 'Sub-Category': sub_category, 'Sales Price': sales_price, 'Quantity': quantity, 'Discount': discount, 'Postal Code': postal_code, 'Customer ID': customer_id, 'Order Date': pd.Timestamp.now()}
        input_df_featured = create_live_features(pd.DataFrame([raw_input]), lookups, df)
        
        classifier = models["v3_classifier"]; forecaster = models["v3_forecaster"]
        
        pred = classifier.predict(input_df_featured)[0]; proba = classifier.predict_proba(input_df_featured)[0]
        
        st.subheader("Profitability Analysis")
        if pred == 1:
            st.success(f"**Prediction: Profitable** (Confidence: {proba[1]:.2%})")
            log_profit_pred = forecaster.predict(input_df_featured)[0]
            profit_pred = np.expm1(log_profit_pred)
            total_sales = sales_price * quantity
            margin_pred = profit_pred / total_sales if total_sales > 0 else 0
            st.info(f"**Predicted Total Profit: ${profit_pred:.2f}**"); st.info(f"**Implied Profit Margin: {margin_pred:.2%}**")
        else:
            st.error(f"**Prediction: Not Profitable** (Confidence: {proba[0]:.2%})")

def render_recommender_page(df, models, lookups):
    st.title("Predictive Market Basket")
    st.markdown("This tool uses a feature-based classifier to predict the likelihood of two products being purchased together.")
    product_list = sorted(df['Product Name'].unique())
    c1, c2 = st.columns(2)
    product_a = c1.selectbox("Select Product A:", product_list, index=100)
    product_b = c2.selectbox("Select Product B:", product_list, index=200)
        
    if st.button("Predict Co-Purchase Likelihood"):
        if product_a == product_b:
            st.warning("Please select two different products.")
        else:
            product_features = lookups['product_features']
            features_a = product_features[product_features['Product Name'] == product_a].iloc[0]
            features_b = product_features[product_features['Product Name'] == product_b].iloc[0]
            input_data = pd.DataFrame([{'A_Category': features_a['Category'], 'A_SubCategory': features_a['SubCategory'], 'A_AvgSalesPrice': features_a['AvgSalesPrice'], 'B_Category': features_b['Category'], 'B_SubCategory': features_b['SubCategory'], 'B_AvgSalesPrice': features_b['AvgSalesPrice']}])
            proba = models['mba_model'].predict_proba(input_data)[0][1]
            st.subheader("Co-Purchase Analysis")
            st.info(f"The predicted probability of these two items being purchased together is **{proba:.2%}**.")

def render_strategic_insights_page(df, models):
    st.title("Strategic Insights Dashboard")
    st.subheader("Key Drivers of Profit (SHAP Analysis)")
    st.markdown("This plot explains the predictions of our V3 Profit Forecaster, showing which factors have the biggest impact on profit.")
    
    if st.button("Generate SHAP Plot (computationally intensive)"):
        with st.spinner("Calculating SHAP values... Please wait."):
            model = models["v3_forecaster"]
            preprocessor = model.named_steps['preprocessor']
            explainer_model = model.named_steps['regressor']
            
            # Use a sample of the data for faster SHAP calculation in the app
            df_sample = df.sample(n=1000, random_state=1)
            master_df_sample = create_master_feature_set(df_sample)
            
            features = [
                'Segment', 'Region', 'State', 'Category', 'Sub-Category', 'Quantity', 
                'Sales Price', 'Discount', 'month_of_year', 'days_since_first_order',
                'postal_code_profitability', 'state_sales_volume', 'subcategory_avg_margin',
                'category_avg_discount', 'customer_avg_order_size'
            ]
            X_for_shap = master_df_sample[features]

            X_transformed = preprocessor.transform(X_for_shap)
            
            explainer = shap.TreeExplainer(explainer_model)
            shap_values = explainer(X_transformed)
            
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_for_shap, show=False, plot_type="bar")
            plt.title("SHAP Feature Importance for V3 Profit Forecaster")
            st.pyplot(fig)

    st.markdown("---")
    st.subheader("AI Strategy Co-Pilot")
    st.markdown("Generate a strategic memo based on the project's key findings.")
    
    output_format = st.selectbox("Select Output Format:", ["Memo to CEO", "Marketing Strategy Brief"])
    
    if st.button("Generate AI Strategy Memo"):
        if not GEMINI_MODEL:
            st.error("Generative AI model is not configured. Please check your API key.")
            return

        with st.spinner("Generating strategic memo..."):
            prompt = create_ceo_memo_prompt()
            response = GEMINI_MODEL.generate_content(prompt)
            st.markdown(response.text)

def create_master_feature_set(df):
    logging.info("Engineering the V3 master feature set...")
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df.dropna(subset=['Order Date'], inplace=True)

    if 'Profit_Margin' not in df.columns:
        df['Profit_Margin'] = np.where(df['Total Sales'] > 0, df['Total Profit'] / df['Total Sales'], 0)

    df['month_of_year'] = df['Order Date'].dt.month
    df['first_order_date'] = df.groupby('Customer ID')['Order Date'].transform('min')
    df['days_since_first_order'] = (df['Order Date'] - df['first_order_date']).dt.days

    df['postal_code_profitability'] = df.groupby('Postal Code')['Profit_Margin'].transform('mean')
    df['state_sales_volume'] = df.groupby('State')['Total Sales'].transform('sum')
    df['subcategory_avg_margin'] = df.groupby('Sub-Category')['Profit_Margin'].transform('mean')
    df['category_avg_discount'] = df.groupby('Category')['Discount'].transform('mean')
    df['customer_avg_order_size'] = df.groupby('Customer ID')['Total Sales'].transform('mean')
    
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

def create_ceo_memo_prompt():
    # This function encapsulates our master prompt
    # In a real app, you could add checkboxes to select which findings to include
    return """
    ## ROLE AND GOAL
    You are a Chief Data Scientist reporting to the CEO. Your goal is to synthesize key findings from our "Customer Intelligence Platform" into a concise, strategic memo.
    
    ## CONTEXT
    [---Briefing Document---]
    1. Customer Segmentation: We have three segments: 'Super Loyalists' (Top 33%, high value), 'High Spenders' (Mid 33%, growth potential), and 'Dormant Customers' (Bottom 33%, churn risk).
    2. Profitability Drivers: 'Binders' and the 'Central' region consistently hurt margins. 'Home Office' segment and high 'Quantity' deals are inherently profitable.
    3. Predictive Capabilities: We have a 'Deal Desk' classifier (94% accurate) to predict if a deal is profitable, and a 'Margin Forecaster' (95% R-squared) to predict the margin.
    4. Feasibility Study: We proved long-term value cannot be predicted from early behavior.
    
    ## TASK
    Draft a formal, one-page strategic memo to the CEO with:
    1. EXECUTIVE SUMMARY
    2. KEY FINDING 1: WHO OUR CUSTOMERS ARE
    3. KEY FINDING 2: WHAT DRIVES OUR PROFITABILITY
    4. NEW CAPABILITIES & RECOMMENDATIONS
    5. CONCLUSION
    
    ## TONE
    Formal, strategic, data-driven, and concise.
    """


# --- Main App Logic ---
def main():
    df = load_data()
    models = load_models()
    lookups = create_feature_lookups(df)
    
    st.sidebar.title("Navigation")
    pages = {
        "Home": lambda: render_home_page(df),
        "Customer Segmentation": render_customer_segmentation_page,
        "Prediction Suite": lambda: render_prediction_suite_page(df, models, lookups),
        "Product Recommender": lambda: render_recommender_page(df, models, lookups),
        "Strategic Insights": lambda: render_strategic_insights_page(df, models)
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selection]()

if __name__ == "__main__":
    main()