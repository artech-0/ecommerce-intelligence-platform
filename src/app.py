# app.py (Final Corrected Version)

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

# --- Asset Loading Functions ---
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
    lookups = {
        "postal_code_profitability": _df.groupby('Postal Code')['Profit_Margin'].mean().to_dict(),
        "state_sales_volume": _df.groupby('State')['Total Sales'].sum().to_dict(),
        "subcategory_avg_margin": _df.groupby('Sub-Category')['Profit_Margin'].mean().to_dict(),
        "category_avg_discount": _df.groupby('Category')['Discount'].mean().to_dict(),
        "customer_avg_order_size": _df.groupby('Customer ID')['Total Sales'].mean().to_dict(),
        "product_features": _df.groupby('Product Name').agg(Category=('Category', 'first'), SubCategory=('Sub-Category', 'first'), AvgSalesPrice=('Sales Price', 'mean')).reset_index()
    }
    return lookups

# --- Generative AI Configuration ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    GEMINI_MODEL = genai.GenerativeModel('gemma-3-27b-it')
except (AttributeError, KeyError):
    GEMINI_MODEL = None
v3_numerical_features = [
    'Sales Price', 'Quantity', 'Discount', 'month_of_year', 'days_since_first_order',
    'postal_code_profitability', 'state_sales_volume', 'subcategory_avg_margin',
    'category_avg_discount', 'customer_avg_order_size'
]
v3_categorical_features = ['Segment', 'Region', 'State', 'Category', 'Sub-Category']
# --- Persona Content (FULL VERSION) ---
PERSONAS = {
    "🏆 Super Loyalists (Top 33%)": {
        "stats": {"Recency": "116.0 days", "Frequency": "8.4 orders", "Monetary Value": "$6,585.2"},
        "narrative": """
### 1. Persona Title
🏆 **"The Established Loyalist" (Rachel)**

### 2. Narrative Summary
Meet Rachel. She represents the bedrock of our customer base—a long-standing, high-value patron who has consistently chosen our brand for significant purchases over the years. As a busy professional in her late 30s or 40s with high disposable income, Rachel values quality, durability, and a brand she can trust. While her deep loyalty is proven by her purchase history, it has been nearly four months since her last order, indicating she may be between buying cycles. Her engagement is crucial for our long-term success.

### 3. Key Attributes
- **High Lifetime Value:** Total spend over $6,500.
- **Brand Loyal & Repeat Purchaser:** An average of 8.4 orders demonstrates a clear preference for our brand.
- **High Average Order Value (AOV):** Her average order is substantial (approx. $784), indicating a focus on premium items.
- **Infrequent Purchaser:** A recency of 116 days suggests a long buying cycle, possibly due to the nature of the products purchased.

### 4. Motivations & Needs
- **Seeks Quality and Durability:** Motivated by long-term value, not short-term deals.
- **Needs Trust and Reliability:** Requires deep trust in our product, service, and delivery.
- **Appreciates Recognition:** Expects a level of service that matches her investment in the brand.
- **Values a Seamless Experience:** Needs a frictionless purchasing process.

### 5. Actionable Marketing Strategies
- **Personalized Re-engagement:** Send targeted emails highlighting "What's New Since Your Last Visit," featuring products complementary to her past purchases.
- **Implement a VIP/Loyalty Tier:** Automatically enroll this segment into a top tier with exclusive benefits like early access to new collections or a dedicated customer service line.
- **High-Touch Content Marketing:** Send exclusive content that reinforces their smart purchase decisions, like care guides or behind-the-scenes looks at craftsmanship.
- **Solicit High-Value Feedback:** Invite her to a "customer advisory panel" to get feedback on potential new products, reinforcing her importance.
        """
    },
    "💰 High Spenders (Middle 33%)": {
        "stats": {"Recency": "134.1 days", "Frequency": "5.8 orders", "Monetary Value": "$3,334.4"},
        "narrative": """
### 1. Persona Title
💰 **"The Selective Investor" (Mark)**

### 2. Narrative Summary
Meet Mark. He is a discerning customer who invests significantly when he has a specific need. Representing our middle-tier, Mark is willing to spend over $3,000, but his engagement is less frequent. His purchasing cycle of over four months suggests he is a project-based or event-driven buyer. While he contributes major revenue, his loyalty is not guaranteed. The key challenge is to bridge the long gaps between his purchases and build a more consistent relationship.

### 3. Key Attributes
- **Significant Lifetime Value (LTV):** Total spend exceeds $3,300, making a substantial impact on revenue.
- **High Average Order Value (AOV):** Spends approximately $575 per order, showing a preference for high-quality items.
- **Moderately Frequent Purchaser:** Has made several repeat purchases (5.8 orders) but is not a deeply embedded loyalist.
- **Currently Dormant:** A 134-day recency is a critical flag; he is at risk of being acquired by a competitor.

### 4. Motivations & Needs
- **Solution-Oriented:** Shops to solve a specific, often large-scale problem (e.g., furnishing a room, upgrading tech).
- **Driven by Research and Value:** Needs strong social proof, detailed product specs, and clear value propositions.
- **Needs a Compelling Reason to Return:** His purchasing habit is not automatic and needs to be prompted.
- **Trust in Quality:** Willing to spend more for products that will last.

### 5. Actionable Marketing Strategies
- **"Next Project" Themed Campaigns:** Create content that inspires their next purchase (e.g., if they bought a desk, target them with content about "Building the Perfect Home Office").
- **Strategic "Bounce-Back" Offer:** After a large purchase, send a time-sensitive, high-value offer to incentivize a follow-up purchase and shorten the recency period.
- **Promote Cross-Category Discovery:** Use targeted marketing to introduce him to other relevant product categories.
- **Re-engage with High-Impact Product Launches:** Use this segment as a primary audience for announcing major new products, framing them as a worthy investment.
        """
    },
    "😴 Dormant Customers (Bottom 33%)": {
        "stats": {"Recency": "234.8 days", "Frequency": "3.4 orders", "Monetary Value": "$1,676.9"},
        "narrative": """
### 1. Persona Title
😴 **"The Forgotten Patron" (Priya)**

### 2. Narrative Summary
Meet Priya. She represents a significant and concerning segment—a once-promising patron who has since gone silent. Priya initially made 3-4 purchases, spending over $1,600, but it's now been over seven months since her last interaction. For all intents and purposes, she has churned. Re-engaging Priya is a high-effort, high-reward challenge, but winning her back requires a powerful, strategic intervention.

### 3. Key Attributes
- **Deeply Lapsed/Churned:** An average recency of 235 days means this segment is no longer actively considering our brand.
- **Proven Historical Value:** A lifetime spend of nearly $1,700 makes them too valuable to ignore completely.
- **Past Multi-Purchaser:** An average of 3.4 orders shows they were once satisfied enough to return.
- **High Average Order Value (AOV):** Their average order value was substantial (approx. $493).

### 4. Motivations & Needs
- **Needs a Strong Reason to Return:** Standard marketing is insufficient. They require a compelling incentive.
- **Lack of Top-of-Mind Awareness:** Our brand has fallen off their radar.
- **Potentially Dissatisfied or Indifferent:** Their dormancy could stem from a negative experience or a competitor meeting their needs better.
- **Requires Re-Proof of Value:** We need to prove our brand is still the best choice through innovation or a superior value proposition.

### 5. Actionable Marketing Strategies
- **Execute a High-Impact Win-Back Campaign:** Deploy a multi-channel campaign with a steep offer (e.g., "25% off to welcome you back") and messaging that explicitly acknowledges their absence.
- **Launch a Feedback-Oriented Survey:** Instead of a sales pitch, send an email asking, "Where did we go wrong?" with a strong incentive for completion. This provides invaluable churn data.
- **"Last Chance" Drip Campaign:** For non-responders, an automated series highlighting major brand improvements since their last visit can help rebuild trust.
- **Isolate and Suppress:** If a user remains unengaged, suppress them from costly marketing to avoid wasting spend.
        """
    }
}

# --- On-the-fly Feature Engineering ---
def create_live_features(df_input, lookups, base_df):
    df = df_input.copy()
    df['month_of_year'] = df['Order Date'].dt.month
    customer_id = df['Customer ID'].iloc[0]
    if customer_id == "New Customer":
        df['days_since_first_order'] = 0
        df['customer_avg_order_size'] = base_df['Total Sales'].mean()
    else:
        first_order_date = base_df[base_df['Customer ID'] == customer_id]['Order Date'].min()
        if pd.notna(first_order_date):
            df['days_since_first_order'] = (df['Order Date'] - first_order_date).dt.days.iloc[0]
        else:
            df['days_since_first_order'] = 0
        df['customer_avg_order_size'] = lookups['customer_avg_order_size'].get(customer_id, base_df['Total Sales'].mean())
    df['postal_code_profitability'] = lookups['postal_code_profitability'].get(df['Postal Code'].iloc[0], base_df['Profit_Margin'].mean())
    df['state_sales_volume'] = lookups['state_sales_volume'].get(df['State'].iloc[0], base_df['Total Sales'].mean())
    df['subcategory_avg_margin'] = lookups['subcategory_avg_margin'].get(df['Sub-Category'].iloc[0], base_df['Profit_Margin'].mean())
    df['category_avg_discount'] = lookups['category_avg_discount'].get(df['Category'].iloc[0], base_df['Discount'].mean())
    return df

# --- Page Rendering Functions ---
# (render_home_page, render_customer_segmentation_page, render_prediction_suite_page, render_recommender_page all remain the same)
# ...
def render_home_page(df):
    st.title("Superstore Intelligence Engine v2.0")
    st.markdown("An integrated MLOps platform for customer intelligence, operational forecasting, and strategic decision support. This page showcases the output of our automated EDA reporting pipeline.")
    
    # --- KPI Section (Unchanged) ---
    st.subheader("Key Performance Indicators")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"${df['Total Sales'].sum():,.0f}")
    col2.metric("Total Profit", f"${df['Total Profit'].sum():,.0f}")
    col3.metric("Total Unique Customers", f"{df['Customer ID'].nunique()}")
    st.markdown("---")

    # --- EDA Report Showcase ---
    st.subheader("Automated EDA Report Highlights")
    
    report_path = Path("reports/comprehensive_eda_report")
    
    tab1, tab2, tab3 = st.tabs(["Business Analytics", "Strategic Analysis", "General Data Health"])

    with tab1:
        st.write("#### Foundational Business Analytics")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Sales vs. Profit**")
            try:
                st.image(str(report_path / "2_specialized_analytics/A_sales_vs_profit.png"))
            except Exception:
                st.warning("Plot not found: A_sales_vs_profit.png")

            st.write("**Discount Impact on Profit**")
            try:
                st.image(str(report_path / "2_specialized_analytics/E_discount_impact.png"))
            except Exception:
                st.warning("Plot not found: E_discount_impact.png")

        with c2:
            st.write("**Regional Performance**")
            try:
                st.image(str(report_path / "2_specialized_analytics/B_regional_performance.png"))
            except Exception:
                st.warning("Plot not found: B_regional_performance.png")

            st.write("**Profit Margin by Category**")
            try:
                st.image(str(report_path / "2_specialized_analytics/G_profit_margin_by_category.png"))
            except Exception:
                st.warning("Plot not found: G_profit_margin_by_category.png")

    with tab2:
        st.write("#### Advanced Strategic Analytics")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Product Portfolio Matrix (Volume vs. Profit)**")
            try:
                st.image(str(report_path / "2_specialized_analytics/M_product_portfolio_matrix.png"))
            except Exception:
                st.warning("Plot not found: M_product_portfolio_matrix.png")

            st.write("**Top 20 Co-Purchased Product Pairs**")
            try:
                pairs_df = pd.read_csv(report_path / "2_specialized_analytics/L_top_20_product_pairs.csv")
                st.dataframe(pairs_df)
            except Exception:
                st.warning("Data not found: L_top_20_product_pairs.csv")

        with c2:
            st.write("**Customer LTV (Sales vs. Profit)**")
            try:
                st.image(str(report_path / "2_specialized_analytics/I_customer_ltv_scatter.png"))
            except Exception:
                st.warning("Plot not found: I_customer_ltv_scatter.png")

            st.write("**Top 20 Customers by Profit**")
            try:
                customers_df = pd.read_csv(report_path / "2_specialized_analytics/H_top_20_customers_by_profit.csv")
                st.dataframe(customers_df)
            except Exception:
                st.warning("Data not found: H_top_20_customers_by_profit.csv")
                
    with tab3:
        st.write("#### General Data Health & Distributions")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Basic Profile**")
            try:
                with open(report_path / "1_general_health/basic_profile.txt", 'r') as f:
                    st.text(f.read())
            except Exception:
                st.warning("File not found: basic_profile.txt")
                
        with c2:
            st.write("**Numerical Summary**")
            try:
                num_summary = pd.read_csv(report_path / "1_general_health/numerical_summary.csv", index_col=0)
                st.dataframe(num_summary)
            except Exception:
                st.warning("File not found: numerical_summary.csv")
                
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
    st.markdown("Use our definitive V3 models to analyze transaction profitability. Input values are constrained to realistic ranges based on historical data.")

    c1, c2 = st.columns(2)
    with c1:
        segment = st.selectbox("Customer Segment", df['Segment'].unique()); region = st.selectbox("Region", df['Region'].unique()); state = st.selectbox("State", sorted(df[df['Region'] == region]['State'].unique())); category = st.selectbox("Product Category", df['Category'].unique()); sub_category = st.selectbox("Product Sub-Category", sorted(df[df['Category'] == category]['Sub-Category'].unique()))
    with c2:
        # --- DEFINITIVE FIX: Constrained Price Input ---
        price_stats = df[df['Sub-Category'] == sub_category]['Sales Price'].describe()
        sales_price = st.number_input(
            f"Sales Price ($) - Realistic Range: ${price_stats['min']:.2f} to ${price_stats['max']:.2f}",
            min_value=float(price_stats['min']),
            max_value=float(price_stats['max']),
            value=round(float(price_stats['mean']), 2),
            step=0.01,
            format="%.2f"
        )
        quantity = st.number_input("Quantity", min_value=1, value=2, step=1); discount = st.slider("Discount (%)", min_value=0, max_value=100, value=10) / 100.0
        postal_code = st.selectbox("Postal Code", sorted(df[df['State'] == state]['Postal Code'].unique()))
        customer_id = st.selectbox("Select Existing Customer (for context)", ["New Customer"] + sorted(list(df['Customer ID'].unique())))

    if st.button("Analyze Profitability"):
        # ... (Prediction logic remains the same)
        raw_input = {'Segment': segment, 'Region': region, 'State': state, 'Category': category, 'Sub-Category': sub_category, 'Sales Price': sales_price, 'Quantity': quantity, 'Discount': discount, 'Postal Code': postal_code, 'Customer ID': customer_id, 'Order Date': pd.Timestamp.now()}
        input_df_featured = create_live_features(pd.DataFrame([raw_input]), lookups, df)
        
        classifier = models["v3_classifier"]; forecaster = models["v3_forecaster"]
        pred = classifier.predict(input_df_featured)[0]; proba = classifier.predict_proba(input_df_featured)[0]
        
        st.subheader("Profitability Analysis")
        if pred == 1:
            st.success(f"**Prediction: Profitable** (Confidence: {proba[1]:.2%})")
            log_profit_pred = forecaster.predict(input_df_featured)[0]; profit_pred = np.expm1(log_profit_pred)
            total_sales = sales_price * quantity; margin_pred = profit_pred / total_sales if total_sales > 0 else 0
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

# --- CORRECTED render_strategic_insights_page ---
# In app.py

def render_strategic_insights_page():
    st.title("Strategic Insights Dashboard")
    
    st.subheader("Key Drivers of Profit (SHAP Analysis)")
    st.markdown("This analysis reveals the most impactful factors driving our V3 Profit Forecaster's predictions. The plots show *what* matters most to our bottom line.")

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Overall Feature Importance (Bar Plot)**")
        try:
            st.image('reports/final_shap_plots/shap_summary_bar.png')
        except Exception:
            st.error("Bar plot image not found.")
    
    with c2:
        st.write("**Feature Impact & Direction (Beeswarm Plot)**")
        try:
            st.image('reports/final_shap_plots/shap_summary_beeswarm.png')
        except Exception:
            st.error("Beeswarm plot image not found.")

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
    


def create_ceo_memo_prompt():
    
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
def create_master_feature_set(df):
    st.info("Engineering the V3 master feature set...")
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
def main():
    base_df = load_data()
    df_engineered = base_df.copy()
    df_engineered['Profit_Margin'] = np.where(df_engineered['Total Sales'] > 0, df_engineered['Total Profit'] / df_engineered['Total Sales'], 0)
    
    models = load_models()
    lookups = create_feature_lookups(df_engineered)
    
    st.sidebar.title("Navigation")
    pages = {
        "Home": lambda: render_home_page(df_engineered),
        "Customer Segmentation": render_customer_segmentation_page,
        "Prediction Suite": lambda: render_prediction_suite_page(df_engineered, models, lookups),
        "Product Recommender": lambda: render_recommender_page(df_engineered, models, lookups),
        "Strategic Insights": lambda: render_strategic_insights_page()
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selection]()



if __name__ == "__main__":
    main()