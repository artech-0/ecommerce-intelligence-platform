# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import google.generativeai as genai

# --- Page Configuration ---
st.set_page_config(
    page_title="E-commerce Customer Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Asset Loading ---

@st.cache_data
def load_data():
    cleaned_data = pd.read_csv('cleaned_superstore_data.csv')
    rfm_data = pd.read_csv('rfm_customer_data.csv')
    return cleaned_data, rfm_data

@st.cache_resource
def load_models():
    models = {
        "profitability_classifier": joblib.load('profitability_classifier.joblib'),
        "margin_regressor": joblib.load('margin_regressor.joblib'),
        "strategic_driver_model": joblib.load('strategic_profit_driver_model.joblib'),
    }
    return models

# --- Generative AI Configuration ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    GEMINI_MODEL = genai.GenerativeModel('gemma-3-27b-it')
except Exception as e:
    st.error(f"Error configuring Generative AI. Ensure your GOOGLE_API_KEY is set in st.secrets. Details: {e}")
    GEMINI_MODEL = None
    
# --- Persona Content ---
# (Persona dictionary remains the same as previous message)
PERSONAS = {
    "🏆 Super Loyalists (Top 33%)": {
        "stats": {"Recency": "116.0 days", "Frequency": "8.4 orders", "Monetary Value": "$6,585.2"},
        "narrative": """
### 1. Persona Title
🏆 **"The Established Loyalist" (Rachel)**

### 2. Narrative Summary
Meet Rachel. She represents the bedrock of our customer base—a long-standing, high-value patron who has consistently chosen our brand for significant purchases over the years. As a busy professional in her late 30s or 40s with high disposable income, Rachel values quality, durability, and a brand she can trust. She doesn't shop frequently, but when she does, she invests in premium products that meet her standards. While her deep loyalty is proven by her purchase history, it has been nearly four months since her last order, indicating she may be between buying cycles or at risk of drifting away. Her engagement is crucial for our long-term success, and re-establishing a connection is a top priority.

### 3. Key Attributes
* **High Lifetime Value (LTV):** With a total spend of over $6,500, Rachel is one of our most financially significant customers.
* **Brand Loyal & Repeat Purchaser:** She has made an average of 8.4 orders, demonstrating a clear and established preference for our brand over competitors.
* **High Average Order Value (AOV):** Her average order is substantial (approx. $784), indicating a focus on premium, high-ticket items rather than small, frequent buys.
* **Infrequent Purchaser:** The recency of 116 days suggests her buying cycle is long, possibly due to the nature of the products purchased (e.g., furniture, high-end electronics, investment pieces).
* **Potentially "At-Risk":** The long period since her last purchase is a critical data point. While she is historically loyal, she is currently a dormant high-value customer who needs re-engagement.

### 4. Motivations & Needs
* **Seeks Quality and Durability:** Her high AOV points to a motivation driven by product quality and long-term value, not short-term deals or impulse buys. She is investing, not just spending.
* **Needs Trust and Reliability:** To make multiple high-value purchases, she needs to have deep trust in our product quality, customer service, and delivery promises.
* **Appreciates Recognition:** As a top-tier customer, she expects a level of service and recognition that matches her investment in the brand. She is motivated by feeling valued, not just by generic discounts.
* **Values a Seamless Experience:** As a likely busy individual, she needs a frictionless purchasing process, from website navigation to checkout and post-purchase support.

### 5. Actionable Marketing Strategies
* **Launch a Personalized "We Miss You" Re-engagement Campaign:**
    * Avoid generic discount codes. Instead, send a targeted email highlighting "What's New Since Your Last Visit," featuring products complementary to her past purchases. Frame it as a personalized update for a valued customer.
* **Implement an Exclusive VIP/Loyalty Tier:**
    * Automatically enroll this segment into a top tier of the loyalty program that offers exclusive benefits like early access to new collections, a dedicated customer service line, or invitations to exclusive virtual events. The goal is to reward her status, not just her spend.
* **Leverage High-Touch Content Marketing:**
    * Send this segment exclusive, high-quality content that reinforces their smart purchase decisions. Examples include "Care Guides" for products they own, behind-the-scenes looks at product craftsmanship, or stories about the brand's values and impact.
* **Solicit High-Value Feedback:**
    * Make her feel like a brand insider. Send exclusive surveys or invitations to a "customer advisory panel" to get her feedback on potential new products. This reinforces her importance and provides invaluable market research.
* **Deploy Smart Retargeting with New Arrivals:**
    * Use targeted social and display ads that focus on new, high-end arrivals rather than sales or items she has already viewed. The messaging should be aspirational and focused on what's next for the brand.""" # Truncated for brevity - paste your full personas here
    },
    "💰 High Spenders (Middle 33%)": {
        "stats": {"Recency": "134.1 days", "Frequency": "5.8 orders", "Monetary Value": "$3,334.4"},
        "narrative": """
### 1. Persona Title
💰 **"The Selective Investor" (Mark)**

### 2. Narrative Summary
Meet Mark. He is a discerning, high-value customer who invests significantly in our products when he has a specific need. Representing the sizable middle-tier of our customer base, Mark is willing to spend over $3,000 with us, but his engagement is less frequent and more calculated than that of our top loyalists. He has made several purchases, indicating a level of trust and satisfaction, but his long purchasing cycle of over four months suggests he is a project-based or event-driven buyer rather than a habitual shopper. While he is a major contributor to revenue, his loyalty is not guaranteed. The key challenge and opportunity with Mark is to bridge the long gaps between his purchases and transform his occasional large investments into a more consistent, long-term relationship.

### 3. Key Attributes
* **Significant Lifetime Value (LTV):** With a total spend exceeding $3,300, Mark is a highly valuable customer who makes a substantial impact on revenue.
* **High Average Order Value (AOV):** His average spend per order is approximately $575 ($3,334.4 / 5.8 orders), showing a preference for high-quality, premium items.
* **Moderately Frequent Purchaser:** He has made multiple repeat purchases (5.8 orders), but not enough to be considered a deeply embedded loyalist.
* **Currently Dormant:** His 134-day recency is a critical flag. He is a disengaged customer who is likely not thinking about our brand and is at risk of being acquired by a competitor for his next major purchase.
* **Major Market Segment:** This group represents a third of our entire customer base, making their engagement and retention a strategic priority.

### 4. Motivations & Needs
* **Solution-Oriented:** Mark shops to solve a specific, often large-scale problem or fulfill a project (e.g., furnishing a room, upgrading technology, buying a seasonal set). He isn't typically a casual browser.
* **Driven by Research and Value:** Before making a significant purchase, he likely invests time in research. He needs strong social proof, detailed product specifications, and clear value propositions to feel confident in his investment.
* **Needs a Compelling Reason to Return:** His purchasing habit is not automatic. He needs to be prompted by a new need, a compelling new product launch, or a highly relevant offer to re-engage with our brand.
* **Trust in Quality:** He is willing to spend more for products that will last. His repeat business, though infrequent, is built on the trust that our products meet a high standard of quality and performance.

### 5. Actionable Marketing Strategies
* **Launch "Next Project" Themed Campaigns:** Since this segment is project-driven, create content and email campaigns that inspire their next purchase. For example, if they bought a desk, target them with content about "Building the Perfect Home Office" featuring our chairs, lighting, and accessories.
* **Implement a Strategic "Bounce-Back" Offer:** After a large purchase, send a time-sensitive, high-value offer (e.g., "$100 off your next purchase of $400 or more") valid for the next 60-90 days. This incentivizes a follow-up purchase and shortens the long recency period.
* **Promote Cross-Category Discovery:** Mark may only know us for one product category. Use targeted email marketing and digital ads to introduce him to other relevant categories. For instance, a customer who bought kitchen appliances could be targeted with new arrivals in our cookware or dining collections.
* **Re-engage with High-Impact Product Launches:** Use this segment as a primary audience for announcing major new products. Frame these launches as significant enough to warrant a new investment, using messaging like "A New Standard in [Product Category]" or "The Upgrade You've Been Waiting For."
* **Build Off-Cycle Engagement with Value-Add Content:** Bridge the long purchase gaps by sending non-promotional content that reinforces their previous purchase decisions. This could include expert tips, maintenance guides, or stories about the craftsmanship of the products they already own, keeping our brand top-of-mind in a positive context. """
    },
    "😴 Dormant Customers (Bottom 33%)": {
        "stats": {"Recency": "234.8 days", "Frequency": "3.4 orders", "Monetary Value": "$1,676.9"},
        "narrative": """

### 1. Persona Title
😴 **"The Forgotten Patron" (Priya)**

### 2. Narrative Summary
Meet Priya. She represents a significant and concerning segment of our customer base—a once-promising patron who has since gone silent. Priya initially showed signs of becoming a loyal customer, making three to four purchases and spending over $1,600 with us. However, it's now been over seven months since her last interaction. For all intents and purposes, she has churned. Whether she had a poor experience, found a better alternative, or simply no longer has a need for our products, she is completely disengaged from our brand. Re-engaging Priya is a high-effort, high-reward challenge. She represents a massive pool of potential revenue, but winning her back will require a powerful, strategic intervention to remind her why she chose us in the first place.

### 3. Key Attributes
* **Deeply Lapsed/Churned:** With an average recency of 235 days, this segment is no longer actively considering our brand.
* **Proven Historical Value:** Despite being dormant, they have a lifetime spend of nearly $1,700, making them too valuable to ignore completely.
* **Past Multi-Purchaser:** With an average of 3.4 orders, they were once satisfied enough to return multiple times. Their loyalty was emerging but never fully solidified.
* **High Average Order Value (AOV):** Their average order value is substantial (approx. $493), indicating that when they *did* shop, they made significant investments.
* **Largest Risk Cohort:** This group constitutes a full third of our database, representing a major leak in our customer retention funnel.

### 4. Motivations & Needs
* **Needs a Strong Reason to Return:** Standard marketing messages and minor discounts are insufficient. They require a compelling, high-impact incentive or a significant brand update to even consider returning.
* **Lack of Top-of-Mind Awareness:** Our brand has fallen off their radar. Their immediate need is to be reminded of the value and quality they once saw in our products.
* **Potentially Dissatisfied or Indifferent:** Their dormancy could stem from a negative experience (product quality, customer service, shipping) or simply from a competitor meeting their needs better. We must address both possibilities.
* **Requires Re-Proof of Value:** To win them back, we need to prove that our brand is still the best choice. This could be through new innovations, superior quality claims, or an unbeatable value proposition.

### 5. Actionable Marketing Strategies
* **Execute a High-Impact Win-Back Campaign:**
    * Deploy a multi-channel campaign with a powerful, steep offer (e.g., "Here's 25% off to welcome you back" or a significant fixed-dollar discount like "$50 off your next order"). The messaging should explicitly acknowledge their absence and express that they are missed.
* **Launch a Feedback-Oriented Survey:**
    * Instead of a sales pitch, send an email asking, "Where did we go wrong?" or "Help us improve." Offer a strong incentive (e.g., a high-value gift card) for completing the survey. This can provide invaluable data on why they churned and re-engage them in a non-transactional way.
* **Segment for a "Last Chance" Drip Campaign:**
    * For customers who don't respond to the initial win-back offer, create a short (2-3 email) automated series highlighting major brand improvements since their last visit: "See What's New," showcasing top-rated new arrivals and positive customer testimonials to rebuild trust.
* **Isolate and Suppress Unresponsive Contacts:**
    * Recognize that not all customers can be won back. If a user from this segment does not engage with the win-back or survey campaigns, suppress them from regular, costly marketing communications. Keep them on a "low-cost" list for maybe 2-3 major brand announcements per year to avoid wasting marketing spend.
* **Leverage New Product Launches for Re-Activation:**
    * Use a major, innovative product launch as a specific reason to message this group. Frame it as an event: "So much has changed. We invite you to take a second look." This provides a fresh, compelling reason to visit, separate from a simple discount. """
    }
}


# --- Page Rendering Functions ---

def render_home_page(df):
    st.title("E-commerce Customer Intelligence Platform")
    st.markdown("Welcome to the central hub for data-driven insights into our e-commerce operations.")
    
    st.subheader("Key Performance Indicators")
    total_sales = df['Total Sales'].sum()
    total_profit = df['Total Profit'].sum()
    total_customers = df['Customer ID'].nunique()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"${total_sales:,.0f}")
    col2.metric("Total Profit", f"${total_profit:,.0f}")
    col3.metric("Total Unique Customers", f"{total_customers}")

    st.subheader("Profitability by Category")
    fig, ax = plt.subplots()
    category_profit = df.groupby('Category')['Total Profit'].sum().sort_values(ascending=False)
    category_profit.plot(kind='bar', ax=ax)
    st.pyplot(fig)

def render_customer_segmentation_page():
    st.title("Customer Segment Explorer")
    st.markdown("Explore the customer segments discovered by our deep learning model. These AI-generated personas translate quantitative data into actionable insights.")
    
    persona_options = list(PERSONAS.keys())
    selected_persona = st.selectbox("Select a persona:", persona_options)
    
    if selected_persona:
        persona_data = PERSONAS[selected_persona]
        st.subheader(f"Statistical Profile: {selected_persona}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg. Recency", persona_data["stats"]["Recency"])
        col2.metric("Avg. Frequency", persona_data["stats"]["Frequency"])
        col3.metric("Avg. Monetary Value", persona_data["stats"]["Monetary Value"])
        st.markdown("---")
        st.subheader("AI-Generated Persona & Strategy")
        st.markdown(persona_data["narrative"])

def render_prediction_suite_page(df, models):
    st.title("Prediction Suite: The Deal Desk")
    st.markdown("Use our validated ML models to assess deal profitability and forecast margins in real-time. Input values are constrained to realistic ranges based on historical data.")

    st.subheader("Enter Transaction Details")
    
    col1, col2 = st.columns(2)
    with col1:
        segment = st.selectbox("Customer Segment", df['Segment'].unique())
        region = st.selectbox("Region", df['Region'].unique())
        category = st.selectbox("Product Category", df['Category'].unique())
        
        # Make Sub-Category dependent on Category
        sub_category_options = df[df['Category'] == category]['Sub-Category'].unique()
        sub_category = st.selectbox("Product Sub-Category", sub_category_options)

    with col2:
        # --- START OF FIX: CONSTRAIN THE PRICE INPUT ---
        # Calculate realistic price range for the selected sub-category
        price_stats = df[df['Sub-Category'] == sub_category]['Sales Price'].describe()
        min_price = float(price_stats['min'])
        max_price = float(price_stats['max'])
        mean_price = float(price_stats['mean'])

        # Use the calculated stats to set bounds and a default for the number_input
        sales_price = st.number_input(
            f"Sales Price (USD) - Realistic range for {sub_category}: USD {min_price:.2f} to USD {max_price:.2f}",
            min_value=min_price,
            max_value=max_price,
            value=round(mean_price, 2), # Default to the mean price
            step=0.01,
            format="%.2f"
        )
        # --- END OF FIX ---
        
        quantity = st.number_input("Quantity", min_value=1, value=2, step=1)
        discount = st.slider("Discount (%)", min_value=0, max_value=100, value=10) / 100.0

    input_data = pd.DataFrame({
        'Segment': [segment], 'Region': [region], 'Category': [category], 
        'Sub-Category': [sub_category], 'Sales Price': [sales_price],
        'Quantity': [quantity], 'Discount': [discount]
    })

    if st.button("Analyze Profitability"):
        classifier = models["profitability_classifier"]
        regressor = models["margin_regressor"]
        
        prediction = classifier.predict(input_data)[0]
        proba = classifier.predict_proba(input_data)[0]
        
        st.subheader("Profitability Analysis")
        if prediction == 1:
            st.success(f"**Prediction: Profitable** (Confidence: {proba[1]:.2%})")
            margin_prediction = regressor.predict(input_data)[0]
            st.info(f"**Predicted Profit Margin: {margin_prediction:.2%}**")
        else:
            st.error(f"**Prediction: Not Profitable** (Confidence: {proba[0]:.2%})")
            st.warning("Consider restructuring this deal, such as by reducing the discount.")

def render_strategic_insights_page(df, models):
    st.title("Strategic Insights Dashboard")
    
    st.subheader("Key Drivers of Profit Margin (SHAP Analysis)")
    st.markdown("This plot reveals the underlying factors that structurally drive our profit margins. It explains *why* some transactions are inherently more or less profitable.")
    
    # --- START OF FIX ---

    # Define the original feature dataframe
    strategic_features = ['Segment', 'Region', 'Category', 'Sub-Category', 'Quantity']
    X_strat = df[strategic_features]
    
    # Load the specific model for this page
    model = models["strategic_driver_model"]
    
    # Extract the two steps from our pipeline
    preprocessor = model.named_steps['columntransformer']
    explainer_model = model.named_steps['xgbregressor']
    
    # Transform the data exactly as the model was trained
    X_transformed = preprocessor.transform(X_strat)
    
    # Get the correct feature names AFTER one-hot encoding
    new_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(strat_categorical_features)
    final_feature_names = strat_numerical_features + list(new_cat_names)
    
    # Convert the transformed data back to a dense DataFrame with correct column names
    X_transformed_df = pd.DataFrame(X_transformed.toarray(), columns=final_feature_names)
    
    # Calculate SHAP values using the transformed data
    explainer = shap.TreeExplainer(explainer_model)
    shap_values = explainer.shap_values(X_transformed_df)
    
    # Create the plot
    fig, ax = plt.subplots()
    
    # CRITICAL FIX: Pass the TRANSFORMED data (X_transformed_df) to the summary plot
    shap.summary_plot(shap_values, X_transformed_df, show=False)
    
    st.pyplot(fig)
    
    # --- END OF FIX ---
    
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

# We also need to define these lists globally or pass them to the function for the fix to work
strat_numerical_features = ['Quantity']
strat_categorical_features = ['Segment', 'Region', 'Category', 'Sub-Category']
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
    cleaned_data, rfm_data = load_data()
    models = load_models()
    
    st.sidebar.title("Navigation")
    page_options = {
        "Home": lambda: render_home_page(cleaned_data),
        "Customer Segmentation": render_customer_segmentation_page,
        "Prediction Suite": lambda: render_prediction_suite_page(cleaned_data, models),
        "Strategic Insights": lambda: render_strategic_insights_page(cleaned_data, models)
    }
    
    selected_page = st.sidebar.radio("Go to", list(page_options.keys()))
    
    page_options[selected_page]()

if __name__ == "__main__":
    main()