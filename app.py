import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="Skin Cancer Treatment EDA",
    page_icon="🩺",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title { font-size: 2.5rem; color: #1f77b4; font-weight: 700; }
    .sub-title { font-size: 1.2rem; color: #555555; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🩺 Skin Cancer Treatment: Exploratory Data Analysis (EDA)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Interactive web app to analyze skin cancer patient datasets, diagnostic features, and treatment efficacy.</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for Data Input
st.sidebar.header("📁 Dataset Configuration")
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

@st.cache_data
def load_sample_data():
    """Generates a realistic sample dataset for skin cancer EDA if no file is uploaded."""
    np.random.seed(42)
    n_samples = 500
    
    ages = np.random.randint(20, 85, size=n_samples)
    genders = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.55, 0.45])
    lesion_types = np.random.choice(
        ['Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma', 'Benign Nevus'], 
        size=n_samples, 
        p=[0.25, 0.35, 0.20, 0.20]
    )
    diameters = np.random.gamma(shape=2, scale=3, size=n_samples).round(1) # lesion diameter in mm
    thickness = np.random.exponential(scale=1.5, size=n_samples).round(2)  # Breslow thickness in mm
    ulceration = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.3, 0.7])
    treatments = np.random.choice(['Surgery', 'Immunotherapy', 'Radiation', 'Chemotherapy'], size=n_samples)
    
    # Treatment outcome success based on thickness and stage
    success_prob = 1 / (1 + np.exp(thickness - 2))
    outcome = np.array(['Success' if p > np.random.rand() else 'Recurrence/Failure' for p in success_prob])
    
    df = pd.DataFrame({
        'Age': ages,
        'Gender': genders,
        'Lesion_Type': lesion_types,
        'Diameter_mm': diameters,
        'Breslow_Thickness_mm': thickness,
        'Ulceration': ulceration,
        'Treatment_Type': treatments,
        'Outcome': outcome
    })
    return df

# Load Data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Dataset successfully loaded from file!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        df = load_sample_data()
else:
    st.sidebar.info("No file uploaded. Using synthetic skin cancer dataset for demonstration.")
    df = load_sample_data()

# Navigation tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "📊 Univariate Analysis", "📈 Bivariate & Multivariate Analysis", "🔍 Summary Insights"])

# --- TAB 1: DATA OVERVIEW ---
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients / Records", df.shape[0])
    col2.metric("Total Features / Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    
    st.subheader("Dataset Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("Data Information & Types")
    buffer = []
    for col in df.columns:
        buffer.append({
            "Column Name": col,
            "Non-Null Count": df[col].notnull().sum(),
            "Data Type": str(df[col].dtype)
        })
    st.dataframe(pd.DataFrame(buffer), use_container_width=True)

# --- TAB 2: UNIVARIATE ANALYSIS ---
with tab2:
    st.subheader("Univariate Distribution Analysis")
    
    # Select column to analyze
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    analysis_type = st.radio("Select Variable Type:", ["Numeric Features", "Categorical Features"], horizontal=True)
    
    if analysis_type == "Numeric Features" and numeric_cols:
        selected_num = st.selectbox("Choose Numeric Column:", numeric_cols)
        
        col_a, col_b = st.columns(2)
        with col_a:
            # Plotly Histogram
            fig_hist = px.histogram(df, x=selected_num, marginal="box", title=f"Distribution of {selected_num}", color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_b:
            st.write(f"**Summary Statistics for {selected_num}:**")
            st.write(df[selected_num].describe())
            
    elif analysis_type == "Categorical Features" and categorical_cols:
        selected_cat = st.selectbox("Choose Categorical Column:", categorical_cols)
        
        col_c, col_d = st.columns(2)
        with col_c:
            val_counts = df[selected_cat].value_counts().reset_index()
            val_counts.columns = [selected_cat, 'Count']
            fig_bar = px.bar(val_counts, x=selected_cat, y='Count', title=f"Frequency Count of {selected_cat}", color=selected_cat)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_d:
            st.write(f"**Value Counts & Proportions for {selected_cat}:**")
            counts_df = df[selected_cat].value_counts().reset_index()
            counts_df.columns = [selected_cat, 'Count']
            counts_df['Percentage (%)'] = (counts_df['Count'] / len(df) * 100).round(2)
            st.dataframe(counts_df, use_container_width=True)

# --- TAB 3: BIVARIATE & MULTIVARIATE ANALYSIS ---
with tab3:
    st.subheader("Bivariate & Multivariate Relationships")
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-axis Feature", numeric_cols, index=0 if len(numeric_cols)>0 else 0)
    with col2:
        y_axis = st.selectbox("Y-axis Feature", numeric_cols, index=1 if len(numeric_cols)>1 else 0)
        
    color_by = st.selectbox("Color / Group By (Categorical)", [None] + categorical_cols)
    
    # Scatter plot
    fig_scatter = px.scatter(df, x=x_axis, y=y_axis, color=color_by, title=f"{y_axis} vs {x_axis}", hover_data=df.columns)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Correlation Heatmap for numeric features
    st.subheader("Correlation Heatmap")
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Matrix of Numeric Features")
        st.plotly_chart(fig_corr, use_container_width=True)

# --- TAB 4: SUMMARY INSIGHTS ---
with tab4:
    st.subheader("Key Clinical & Statistical Insights")
    st.info("""
    - **Lesion Size and Thickness:** Larger lesion diameters often correspond with higher Breslow thickness values, which critically impact treatment selection and survival rates.
    - **Treatment Efficacy:** Analyzing treatment types against patient outcomes highlights which therapies yield higher success rates across specific cancer stages or lesion classifications.
    - **Demographic Distribution:** Age and gender distributions provide important context regarding risk factors prevalent in skin cancer patient cohorts.
    """)
    
    st.subheader("Filter Dataset for Export")
    if categorical_cols:
        filter_col = st.selectbox("Filter by Category:", categorical_cols)
        unique_vals = df[filter_col].unique()
        selected_val = st.selectbox("Select Value:", unique_vals)
        filtered_df = df[df[filter_col] == selected_val]
        
        st.write(f"Showing filtered results for **{filter_col} = {selected_val}** ({len(filtered_df)} records):")
        st.dataframe(filtered_df, use_container_width=True)
