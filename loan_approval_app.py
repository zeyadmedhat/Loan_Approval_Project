import streamlit as st
import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    
    [data-testid="stSidebar"] {
        min-width: 200px !important;
        max-width: 200px !important;
    }
    
    .main > div:first-child {
        padding-top: 1rem;
    }
    
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: var(--text-color);
    }
    
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .approved {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
        animation: pulse 0.5s ease-in-out;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .rejected {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    .feature-card h4 {
        color: white !important;
    }
    
    .feature-card p {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('CatBoost.pkl')
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'CatBoost.pkl' not found!")
        return None

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Loan_Data_Cleaned.csv')
        return df
    except FileNotFoundError:
        st.warning("⚠️ Dataset file 'cleaned_df.csv' not found.")
        return None

# Sidebar Navigation
with st.sidebar:
    st.markdown("<p style='margin-bottom: 5px;'><strong>📂 Navigation</strong></p>", unsafe_allow_html=True)
    page = st.radio("Navigation", ["🏠 Home", "📊 EDA", "🔮 Prediction", "📑 Presentation"], label_visibility="collapsed")

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.markdown("<h2 style='text-align: center;'>🏠 Welcome to Loan Approval Predictor</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic; opacity: 0.7;'>AI-powered loan decision system</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎯 About This Application
        
        This **Machine Learning-powered system** predicts loan approval decisions with exceptional accuracy,
        helping financial institutions make faster, more consistent, and data-driven lending decisions.
        
        ### ⚡ Key Features
        
        - **Exceptional Accuracy**: 99.65% test accuracy using CatBoost
        - **Comprehensive Analysis**: 33 features analyzed per application
        - **Real-time Predictions**: Instant loan approval decisions
        - **Interactive EDA**: Explore 20,000 real loan applications
        - **Professional Reports**: Presentation-ready analytics
        
        ### 📊 Model Performance
        
        **Test Set (4,000 samples):**
        - Accuracy: 99.65%
        - Precision: 99.47%
        - Recall: 99.06%
        - F1 Score: 99.27%
        - ROC AUC: 99.99%
        
        **Training Set (16,000 samples):**
        - Accuracy: 99.59%
        - F1 Score: 99.14%
        """)
    
    with col2:
        st.info("""
        ### 📊 Quick Stats
        
        **Dataset**: 20,000 applications
        
        **Model**: CatBoost Pipeline
        
        **Features**: 33
        
        **Train Accuracy**: 99.59%
        
        **Test Accuracy**: 99.65%
        
        **Approval Rate**: 23.9%
        
        **Technology**: 
        - Python 3.12
        - Streamlit
        - CatBoost
        - Plotly
        """)
    
    st.markdown("---")
    
    # Features
    st.markdown("### ✨ What Can You Do?")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
        <div style='font-size: 48px; margin-bottom: 10px;'>🏠</div>
        <h4 style='margin: 5px 0; color: white !important;'>Home</h4>
        <p style='font-size: 0.9em; color: rgba(255, 255, 255, 0.95) !important;'>Overview</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
        <div style='font-size: 48px; margin-bottom: 10px;'>📊</div>
        <h4 style='margin: 5px 0; color: white !important;'>EDA</h4>
        <p style='font-size: 0.9em; color: rgba(255, 255, 255, 0.95) !important;'>Data Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-card'>
        <div style='font-size: 48px; margin-bottom: 10px;'>🔮</div>
        <h4 style='margin: 5px 0; color: white !important;'>Prediction</h4>
        <p style='font-size: 0.9em; color: rgba(255, 255, 255, 0.95) !important;'>Make Predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='feature-card'>
        <div style='font-size: 48px; margin-bottom: 10px;'>📑</div>
        <h4 style='margin: 5px 0; color: white !important;'>Presentation</h4>
        <p style='font-size: 0.9em; color: rgba(255, 255, 255, 0.95) !important;'>Documentation</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # How It Works
    st.markdown("### 🔄 How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center;'>
        <div style='font-size: 80px; margin-bottom: 15px;'>📋</div>
        <h4>1️⃣ Data Collection</h4>
        <p>Collect comprehensive applicant information including personal details, 
        financial status, credit history, and loan requirements.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
        <div style='font-size: 80px; margin-bottom: 15px;'>🧠</div>
        <h4>2️⃣ ML Analysis</h4>
        <p>CatBoost algorithm analyzes 33 features, identifies patterns, 
        and calculates approval probability with 99.65% accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center;'>
        <div style='font-size: 80px; margin-bottom: 15px;'>✅</div>
        <h4>3️⃣ Decision</h4>
        <p>Get instant approval/rejection decision with probability score 
        and detailed insights on key factors.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Why Choose Us
    st.markdown("### 🌟 Why Choose Our System?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **✅ Exceptional Accuracy**
        - 99.65% test accuracy
        - Only 14 errors in 4,000 predictions
        - Industry-leading performance
        - Validated on real data
        
        **⚡ Speed & Efficiency**
        - Instant predictions (< 1 second)
        - Process thousands of applications
        - Real-time decision making
        - Automated workflow
        """)
    
    with col2:
        st.info("""
        **🎯 Comprehensive Analysis**
        - 33 features analyzed per application
        - Multiple data categories covered
        - Credit, financial, personal factors
        - Historical patterns considered
        
        **📊 Transparent Results**
        - Probability scores provided
        - Key factors highlighted
        - Easy to understand output
        - Detailed insights available
        """)
    
    st.markdown("---")
    
    # Get Started
    st.markdown("### 🚀 Get Started")
    st.info("""
    **Ready to begin?** Use the navigation panel on the left to:
    - 📊 **EDA**: Explore the training data and discover insights
    - 🔮 **Prediction**: Make loan approval predictions
    - 📑 **Presentation**: View model performance and documentation
    """)

# ==================== EDA PAGE ====================
elif page == "📊 EDA":
    st.markdown("<h2 style='text-align: center;'>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic; opacity: 0.7;'>Insights from 20,000 loan applications</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    df = load_data()
    
    if df is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "💳 Credit Analysis", "💰 Financial Patterns", "🎯 Target Distribution"])
        
        with tab1:
            st.markdown("#### 📈 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                approved = df['LoanApproved'].sum()
                st.metric("Approved Loans", f"{approved:,}")
            with col3:
                approval_rate = (approved / len(df)) * 100
                st.metric("Approval Rate", f"{approval_rate:.1f}%")
            with col4:
                avg_loan = df['LoanAmount'].mean()
                st.metric("Avg Loan Amount", f"${avg_loan:,.0f}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Approval Distribution
                approval_counts = df['LoanApproved'].value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=['Rejected', 'Approved'],
                    values=approval_counts.values,
                    marker=dict(colors=['#ef4444', '#22c55e']),
                    hole=0.4
                )])
                fig.update_layout(title="Loan Approval Distribution", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Approval by Employment
                approval_by_emp = df.groupby('EmploymentStatus')['LoanApproved'].agg(['sum', 'count'])
                approval_by_emp['rate'] = (approval_by_emp['sum'] / approval_by_emp['count']) * 100
                
                fig = go.Figure(data=[go.Bar(
                    x=approval_by_emp.index,
                    y=approval_by_emp['rate'],
                    marker_color='#667eea',
                    text=approval_by_emp['rate'].round(1),
                    texttemplate='%{text}%',
                    textposition='outside'
                )])
                fig.update_layout(
                    title="Approval Rate by Employment Status",
                    xaxis_title="Employment Status",
                    yaxis_title="Approval Rate (%)",
                    height=400,
                    yaxis=dict(range=[0, 35])
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Age Distribution
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df[df['LoanApproved']==0]['Age'],
                    name='Rejected',
                    marker_color='#ef4444',
                    opacity=0.7
                ))
                fig.add_trace(go.Histogram(
                    x=df[df['LoanApproved']==1]['Age'],
                    name='Approved',
                    marker_color='#22c55e',
                    opacity=0.7
                ))
                fig.update_layout(
                    title="Age Distribution by Approval",
                    xaxis_title="Age",
                    yaxis_title="Count",
                    barmode='overlay',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Credit Score Distribution
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df[df['LoanApproved']==0]['CreditScore'],
                    name='Rejected',
                    marker_color='#ef4444',
                    opacity=0.7
                ))
                fig.add_trace(go.Histogram(
                    x=df[df['LoanApproved']==1]['CreditScore'],
                    name='Approved',
                    marker_color='#22c55e',
                    opacity=0.7
                ))
                fig.update_layout(
                    title="Credit Score Distribution",
                    xaxis_title="Credit Score",
                    yaxis_title="Count",
                    barmode='overlay',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("#### 💳 Categorical Features Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Education Level
                edu_approval = df.groupby('EducationLevel')['LoanApproved'].agg(['sum', 'count'])
                edu_approval['rate'] = (edu_approval['sum'] / edu_approval['count']) * 100
                edu_approval = edu_approval.sort_values('rate', ascending=False)
                
                fig = go.Figure(data=[go.Bar(
                    x=edu_approval.index,
                    y=edu_approval['rate'],
                    marker_color='#667eea',
                    text=edu_approval['rate'].round(1),
                    texttemplate='%{text}%',
                    textposition='outside'
                )])
                fig.update_layout(
                    title="Approval Rate by Education Level",
                    xaxis_title="Education Level",
                    yaxis_title="Approval Rate (%)",
                    height=500,
                    xaxis_autorange=True,
                    yaxis_autorange=True,
                    yaxis=dict(range=[0, 35])
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Marital Status
                marital_approval = df.groupby('MaritalStatus')['LoanApproved'].agg(['sum', 'count'])
                marital_approval['rate'] = (marital_approval['sum'] / marital_approval['count']) * 100
                
                fig = go.Figure(data=[go.Bar(
                    x=marital_approval.index,
                    y=marital_approval['rate'],
                    marker_color='#764ba2',
                    text=marital_approval['rate'].round(1),
                    texttemplate='%{text}%',
                    textposition='outside'
                )])
                fig.update_layout(
                    title="Approval Rate by Marital Status",
                    xaxis_title="Marital Status",
                    yaxis_title="Approval Rate (%)",
                    height=500,
                    yaxis=dict(range=[0, 35])
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Home Ownership
                home_approval = df.groupby('HomeOwnershipStatus')['LoanApproved'].agg(['sum', 'count'])
                home_approval['rate'] = (home_approval['sum'] / home_approval['count']) * 100
                
                fig = go.Figure(data=[go.Bar(
                    x=home_approval.index,
                    y=home_approval['rate'],
                    marker_color='#10b981',
                    text=home_approval['rate'].round(1),
                    texttemplate='%{text}%',
                    textposition='outside'
                )])
                fig.update_layout(
                    title="Approval Rate by Home Ownership",
                    xaxis_title="Home Ownership Status",
                    yaxis_title="Approval Rate (%)",
                    height=450,
                    yaxis=dict(range=[0, 35])
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Loan Purpose
                purpose_approval = df.groupby('LoanPurpose')['LoanApproved'].agg(['sum', 'count'])
                purpose_approval['rate'] = (purpose_approval['sum'] / purpose_approval['count']) * 100
                purpose_approval = purpose_approval.sort_values('rate', ascending=False)
                
                fig = go.Figure(data=[go.Bar(
                    x=purpose_approval.index,
                    y=purpose_approval['rate'],
                    marker_color='#f59e0b',
                    text=purpose_approval['rate'].round(1),
                    texttemplate='%{text}%',
                    textposition='outside'
                )])
                fig.update_layout(
                    title="Approval Rate by Loan Purpose",
                    xaxis_title="Loan Purpose",
                    yaxis_title="Approval Rate (%)",
                    height=450,
                    yaxis=dict(range=[0, 35])
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("#### 💰 Financial Patterns")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_income = df['AnnualIncome'].mean()
                st.metric("Avg Annual Income", f"${avg_income:,.0f}")
            with col2:
                avg_dti = df['DebtToIncomeRatio'].mean()
                st.metric("Avg DTI Ratio", f"{avg_dti:.2%}")
            with col3:
                avg_net_worth = df['NetWorth'].mean()
                st.metric("Avg Net Worth", f"${avg_net_worth:,.0f}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Income vs Loan Amount
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df[df['LoanApproved']==0]['AnnualIncome'],
                    y=df[df['LoanApproved']==0]['LoanAmount'],
                    mode='markers',
                    name='Rejected',
                    marker=dict(color='#ef4444', size=4, opacity=0.5)
                ))
                fig.add_trace(go.Scatter(
                    x=df[df['LoanApproved']==1]['AnnualIncome'],
                    y=df[df['LoanApproved']==1]['LoanAmount'],
                    mode='markers',
                    name='Approved',
                    marker=dict(color='#22c55e', size=4, opacity=0.5)
                ))
                fig.update_layout(
                    title="Income vs Loan Amount",
                    xaxis_title="Annual Income ($)",
                    yaxis_title="Loan Amount ($)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Net Worth Distribution
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=df[df['LoanApproved']==0]['NetWorth'],
                    name='Rejected',
                    marker_color='#ef4444'
                ))
                fig.add_trace(go.Box(
                    y=df[df['LoanApproved']==1]['NetWorth'],
                    name='Approved',
                    marker_color='#22c55e'
                ))
                fig.update_layout(
                    title="Net Worth by Approval Status",
                    yaxis_title="Net Worth ($)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown("#### 🎯 Target Distribution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Approval Distribution
                approval_counts = df['LoanApproved'].value_counts()
                total = len(df)
                approved_pct = (approval_counts[1] / total) * 100
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Rejected', 'Approved'],
                    values=[approval_counts[0], approval_counts[1]],
                    marker=dict(colors=['#ef4444', '#22c55e']),
                    hole=0.4,
                    textinfo='label+percent'
                )])
                fig.update_layout(title="Loan Approval Distribution", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Total Applications", f"{total:,}")
                with col_b:
                    st.metric("Approval Rate", f"{approved_pct:.1f}%")
            
            with col2:
                # Demographics
                st.markdown("#### 👥 Applicant Demographics")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    avg_age = df['Age'].mean()
                    st.metric("Avg Age", f"{avg_age:.0f} years")
                    avg_exp = df['Experience'].mean()
                    st.metric("Avg Experience", f"{avg_exp:.0f} years")
                with col_b:
                    avg_credit = df['CreditScore'].mean()
                    st.metric("Avg Credit Score", f"{avg_credit:.0f}")
                    avg_income = df['AnnualIncome'].mean()
                    st.metric("Avg Income", f"${avg_income:,.0f}")
            
            st.markdown("---")
            
            # Historical Factors
            st.markdown("#### 📊 Historical Factors Impact")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bankruptcy
                bankruptcy_data = df.groupby('BankruptcyHistory')['LoanApproved'].agg(['sum', 'count'])
                bankruptcy_data['rate'] = (bankruptcy_data['sum'] / bankruptcy_data['count']) * 100
                
                fig = go.Figure(data=[go.Bar(
                    x=['No Bankruptcy', 'Has Bankruptcy'],
                    y=bankruptcy_data['rate'].values,
                    marker_color=['#22c55e', '#ef4444'],
                    text=bankruptcy_data['rate'].round(1),
                    texttemplate='%{text}%',
                    textposition='outside'
                )])
                fig.update_layout(
                    title="Approval Rate by Bankruptcy History",
                    yaxis_title="Approval Rate (%)",
                    height=500,
                    xaxis_autorange=True,
                    yaxis_autorange=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Defaults
                defaults_data = df.groupby('PreviousLoanDefaults')['LoanApproved'].agg(['sum', 'count'])
                defaults_data['rate'] = (defaults_data['sum'] / defaults_data['count']) * 100
                
                fig = go.Figure(data=[go.Bar(
                    x=['No Defaults', 'Has Defaults'],
                    y=defaults_data['rate'].values,
                    marker_color=['#22c55e', '#ef4444'],
                    text=defaults_data['rate'].round(1),
                    texttemplate='%{text}%',
                    textposition='outside'
                )])
                fig.update_layout(
                    title="Approval Rate by Previous Defaults",
                    yaxis_title="Approval Rate (%)",
                    height=500,
                    xaxis_autorange=True,
                    yaxis_autorange=True
                )
                st.plotly_chart(fig, use_container_width=True)

# ==================== PREDICTION PAGE ====================
elif page == "🔮 Prediction":
    st.markdown("<h2 style='text-align: center;'>🔮 Loan Approval Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic; opacity: 0.7;'>Enter applicant details for instant prediction</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    model = load_model()
    
    if model is not None:
        st.markdown("### 📝 Enter Applicant Information")
        
        tab1, tab2, tab3, tab4 = st.tabs(["👤 Personal Info", "💼 Employment & Income", "💳 Credit & Debt", "🏦 Loan Details"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", min_value=18, max_value=80, value=35)
                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
                num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=5, value=0)
            
            with col2:
                education_level = st.selectbox("Education Level", 
                    ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
                home_ownership = st.selectbox("Home Ownership Status", 
                    ["Rent", "Own", "Mortgage"])
            
            with col3:
                employment_status = st.selectbox("Employment Status", 
                    ["Employed", "Self-Employed", "Unemployed"])
                experience = st.number_input("Years of Experience", min_value=0, max_value=61, value=10)
                job_tenure = st.number_input("Job Tenure (years)", min_value=0, max_value=40, value=5)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                annual_income = st.number_input("Annual Income ($)", 
                    min_value=15000, max_value=500000, value=60000, step=1000)
                monthly_income = annual_income / 12
                
                savings_balance = st.number_input("Savings Account Balance ($)", 
                    min_value=0, max_value=500000, value=10000, step=100)
                checking_balance = st.number_input("Checking Account Balance ($)", 
                    min_value=0, max_value=200000, value=5000, step=100)
            
            with col2:
                total_assets = st.number_input("Total Assets ($)", 
                    min_value=0, max_value=1000000, value=50000, step=1000)
                total_liabilities = st.number_input("Total Liabilities ($)", 
                    min_value=0, max_value=500000, value=20000, step=1000)
                net_worth = total_assets - total_liabilities
                
                st.metric("Net Worth", f"${net_worth:,.2f}")
        
        with tab3:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                credit_score = st.number_input("Credit Score", 
                    min_value=343, max_value=712, value=600)
                credit_card_util = st.slider("Credit Card Utilization Rate", 
                    min_value=0.0, max_value=1.0, value=0.3, step=0.01)
                num_open_credit = st.number_input("Number of Open Credit Lines", 
                    min_value=0, max_value=20, value=3)
            
            with col2:
                monthly_debt = st.number_input("Monthly Debt Payments ($)", 
                    min_value=50, max_value=3000, value=400, step=10)
                debt_to_income = monthly_debt / (annual_income / 12) if annual_income > 0 else 0
                total_debt_to_income = (monthly_debt * 12) / annual_income if annual_income > 0 else 0
                
                st.metric("Debt-to-Income Ratio", f"{debt_to_income:.2%}")
                
                num_inquiries = st.number_input("Number of Credit Inquiries", 
                    min_value=0, max_value=20, value=2)
            
            with col3:
                length_credit_history = st.number_input("Length of Credit History (years)", 
                    min_value=0, max_value=50, value=10)
                payment_history = st.slider("Payment History Score", 
                    min_value=0, max_value=100, value=85)
                utility_bills_payment = st.slider("Utility Bills Payment History", 
                    min_value=0, max_value=100, value=90)
                
                bankruptcy = st.selectbox("Bankruptcy History", [0, 1], 
                    format_func=lambda x: "Yes" if x == 1 else "No")
                previous_defaults = st.selectbox("Previous Loan Defaults", [0, 1], 
                    format_func=lambda x: "Yes" if x == 1 else "No")
        
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                loan_amount = st.number_input("Loan Amount ($)", 
                    min_value=3674, max_value=200000, value=25000, step=100)
                loan_duration = st.selectbox("Loan Duration (months)", 
                    [12, 24, 36, 48, 60, 72, 84, 96, 108, 120])
                loan_purpose = st.selectbox("Loan Purpose", 
                    ["Home", "Auto", "Education", "Business", "Other"])
            
            with col2:
                base_interest_rate = st.slider("Base Interest Rate (%)", 
                    min_value=3.0, max_value=15.0, value=6.5, step=0.1)
                
                interest_rate_adjustment = (700 - credit_score) / 100 * 0.5
                interest_rate = base_interest_rate + interest_rate_adjustment
                
                monthly_rate = interest_rate / 100 / 12
                num_payments = loan_duration
                monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / \
                                ((1 + monthly_rate)**num_payments - 1) if monthly_rate > 0 else loan_amount / num_payments
                
                st.metric("Adjusted Interest Rate", f"{interest_rate:.2f}%")
                st.metric("Monthly Loan Payment", f"${monthly_payment:.2f}")
        
        # Prediction Button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🔮 Predict Loan Approval", use_container_width=True)
        
        if predict_button:
            # Prepare input data
            input_data = pd.DataFrame({
                'Age': [age],
                'AnnualIncome': [annual_income],
                'CreditScore': [credit_score],
                'EmploymentStatus': [employment_status],
                'EducationLevel': [education_level],
                'Experience': [experience],
                'LoanAmount': [loan_amount],
                'LoanDuration': [loan_duration],
                'MaritalStatus': [marital_status],
                'NumberOfDependents': [num_dependents],
                'HomeOwnershipStatus': [home_ownership],
                'MonthlyDebtPayments': [monthly_debt],
                'CreditCardUtilizationRate': [credit_card_util],
                'NumberOfOpenCreditLines': [num_open_credit],
                'NumberOfCreditInquiries': [num_inquiries],
                'DebtToIncomeRatio': [debt_to_income],
                'BankruptcyHistory': [bankruptcy],
                'LoanPurpose': [loan_purpose],
                'PreviousLoanDefaults': [previous_defaults],
                'PaymentHistory': [payment_history],
                'LengthOfCreditHistory': [length_credit_history],
                'SavingsAccountBalance': [savings_balance],
                'CheckingAccountBalance': [checking_balance],
                'TotalAssets': [total_assets],
                'TotalLiabilities': [total_liabilities],
                'MonthlyIncome': [monthly_income],
                'UtilityBillsPaymentHistory': [utility_bills_payment],
                'JobTenure': [job_tenure],
                'NetWorth': [net_worth],
                'BaseInterestRate': [base_interest_rate],
                'InterestRate': [interest_rate],
                'MonthlyLoanPayment': [monthly_payment],
                'TotalDebtToIncomeRatio': [total_debt_to_income]
            })
            
            # Make prediction
            with st.spinner('🤔 Analyzing application...'):
                try:
                    prediction = model.predict(input_data)[0]
                    prediction_proba = model.predict_proba(input_data)[0]
                    
                    st.markdown("---")
                    
                    # Display result
                    if prediction == 1:
                        st.markdown("""
                            <div class='prediction-box approved'>
                                ✅ LOAN APPROVED
                            </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown("""
                            <div class='prediction-box rejected'>
                                ❌ LOAN REJECTED
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Show probability gauge
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        approval_prob = prediction_proba[1] * 100
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=approval_prob,
                            title={'text': "Approval Probability", 'font': {'size': 24}},
                            delta={'reference': 50},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkgreen" if prediction == 1 else "darkred"},
                                'steps': [
                                    {'range': [0, 40], 'color': "lightcoral"},
                                    {'range': [40, 70], 'color': "lightyellow"},
                                    {'range': [70, 100], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show key factors
                    st.markdown("### 📊 Application Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Credit Score", credit_score, 
                                 delta="Good" if credit_score >= 600 else "Fair")
                    with col2:
                        st.metric("Debt-to-Income", f"{debt_to_income:.1%}", 
                                 delta="Good" if debt_to_income < 0.36 else "High")
                    with col3:
                        st.metric("Net Worth", f"${net_worth:,.0f}", 
                                 delta="Positive" if net_worth > 0 else "Negative")
                    with col4:
                        st.metric("Monthly Payment", f"${monthly_payment:.0f}")
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")

# ==================== PRESENTATION PAGE ====================
elif page == "📑 Presentation":
    st.markdown("<h2 style='text-align: center;'>📑 Project Documentation & Insights</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic; opacity: 0.7;'>Comprehensive model overview and performance metrics</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    pres_tab1, pres_tab2, pres_tab3, pres_tab4 = st.tabs([
        "🎯 Project Overview",
        "🤖 Model Details",
        "📊 Performance Metrics",
        "🚀 Deployment Info"
    ])
    
    with pres_tab1:
        st.markdown("## 💰 Loan Approval Prediction System")
        
        df = load_data()
        
        if df is not None:
            total_records = len(df)
            approved = df['LoanApproved'].sum()
            approval_rate = (approved / total_records) * 100
            avg_age = df['Age'].mean()
            avg_income = df['AnnualIncome'].mean()
            avg_credit = df['CreditScore'].mean()
        else:
            total_records = 20000
            approval_rate = 23.9
            avg_age = 40
            avg_income = 59161
            avg_credit = 575
        
        st.markdown(f"""
        ### 🎯 Project Objective
        
        Develop an automated machine learning system to predict loan approval decisions
        based on comprehensive applicant information, helping financial institutions make
        faster, more consistent, and data-driven lending decisions.
        
        ---
        
        ### 📊 Dataset Statistics
        
        - **Total Applications**: {total_records:,}
        - **Approval Rate**: {approval_rate:.1f}%
        - **Average Applicant Age**: {avg_age:.1f} years
        - **Average Annual Income**: ${avg_income:,.0f}
        - **Average Credit Score**: {avg_credit:.0f}
        
        ---
        
        ### 🔍 Problem Statement
        
        Traditional loan approval processes are:
        - ⏰ **Time-consuming**: Manual review takes days or weeks
        - 🔄 **Inconsistent**: Different analysts may reach different conclusions
        - 📊 **Subjective**: Prone to human bias and errors
        - 💰 **Costly**: Requires significant human resources
        
        ---
        
        ### 💡 Our Solution
        
        A machine learning-powered prediction system that:
        - ⚡ **Instant Analysis**: Real-time predictions in seconds
        - 🎯 **Consistent**: Standardized decision criteria
        - 📈 **Exceptional Accuracy**: 99.65% test accuracy
        - 💻 **Scalable**: Handle thousands of applications
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 📥 Input Features (33 Total)
            
            **Personal (4)**
            - Age, Education, Marital Status, Dependents
            
            **Employment (3)**
            - Status, Experience, Tenure
            
            **Financial (7)**
            - Income, Assets, Liabilities, Net Worth, Savings, Checking, Monthly Income
            
            **Credit (8)**
            - Score, Utilization, Open Lines, Inquiries, History, Payment, Length, Utility Payment
            
            **Loan (7)**
            - Amount, Duration, Purpose, Base Rate, Interest, Monthly Payment, Total DTI
            
            **History (4)**
            - Monthly Debt, Bankruptcy, Defaults, Debt-to-Income
            """)
        
        with col2:
            st.markdown("""
            #### 📤 Output
            
            **Binary Classification**
            - ✅ Approved (1)
            - ❌ Rejected (0)
            
            **Probability Score**
            - 0-100% approval likelihood
            
            **Key Insights**
            - Credit score analysis
            - Debt-to-income ratio
            - Net worth evaluation
            - Risk factors identified
            
            ### 🎯 Model Performance
            
            **Test Accuracy**: 99.65%
            **Train Accuracy**: 99.59%
            **F1 Score**: 99.27%
            **ROC AUC**: 99.99%
            """)
    
    with pres_tab2:
        st.markdown("## 🤖 Model Architecture & Pipeline")
        
        st.markdown("""
        ### 🔄 ML Pipeline Overview
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style='background-color: rgba(102, 126, 234, 0.2); padding: 15px; border-radius: 10px; text-align: center;'>
            <h3>1️⃣</h3>
            <h4>Data Input</h4>
            <p>33 Features</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background-color: rgba(102, 126, 234, 0.2); padding: 15px; border-radius: 10px; text-align: center;'>
            <h3>2️⃣</h3>
            <h4>Preprocessing</h4>
            <p>RobustScaler + OneHotEncoder</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background-color: rgba(102, 126, 234, 0.2); padding: 15px; border-radius: 10px; text-align: center;'>
            <h3>3️⃣</h3>
            <h4>Balancing</h4>
            <p>SMOTE</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style='background-color: rgba(102, 126, 234, 0.2); padding: 15px; border-radius: 10px; text-align: center;'>
            <h3>4️⃣</h3>
            <h4>Prediction</h4>
            <p>CatBoost</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔧 Preprocessing Steps
            
            **Numerical Features (27)**
            - **Scaler**: RobustScaler
            - **Why**: Handles outliers better
            - **Features**: Income, age, credit score, amounts, ratios
            
            **Categorical Features (6)**
            - **Encoder**: OneHotEncoder (drop='first')
            - **Why**: Prevents multicollinearity
            - **Features**: Employment, education, marital status, home ownership, loan purpose
            
            **Feature Engineering**
            - Debt-to-Income Ratio
            - Total Debt-to-Income Ratio
            - Net Worth calculation
            - Monthly payment estimation
            - Interest rate adjustment
            """)
        
        with col2:
            st.markdown("""
            ### ⚖️ Class Balancing
            
            **Technique**: SMOTE
            - Synthetic Minority Over-sampling
            - Creates synthetic examples
            - Prevents overfitting
            
            **Why SMOTE?**
            - Better than random over-sampling
            - Avoids exact duplication
            - Maintains feature space integrity
            
            **Algorithm**: CatBoost Classifier
            - Gradient boosting algorithm
            - Excellent with categorical features
            - Robust to overfitting
            - Fast training and prediction
            
            **Hyperparameters**:
            - Iterations: 1000
            - Learning Rate: 0.03
            - Depth: 6
            - Random State: 42
            """)
    
    with pres_tab3:
        st.markdown("## 📊 Model Performance Metrics")
        
        st.markdown("### 📈 Test Set Performance (4,000 samples)")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", "99.65%", delta="±0.2%")
        with col2:
            st.metric("Precision", "99.47%", delta="±0.3%")
        with col3:
            st.metric("Recall", "99.06%", delta="±0.4%")
        with col4:
            st.metric("F1 Score", "99.27%", delta="±0.3%")
        with col5:
            st.metric("ROC AUC", "99.99%", delta="±0.1%")
        
        st.markdown("---")
        
        # Train vs Test Comparison
        st.info("""
        **📊 Training Set Performance** (16,000 samples):
        - Accuracy: 99.59% | Precision: 98.93% | Recall: 99.35% | F1: 99.14% | ROC AUC: 99.99%
        
        **🎯 Test Set Performance** (4,000 samples) - Shown above:
        - Accuracy: 99.65% | Precision: 99.47% | Recall: 99.06% | F1: 99.27% | ROC AUC: 99.99%
        
        **Analysis**: Model performs consistently on both sets with no overfitting. 
        Test accuracy slightly higher shows excellent generalization.
        """)
        
        st.markdown("---")
        
        # Confusion Matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Confusion Matrix (Test Set)")
            
            conf_matrix = np.array([[3039, 5], [9, 947]])
            
            fig = go.Figure(data=go.Heatmap(
                z=conf_matrix,
                x=['Predicted Rejected', 'Predicted Approved'],
                y=['Actual Rejected', 'Actual Approved'],
                text=conf_matrix,
                texttemplate='%{text}',
                textfont={"size": 20},
                colorscale='Blues',
                showscale=False
            ))
            
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Matrix Breakdown:**
            - True Negatives (TN): 3,039
            - False Positives (FP): 5
            - False Negatives (FN): 9
            - True Positives (TP): 947
            """)
        
        with col2:
            st.markdown("#### 📊 Metric Comparison")
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
            scores = [99.65, 99.47, 99.06, 99.27, 99.99]
            
            fig = go.Figure(data=[go.Bar(
                x=metrics,
                y=scores,
                marker_color='#667eea',
                text=scores,
                texttemplate='%{text:.2f}%',
                textposition='outside'
            )])
            
            fig.update_layout(
                height=350,
                yaxis=dict(range=[95, 101]),
                yaxis_title="Score (%)",
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 💡 Performance Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **✅ Exceptional Strengths**
            - Near-perfect accuracy (99.65%)
            - Excellent precision (99.47%)
            - Outstanding recall (99.06%)
            - Almost perfect ROC AUC (99.99%)
            - Only 14 misclassifications out of 4,000
            - Extremely low false positive rate (0.16%)
            - Very low false negative rate (0.94%)
            """)
        
        with col2:
            st.info("""
            **🎯 Business Impact**
            - 99.65% correct decisions
            - Only 5 bad loans approved (minimal risk)
            - Only 9 good applicants rejected (minimal loss)
            - Exceptional reliability for production use
            - Can process thousands of applications
            - Drastically reduces manual review needs
            - Industry-leading performance
            """)
    
    with pres_tab4:
        st.markdown("## 🚀 Deployment & Technical Stack")
        
        st.markdown("### 💻 Technology Stack")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Frontend**
            - Streamlit 1.29.0
            - Plotly 5.18.0
            - Custom CSS styling
            
            **Features**
            - Responsive design
            - Interactive charts
            - Real-time validation
            """)
        
        with col2:
            st.markdown("""
            **Machine Learning**
            - CatBoost 1.2.2
            - Scikit-learn 1.3.2
            - Imbalanced-learn 0.11.0
            
            **Capabilities**
            - Classification
            - Preprocessing
            - Class balancing
            """)
        
        with col3:
            st.markdown("""
            **Data Processing**
            - Pandas 2.1.3
            - NumPy 1.26.2
            - Joblib 1.3.2
            
            **Operations**
            - Data manipulation
            - Numerical computing
            - Model persistence
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📁 Project Structure
        
        ```
        loan-approval-predictor/
        │
        ├── loan_approval_app.py      # Main Streamlit application
        ├── CatBoost.pkl               # Trained model (2.7 MB)
        ├── Loan_Data_Cleaned.csv      # Dataset (20,000 records)
        ├── requirements.txt           # Dependencies
        └── README.md                  # Documentation
        ```
        """)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Key Features Implemented")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **User Interface**
            - ✅ Home page with overview
            - ✅ Interactive EDA with visualizations
            - ✅ Single prediction form (4 tabs)
            - ✅ Professional presentation page
            - ✅ Responsive design
            - ✅ Custom gradient theme
            """)
        
        with col2:
            st.markdown("""
            **Functionality**
            - ✅ Real-time predictions (99.65% accuracy)
            - ✅ Probability gauges
            - ✅ Error handling
            - ✅ Input validation
            - ✅ Model caching
            - ✅ Interactive charts
            """)
        
        st.markdown("---")
        
        st.success("""
        ### ✅ Project Deliverables
        
        - ✅ Fully functional ML web application
        - ✅ Trained and validated model (99.65% accuracy)
        - ✅ Comprehensive documentation
        - ✅ Deployment-ready code
        - ✅ EDA with real data visualizations
        - ✅ Professional presentation
        - ✅ User-friendly interface
        - ✅ Production-quality code
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
<p style='font-size: 15px; margin-bottom: 15px;'>
💰 Loan Approval Predictor © 2025 | Developed by <strong>Zeyad Medhat</strong>
</p>
<p style='font-size: 16px; margin-bottom: 10px;'>
🔗 <strong>Connect with Me</strong>
</p>
<p style='font-size: 15px;'>
<a href='https://github.com/zeyadmedhat' target='_blank' style='color: #667eea; text-decoration: none; margin: 0 15px;'>🌐 GitHub</a> | 
<a href='https://linkedin.com/in/zeyad-medhat' target='_blank' style='color: #667eea; text-decoration: none; margin: 0 15px;'>💼 LinkedIn</a> | 
<a href='mailto:zeyadmedhat.official@gmail.com' style='color: #667eea; text-decoration: none; margin: 0 15px;'>📧 Email</a>
</p>
</div>
""", unsafe_allow_html=True)
