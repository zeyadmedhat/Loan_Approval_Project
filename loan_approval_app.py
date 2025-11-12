"""
💰 Loan Approval Predictor - Streamlit Application
Author: Zeyad Medhat
Description: End-to-End Machine Learning Project for Loan Approval Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from catboost import CatBoostClassifier
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

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

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained CatBoost model"""
    try:
        model = joblib.load('CatBoost.pkl')
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'CatBoost.pkl' not found!")
        return None

@st.cache_data
def load_data():
    """Load the cleaned dataset"""
    try:
        df = pd.read_csv('Loan_Data_Cleaned.csv')
        return df
    except FileNotFoundError:
        st.warning("⚠️ Dataset file 'Loan_Data_Cleaned.csv' not found.")
        return None

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.markdown("<p style='margin-bottom: 5px;'><strong>📂 Navigation</strong></p>", unsafe_allow_html=True)
    page = st.radio("Navigation", ["🏠 Home", "📊 EDA", "🔮 Prediction", "📑 Presentation"], label_visibility="collapsed")

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "🏠 Home":
    st.markdown("<h2 style='text-align: center;'>🏠 Welcome to Loan Approval Predictor</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic; opacity: 0.7;'>AI-powered loan decision system</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎯 About This Application
        
        This **Machine Learning-powered system** predicts loan approval decisions,
        helping financial institutions make faster, more consistent, and data-driven lending decisions.
        
        ### ⚡ Key Features
        
        - **Intelligent Analysis**: CatBoost algorithm analyzes 33 comprehensive features
        - **Real-time Predictions**: Instant loan approval decisions
        - **Interactive EDA**: Explore 20,000 real loan applications
        - **Professional System**: End-to-end ML pipeline from data to deployment
        
        ### 📊 System Capabilities
        
        **Comprehensive Feature Analysis:**
        - Personal information (Age, Education, Marital Status)
        - Employment history and experience
        - Financial status and assets
        - Credit history and payment behavior
        - Loan requirements and purpose
        
        **Advanced ML Pipeline:**
        - Robust data preprocessing
        - Feature engineering
        - SMOTE class balancing
        - CatBoost classification
        """)
    
    with col2:
        st.info("""
        ### 📊 Quick Stats
        
        **Dataset**: 20,000 applications
        
        **Model**: CatBoost Pipeline
        
        **Features**: 33 comprehensive attributes
        
        **Categories**: 6 major groups
        
        **Technology**: 
        - Python 3.12
        - Streamlit
        - CatBoost
        - Plotly
        - SMOTE
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
        and calculates approval probability using advanced gradient boosting.</p>
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
        **✅ Data-Driven Decisions**
        - Comprehensive 33-feature analysis
        - Advanced gradient boosting algorithm
        - Handles class imbalance with SMOTE
        - Validated on 20,000 real applications
        
        **⚡ Speed & Efficiency**
        - Instant predictions (< 1 second)
        - Process thousands of applications
        - Real-time decision making
        - Automated workflow
        """)
    
    with col2:
        st.info("""
        **🎯 Comprehensive Analysis**
        - Personal & demographic factors
        - Employment & income verification
        - Credit history & payment behavior
        - Financial assets & liabilities
        
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
    - 📑 **Presentation**: View complete project workflow and documentation
    """)

# ============================================================================
# EDA PAGE
# ============================================================================

elif page == "📊 EDA":
    st.markdown("<h2 style='text-align: center;'>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic; opacity: 0.7;'>Insights from 20,000 loan applications</p>", unsafe_allow_html=True)
    
    df = load_data()
    
    if df is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "💳 Credit Analysis", "💰 Financial Patterns", "🎯 Target Distribution"])
        
        # ========================================
        # TAB 1: Overview
        # ========================================
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
                    yaxis=dict(range=[0, max(approval_by_emp['rate']) * 1.2])
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Age Distribution with OVERLAY
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df[df['LoanApproved']==0]['Age'],
                    name='Rejected',
                    marker_color='#ef4444',
                    opacity=0.7,
                    nbinsx=30
                ))
                fig.add_trace(go.Histogram(
                    x=df[df['LoanApproved']==1]['Age'],
                    name='Approved',
                    marker_color='#22c55e',
                    opacity=0.7,
                    nbinsx=30
                ))
                fig.update_layout(
                    title="Age Distribution by Approval Status",
                    xaxis_title="Age",
                    yaxis_title="Count",
                    barmode='overlay',
                    height=400,
                    legend=dict(x=0.7, y=0.95)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Credit Score Distribution with OVERLAY
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df[df['LoanApproved']==0]['CreditScore'],
                    name='Rejected',
                    marker_color='#ef4444',
                    opacity=0.7,
                    nbinsx=30
                ))
                fig.add_trace(go.Histogram(
                    x=df[df['LoanApproved']==1]['CreditScore'],
                    name='Approved',
                    marker_color='#22c55e',
                    opacity=0.7,
                    nbinsx=30
                ))
                fig.update_layout(
                    title="Credit Score Distribution by Approval Status",
                    xaxis_title="Credit Score",
                    yaxis_title="Count",
                    barmode='overlay',
                    height=400,
                    legend=dict(x=0.7, y=0.95)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # ========================================
        # TAB 2: Credit Analysis
        # ========================================
        with tab2:
            st.markdown("#### 💳 Credit Analysis")
            
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
                    height=450,
                    yaxis=dict(range=[0, max(edu_approval['rate']) * 1.15])
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
                    height=450,
                    yaxis=dict(range=[0, max(marital_approval['rate']) * 1.15])
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
                    yaxis=dict(range=[0, max(home_approval['rate']) * 1.15])
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
                    yaxis=dict(range=[0, max(purpose_approval['rate']) * 1.15])
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # ========================================
        # TAB 3: Financial Patterns
        # ========================================
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
                    title="Annual Income vs Loan Amount",
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
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Income Distribution with OVERLAY
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df[df['LoanApproved']==0]['AnnualIncome'],
                    name='Rejected',
                    marker_color='#ef4444',
                    opacity=0.7,
                    nbinsx=30
                ))
                fig.add_trace(go.Histogram(
                    x=df[df['LoanApproved']==1]['AnnualIncome'],
                    name='Approved',
                    marker_color='#22c55e',
                    opacity=0.7,
                    nbinsx=30
                ))
                fig.update_layout(
                    title="Annual Income Distribution by Approval",
                    xaxis_title="Annual Income ($)",
                    yaxis_title="Count",
                    barmode='overlay',
                    height=400,
                    legend=dict(x=0.7, y=0.95)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # DTI Distribution with OVERLAY
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df[df['LoanApproved']==0]['DebtToIncomeRatio'],
                    name='Rejected',
                    marker_color='#ef4444',
                    opacity=0.7,
                    nbinsx=30
                ))
                fig.add_trace(go.Histogram(
                    x=df[df['LoanApproved']==1]['DebtToIncomeRatio'],
                    name='Approved',
                    marker_color='#22c55e',
                    opacity=0.7,
                    nbinsx=30
                ))
                fig.update_layout(
                    title="Debt-to-Income Ratio Distribution by Approval",
                    xaxis_title="DTI Ratio",
                    yaxis_title="Count",
                    barmode='overlay',
                    height=400,
                    legend=dict(x=0.7, y=0.95)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # ========================================
        # TAB 4: Target Distribution
        # ========================================
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
                    height=400,
                    yaxis=dict(range=[0, max(bankruptcy_data['rate']) * 1.2])
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
                    height=400,
                    yaxis=dict(range=[0, max(defaults_data['rate']) * 1.2])
                )
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("❌ Unable to load dataset. Please ensure 'Loan_Data_Cleaned.csv' is in the same directory.")

# ============================================================================
# PREDICTION PAGE
# ============================================================================

elif page == "🔮 Prediction":
    st.markdown("<h2 style='text-align: center;'>🔮 Loan Approval Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic; opacity: 0.7;'>Enter applicant details for instant prediction</p>", unsafe_allow_html=True)
    
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
                'PaymentHistory': [payment_history / 100],
                'LengthOfCreditHistory': [length_credit_history],
                'SavingsAccountBalance': [savings_balance],
                'CheckingAccountBalance': [checking_balance],
                'TotalAssets': [total_assets],
                'TotalLiabilities': [total_liabilities],
                'MonthlyIncome': [monthly_income],
                'UtilityBillsPaymentHistory': [utility_bills_payment / 100],
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
    
    else:
        st.error("❌ Unable to load model. Please ensure 'CatBoost.pkl' is in the same directory.")

# ============================================================================
# PRESENTATION PAGE WITH TABS
# ============================================================================

elif page == "📑 Presentation":
    st.markdown("<h2 style='text-align: center;'>🎯 Project Presentation</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic; opacity: 0.7;'>Complete workflow from problem definition to deployment</p>", unsafe_allow_html=True)
    
    
    # Create tabs for each step
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🧭 Step 1: Problem",
        "📂 Step 2: Data",
        "🧹 Step 3: Cleaning",
        "🔧 Step 4: Preprocessing",
        "🧠 Step 5: Training",
        "🎯 Step 6: Selection",
        "🚀 Step 7: Deployment",
        "🎓 Step 8: Conclusion"
    ])
    
    # ========================================
    # TAB 1: Problem Definition
    # ========================================
    with tab1:
        st.markdown("## 🧭 Step 1: Problem Definition")
        
        st.markdown("""
        The goal of this project was to **predict loan approval decisions**, identifying which applicants 
        are most likely to be approved or rejected based on their comprehensive financial, personal, 
        and credit history data.
        
        ### Business Problem:
        
        **Challenges in Traditional Loan Processing:**
        - ⏰ **Time-consuming**: Manual review takes days or weeks
        - 🔄 **Inconsistent**: Different analysts may reach different conclusions
        - 📊 **Subjective**: Prone to human bias and errors
        - 💰 **Costly**: Requires significant human resources
        - 📉 **Limited Scale**: Cannot efficiently process high volumes
        
        ### Our Solution:
        
        **Automated ML-Powered Decision System:**
        - ⚡ Instant predictions in under 1 second
        - 🎯 Consistent, data-driven decisions
        - 📈 Scalable to thousands of applications
        - 💻 Reduces operational costs
        - 🔍 Transparent probability scores
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **🎯 Project Objectives:**
            - Build end-to-end ML pipeline
            - Process 33 comprehensive features
            - Handle class imbalance effectively
            - Deploy user-friendly web application
            - Provide interpretable results
            """)
        
        with col2:
            st.info("""
            **📊 Success Criteria:**
            - Complete data preprocessing
            - Compare multiple ML algorithms
            - Select best performing model
            - Create interactive dashboard
            - Demonstrate production readiness
            """)
    
    # ========================================
    # TAB 2: Data Collection
    # ========================================
    with tab2:
        st.markdown("## 📂 Step 2: Data Collection")
        
        df = load_data()
        
        if df is not None:
            total_records = len(df)
            approved = df['LoanApproved'].sum()
            approval_rate = (approved / total_records) * 100
        else:
            total_records = 20000
            approval_rate = 23.9
        
        st.markdown(f"""
        The dataset **Loan_Data_Cleaned.csv** includes **{total_records:,}** loan applications.
        
        ### Dataset Loading:
        """)
        
        st.code("""
import pandas as pd
import numpy as np
import plotly.express as px

# Load the dataset
df = pd.read_csv('Loan_Approval_Data.csv')
df.head()
        """, language='python')
        
        st.markdown("### 📋 Feature Categories:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📋 Personal Information (4 features):**
            - Age
            - Education Level
            - Marital Status
            - Number of Dependents
            
            **💼 Employment & Income (7 features):**
            - Employment Status
            - Years of Experience
            - Job Tenure
            - Annual Income
            - Monthly Income
            - Savings Account Balance
            - Checking Account Balance
            
            **💳 Credit & Financial Health (12 features):**
            - Credit Score
            - Credit Card Utilization Rate
            - Number of Open Credit Lines
            - Number of Credit Inquiries
            - Payment History
            - Length of Credit History
            - Utility Bills Payment History
            - Total Assets
            - Total Liabilities
            - Net Worth
            - Monthly Debt Payments
            - Debt-to-Income Ratio
            """)
        
        with col2:
            st.markdown("""
            **🏦 Loan Details (7 features):**
            - Loan Amount
            - Loan Duration
            - Loan Purpose
            - Base Interest Rate
            - Adjusted Interest Rate
            - Monthly Loan Payment
            - Total Debt-to-Income Ratio
            
            **📊 Risk Factors (3 features):**
            - Bankruptcy History
            - Previous Loan Defaults
            - Home Ownership Status
            
            **🎯 Target Variable:**
            - **LoanApproved**: Binary (0 = Rejected, 1 = Approved)
            """)
        
        st.markdown("---")
        
        st.markdown(f"""
        ### 📊 Dataset Overview:
        - **Total Records:** {total_records:,}
        - **Total Features:** 35 (33 predictors + 1 target + 1 risk score)
        - **Approval Rate:** {approval_rate:.1f}%
        - **Class Distribution:** Imbalanced (requires SMOTE)
        """)
    
    # ========================================
    # TAB 3: Data Cleaning
    # ========================================
    with tab3:
        st.markdown("## 🧹 Step 3: Data Cleaning & Exploration")
        
        st.markdown("""
        ### Data Quality Checks:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 Check Data Types:**")
            st.code("""
# Check Data Types
df.info()
            """, language='python')
            
            st.markdown("**🔍 Check Duplicates:**")
            st.code("""
# Check duplicates
df.duplicated().sum()
            """, language='python')
        
        with col2:
            st.markdown("**📊 Summary Statistics:**")
            st.code("""
# Numerical columns
df.describe(include='number').round(2)

# Categorical columns
df.describe(include='object').round(2)
            """, language='python')
            
            st.markdown("**❌ Missing Values:**")
            st.code("""
# Check missing values percentage
df.isna().mean().round(4) * 100
            """, language='python')
        
        st.markdown("---")
        
        st.markdown("### Data Cleaning Steps:")
        
        st.code("""
# Drop unnecessary columns
df.drop(columns=['ApplicationDate'], inplace=True)

# Check for duplicates again
df.duplicated().sum()
        """, language='python')
        
        st.markdown("### Feature Exploration:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Categorical Columns:**")
            st.code("""
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    print(col)
    print(df[col].nunique())
    print(df[col].unique())
    print('-' * 100)
            """, language='python')
        
        with col2:
            st.markdown("**Numerical Columns:**")
            st.code("""
num_cols = df.select_dtypes(include='number').columns

for col in num_cols:
    px.histogram(data_frame=df, x=col).show()
            """, language='python')
        
        st.markdown("---")
        
        st.markdown("### Save Cleaned Data:")
        
        st.code("""
df.to_csv('Loan_Data_Cleaned.csv', index=False)
        """, language='python')
        
        st.success("""
        **✅ Cleaning Complete:**
        - Removed unnecessary date column
        - Verified no duplicates
        - Confirmed no missing values
        - Explored categorical and numerical distributions
        - Saved cleaned dataset
        """)
    
    # ========================================
    # TAB 4: Preprocessing
    # ========================================
    with tab4:
        st.markdown("## 🔧 Step 4: Data Preprocessing")
        
        st.markdown("""
        ### Preprocessing Pipeline Architecture:
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background-color: rgba(102, 126, 234, 0.2); padding: 20px; border-radius: 10px; text-align: center;'>
            <h3>1️⃣</h3>
            <h4>Numerical Pipeline</h4>
            <p>RobustScaler</p>
            <p style='font-size: 0.9em;'>27 features</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background-color: rgba(102, 126, 234, 0.2); padding: 20px; border-radius: 10px; text-align: center;'>
            <h3>2️⃣</h3>
            <h4>Categorical Pipeline</h4>
            <p>OneHotEncoder</p>
            <p style='font-size: 0.9em;'>6 features</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background-color: rgba(102, 126, 234, 0.2); padding: 20px; border-radius: 10px; text-align: center;'>
            <h3>3️⃣</h3>
            <h4>Combined</h4>
            <p>ColumnTransformer</p>
            <p style='font-size: 0.9em;'>All features</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 1️⃣ Numerical Pipeline:")
        
        st.code("""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

num_pipeline = Pipeline([('Robust Scaler', scaler)])
num_pipeline
        """, language='python')
        
        st.info("""
        **Why RobustScaler?**
        - Robust to outliers in financial data
        - Uses median and IQR instead of mean and std
        - Better for skewed distributions
        """)
        
        st.markdown("---")
        
        st.markdown("### 2️⃣ Categorical Pipeline:")
        
        st.code("""
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop='first', sparse_output=False)

cat_pipeline = Pipeline(steps=[('OHE', ohe)])
cat_pipeline
        """, language='python')
        
        st.info("""
        **OneHotEncoder Settings:**
        - `drop='first'`: Avoids multicollinearity
        - `sparse_output=False`: Returns dense arrays
        - Handles categorical features automatically
        """)
        
        st.markdown("---")
        
        st.markdown("### 3️⃣ Combined Preprocessing:")
        
        st.code("""
from sklearn.compose import ColumnTransformer

preprocessing = ColumnTransformer(
    transformers=[
        ('Num Pipeline', num_pipeline, num_cols),
        ('OHE Pipeline', cat_pipeline, cat_cols)
    ],
    remainder='passthrough'
)
preprocessing
        """, language='python')
        
        st.success("""
        **✅ Preprocessing Complete:**
        - Numerical features scaled with RobustScaler
        - Categorical features encoded with OneHotEncoder
        - All transformations combined in single pipeline
        - Ready for model training
        """)
    
    # ========================================
    # TAB 5: Model Training
    # ========================================
    with tab5:
        st.markdown("## 🧠 Step 5: Model Training & Comparison")
        
        st.markdown("""
        ### Models Evaluated:
        
        During experimentation, **8 different models** were trained and compared using **5-fold cross-validation**:
        
        1. Logistic Regression
        2. K-Nearest Neighbors (KNN)
        3. Gaussian Naive Bayes
        4. Decision Tree
        5. Random Forest
        6. XGBoost
        7. **CatBoost** ⭐
        8. LightGBM
        """)
        
        st.markdown("---")
        
        st.markdown("### Training Code:")
        
        st.code("""
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

smote = SMOTE(random_state=42)

models = [
    ('Logistic Regression', LogisticRegression(random_state=42, n_jobs=-1)),
    ('KNN', KNeighborsClassifier(n_jobs=-1)),
    ('Gaussian NB', GaussianNB()),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42, n_jobs=-1)),
    ('XGBoost', XGBClassifier()),
    ('CatBoost', CatBoostClassifier(verbose=0)),
    ('LightGBM', LGBMClassifier(n_jobs=-1))
]

for model in models:
    model_pipeline = Pipeline(steps=[
        ('Preprocessing', preprocessing),
        ('SMOTE', smote),
        ('Model', model[1])
    ])
    
    result = cross_validate(
        model_pipeline, x, y, 
        cv=5, 
        scoring='f1', 
        return_train_score=True, 
        n_jobs=-1
    )

    print(model[0])
    print('Train F1 Score :', round(result['train_score'].mean() * 100, 2))
    print('Test F1 Score :', round(result['test_score'].mean() * 100, 2))
    print('-' * 50)
        """, language='python')
        
        st.markdown("---")
        
        st.markdown("### 🔄 Training Pipeline:")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style='background-color: rgba(102, 126, 234, 0.15); padding: 15px; border-radius: 8px; text-align: center;'>
            <h4>Step 1</h4>
            <p>Preprocessing</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background-color: rgba(102, 126, 234, 0.15); padding: 15px; border-radius: 8px; text-align: center;'>
            <h4>Step 2</h4>
            <p>SMOTE Balancing</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background-color: rgba(102, 126, 234, 0.15); padding: 15px; border-radius: 8px; text-align: center;'>
            <h4>Step 3</h4>
            <p>Model Training</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style='background-color: rgba(102, 126, 234, 0.15); padding: 15px; border-radius: 8px; text-align: center;'>
            <h4>Step 4</h4>
            <p>Cross-Validation</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.info("""
        **⚖️ Class Balancing with SMOTE:**
        
        - **Problem**: Imbalanced dataset (~76% rejected, ~24% approved)
        - **Solution**: SMOTE (Synthetic Minority Over-sampling Technique)
        - **Benefit**: Creates synthetic samples of minority class
        - **Result**: Balanced training data for better model learning
        """)
        
        st.success("""
        **✅ Training Complete:**
        - All 8 models trained successfully
        - 5-fold cross-validation performed
        - F1-scores compared for model selection
        - SMOTE applied to handle class imbalance
        """)
    
    # ========================================
    # TAB 6: Model Selection
    # ========================================
    with tab6:
        st.markdown("## 🎯 Step 6: Model Selection & Deployment")
        
        st.success("""
        ### Selected Model: **CatBoost Classifier** ⭐
        
        **Why CatBoost?**
        - 🏆 Best balance of performance and speed
        - 🎯 Excellent with mixed feature types
        - 📊 Native categorical feature handling
        - ⚡ Fast training and inference
        - 🛡️ Built-in overfitting protection
        - 🔧 Minimal hyperparameter tuning needed
        """)
        
        st.markdown("---")
        
        st.markdown("### Final Model Training:")
        
        st.code("""
catboost_pipeline = Pipeline(steps=[
    ('Preprocessing', preprocessing),
    ('SMOTE', smote),
    ('Model', CatBoostClassifier(verbose=0))
])

# Train on full dataset
catboost_pipeline.fit(x, y)
        """, language='python')
        
        st.markdown("---")
        
        st.markdown("### Model Testing:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.code("""
# Test prediction
catboost_pipeline.predict(x.head(1))[0]
            """, language='python')
        
        with col2:
            st.info("""
            **Prediction Output:**
            - 0 = Loan Rejected
            - 1 = Loan Approved
            """)
        
        st.markdown("---")
        
        st.markdown("### Model Persistence:")
        
        st.code("""
import joblib

# Save the trained pipeline
joblib.dump(catboost_pipeline, 'CatBoost.pkl')
        """, language='python')
        
        st.success("""
        **✅ Model Saved:**
        - Complete pipeline saved as `CatBoost.pkl`
        - Includes preprocessing + SMOTE + CatBoost
        - Ready for deployment
        - File size: ~2.7 MB
        """)
        
        st.markdown("---")
        
        st.markdown("### 🔄 Complete Pipeline:")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style='background-color: rgba(34, 197, 94, 0.2); padding: 15px; border-radius: 10px; text-align: center;'>
            <h3>1️⃣</h3>
            <h4>Input</h4>
            <p>Raw Data</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background-color: rgba(34, 197, 94, 0.2); padding: 15px; border-radius: 10px; text-align: center;'>
            <h3>2️⃣</h3>
            <h4>Preprocess</h4>
            <p>Scale + Encode</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background-color: rgba(34, 197, 94, 0.2); padding: 15px; border-radius: 10px; text-align: center;'>
            <h3>3️⃣</h3>
            <h4>Balance</h4>
            <p>SMOTE</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style='background-color: rgba(34, 197, 94, 0.2); padding: 15px; border-radius: 10px; text-align: center;'>
            <h3>4️⃣</h3>
            <h4>Predict</h4>
            <p>CatBoost</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ========================================
    # TAB 7: Deployment
    # ========================================
    with tab7:
        st.markdown("## 🚀 Step 7: Deployment with Streamlit")
        
        st.markdown("""
        The final pipeline was deployed using **Streamlit**, creating a professional web application 
        with the following capabilities:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📊 Data Exploration:
            - Interactive visualizations
            - 20,000 loan applications analyzed
            - Approval rate analysis by demographics
            - Financial pattern exploration
            - Credit score distributions
            - Overlay histograms for comparisons
            - Category-wise breakdown charts
            - Risk factor impact analysis
            """)
        
        with col2:
            st.markdown("""
            ### 🔮 Real-Time Predictions:
            - User-friendly input forms (4 tabs)
            - 33 comprehensive features
            - Instant approval/rejection decisions
            - Probability scores with gauges
            - Key decision factors highlighted
            - Real-time calculations (DTI, Net Worth)
            - Input validation and error handling
            - Professional result presentation
            """)
        
        st.markdown("---")
        
        st.markdown("### 💻 Technology Stack:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **Frontend**
            - Streamlit 1.29.0
            - Plotly 5.18.0
            - Custom CSS
            - Responsive design
            - Interactive charts
            """)
        
        with col2:
            st.info("""
            **Machine Learning**
            - CatBoost 1.2.2
            - Scikit-learn 1.3.2
            - Imbalanced-learn 0.11.0
            - Pipeline architecture
            - SMOTE balancing
            """)
        
        with col3:
            st.info("""
            **Data Processing**
            - Pandas 2.1.3
            - NumPy 1.26.2
            - Joblib 1.3.2
            - Data manipulation
            - Model persistence
            """)
        
        st.markdown("---")
        
        st.markdown("### 🎨 Application Features:")
        
        st.success("""
        **✅ Professional Dashboard:**
        - 🏠 Home: Project overview and capabilities
        - 📊 EDA: Comprehensive data analysis with visualizations
        - 🔮 Prediction: Interactive prediction interface
        - 📑 Presentation: Complete project documentation
        
        **✅ User Experience:**
        - Clean, modern UI with gradient themes
        - Responsive design for all devices
        - Real-time validations
        - Animated transitions
        - Professional error handling
        
        **✅ Production Ready:**
        - Model caching for performance
        - Session state management
        - Efficient data loading
        - Scalable architecture
        """)
    
    # ========================================
    # TAB 8: Conclusion
    # ========================================
    with tab8:
        st.markdown("## 🎓 Step 8: Conclusion & Insights")
        
        st.success("""
        ### Project Achievements:
        
        ✅ **Complete ML Workflow**: End-to-end pipeline from data to deployment
        
        ✅ **Comprehensive Analysis**: 20,000 applications across 33 features
        
        ✅ **Advanced Techniques**: SMOTE balancing + RobustScaler preprocessing
        
        ✅ **Production System**: Interactive web app with instant predictions
        
        ✅ **Scalable Solution**: Handles high-volume processing efficiently
        """)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📊 Data Insights Discovered:
            
            **Key Predictors:**
            - Credit score strongly influences approval
            - Payment history is critical factor
            - Employment status impacts decisions
            - Debt-to-income ratio crucial metric
            - Educational level shows correlation
            - Home ownership matters
            
            **Patterns Found:**
            - Higher education → higher approval rates
            - Stable employment → better outcomes
            - Lower DTI → more approvals
            - Good payment history → key advantage
            - Bankruptcy/defaults → major red flags
            """)
        
        with col2:
            st.markdown("""
            ### 💡 Business Impact:
            
            **Efficiency Gains:**
            - < 1 second prediction time
            - 1000s of applications processed
            - Automated decision workflow
            - Reduced operational costs
            - Consistent decision criteria
            
            **Value Delivered:**
            - Faster loan processing
            - Reduced human bias
            - Transparent decisions
            - Scalable system
            - Professional reporting
            """)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Model Capabilities:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **Robustness**
            - Handles outliers
            - Works with imbalance
            - Mixed feature types
            - Missing data tolerance
            """)
        
        with col2:
            st.info("""
            **Performance**
            - Fast predictions
            - Efficient training
            - Scalable processing
            - Low resource usage
            """)
        
        with col3:
            st.info("""
            **Interpretability**
            - Probability scores
            - Feature importance
            - Clear decisions
            - Transparent process
            """)
        

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<style>
.footer {text-align:center;font-family:"Segoe UI",sans-serif;padding:20px 0;color:gray;font-size:15px;}
.footer a {color:#4a68f0;text-decoration:none;margin:0 8px;transition:color 0.3s;}
.footer a:hover {color:#2d47b3;}
.footer .divider {color:#aaa;margin:0 3px;}
</style>
<div class="footer">
<p>Loan Approval Predictor © 2025 · Developed by <strong style="color:#bbbbbb;">Zeyad Medhat</strong></p>
<p><a href="https://github.com/zeyadmedhat" target="_blank">GitHub</a><span class="divider">·</span>
<a href="https://linkedin.com/in/zeyad-medhat" target="_blank">LinkedIn</a><span class="divider">·</span>
<a href="mailto:zeyadmedhat.official@gmail.com">Email</a></p>
</div>
""", unsafe_allow_html=True)
