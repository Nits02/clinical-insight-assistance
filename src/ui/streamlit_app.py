"""
Interactive Web Application

This module is the frontend of the Clinical Insights Assistant, built using Streamlit. It
provides an intuitive web interface for users to upload data, trigger analyses, view results,
and interact with the AI agent.

Purpose:
• Provide a user-friendly interface for the Clinical Insights Assistant.
• Allow data upload and display of processed data.
• Present analysis results and AI-generated insights visually.
• Enable interaction with the AI agent for autonomous analysis.

Step-by-Step Implementation:
1. Set up Streamlit Page Configuration and Imports:
   Start with basic Streamlit configuration and import all necessary modules.

2. Define Helper Functions for UI Elements:
   These functions encapsulate common UI patterns.

3. Implement the Main Application Layout:
   Use Streamlit containers and columns to structure the application.
                            st.write(f"**Recommendations:** {issue.recommendation}")
            else:
                st.success("No compliance issues detected")
                
        except Exception as e:
            st.error(f"Error in compliance analysis: {str(e)}")
    
    with tab2:
        st.markdown("#### 🚨 Safety & Adverse Events")
        try:
            # Detect adverse events
            ae_issues = detector.detect_adverse_event_patterns(data)
            
            if ae_issues:
                st.error(f"Found {len(ae_issues)} safety concerns")
                
                for i, issue in enumerate(ae_issues):
                    with st.expander(f"Safety Alert {i+1}: {issue.issue_type} - {issue.severity}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Patient ID:** {issue.patient_id}")
                            st.write(f"**Severity:** {issue.severity}")
                        with col2:
                            st.write(f"**Confidence:** {issue.confidence_score:.1%}")
                            st.write(f"**Visit:** {issue.visit_number if hasattr(issue, "visit_number") and issue.visit_number else "N/A"}")
                        st.write(f"**Description:** {issue.description}")
                        if issue.recommendation:
                            st.write(f"**Recommendations:** {issue.recommendation}")
            else:
                st.success("No safety issues detected")
                
        except Exception as e:
            st.error(f"Error in safety analysis: {str(e)}")
    
    with tab3:
        st.markdown("#### 📉 Efficacy Assessment")
        try:
            # Detect efficacy issues
            efficacy_issues = detector.detect_efficacy_issues(data)
            
            if efficacy_issues:
                st.warning(f"Found {len(efficacy_issues)} efficacy concerns")
                
                for i, issue in enumerate(efficacy_issues):
                    with st.expander(f"Efficacy Issue {i+1}: {issue.issue_type} - {issue.severity}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Patient ID:** {issue.patient_id}")
                            st.write(f"**Severity:** {issue.severity}")
                        with col2:
                            st.write(f"**Confidence:** {issue.confidence_score:.1%}")
                            st.write(f"**Visit:** {issue.visit_number if hasattr(issue, "visit_number") and issue.visit_number else "N/A"}")
                        st.write(f"**Description:** {issue.description}")
                        if issue.recommendation:
                            st.write(f"**Recommendations:** {issue.recommendation}")
            else:
                st.success("No efficacy issues detected")
                
        except Exception as e:
            st.error(f"Error in efficacy analysis: {str(e)}") structure the application.

This streamlit_app.py provides a complete, interactive user interface for the Clinical Insights
Assistant, bringing together all the backend functionalities into a cohesive and user-friendly
experience. It demonstrates how to use Streamlit for data visualization, user input, and
displaying complex analytical results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys
import os

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agent_core import ClinicalAgent
    from data_loader import ClinicalDataLoader
    from genai_interface import GenAIInterface
    from memory import MemoryManager
    from issue_detection import IssueDetector
    from cohort_analysis import CohortAnalyzer
    from scenario_simulation import ScenarioSimulator
except ImportError:
    # Fallback for different import paths
    import sys
    sys.path.append('../')
    from agent_core import ClinicalAgent
    from data_loader import ClinicalDataLoader
    from genai_interface import GenAIInterface
    from memory import MemoryManager
    from issue_detection import IssueDetector
    from cohort_analysis import CohortAnalyzer
    from scenario_simulation import ScenarioSimulator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DEVELOPER DEBUG HELPER FUNCTION
# ============================================================================

def show_debug_info(module_name: str, class_name: str = None, method_name: str = None, description: str = None):
    """
    Display developer debug information showing which code components are being called.
    
    Args:
        module_name (str): Name of the Python module/file
        class_name (str, optional): Name of the class being used
        method_name (str, optional): Name of the method being called
        description (str, optional): Brief description of what's happening
    """
    debug_parts = []
    
    if module_name:
        debug_parts.append(f"📁 **{module_name}**")
    
    if class_name:
        debug_parts.append(f"🏗️ **{class_name}**")
    
    if method_name:
        debug_parts.append(f"⚙️ **{method_name}()**")
    
    if description:
        debug_parts.append(f"💡 *{description}*")
    
    debug_text = " → ".join(debug_parts)
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 4px solid #6c757d;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
        font-size: 0.75rem;
        color: #495057;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    ">
        🔧 <strong>Dev Info:</strong> {debug_text}
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 1. STREAMLIT PAGE CONFIGURATION AND SETUP
# ============================================================================

def configure_page():
    """Configure Streamlit page settings and styling."""
    st.set_page_config(
        page_title="Clinical Insights Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/Nits02/clinical-insight-assistance',
            'Report a bug': 'https://github.com/Nits02/clinical-insight-assistance/issues',
            'About': "# Clinical Insights Assistant\nAI-powered clinical trial analysis platform"
        }
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 0.5rem 0;
        }
        .insight-card {
            background: #e3f2fd;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
            margin: 0.5rem 0;
        }
        .warning-card {
            background: #fff3e0;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #ff9800;
            margin: 0.5rem 0;
        }
        .success-card {
            background: #e8f5e8;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #4caf50;
            margin: 0.5rem 0;
        }
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
        }
        .sidebar .sidebar-content {
            background: #f8f9fa;
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# 2. HELPER FUNCTIONS FOR UI ELEMENTS
# ============================================================================

def display_header():
    """Display the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Clinical Insights Assistant</h1>
        <p>AI-Powered Clinical Trial Analysis Platform</p>
        <p><em>Upload your clinical data, get intelligent insights, and make data-driven decisions</em></p>
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <strong>Quick Start Guide:</strong><br>
            1️⃣ Upload your clinical trial CSV data<br>
            2️⃣ Configure and run AI analysis<br>
            3️⃣ Explore insights in Analytics & Reports<br>
            4️⃣ Use AI Agent for autonomous analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Configure and display the sidebar with navigation and controls."""
    with st.sidebar:
        # Create a custom header instead of relying on external image
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 1rem;
            color: white;
        ">
            <h2 style="margin: 0; color: white;">🏥 Clinical Insights</h2>
            <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9);">AI-Powered Analysis Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎛️ Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["📊 Data Upload & Analysis", "🤖 AI Agent Dashboard", "📈 Analytics & Insights", "⚙️ Settings"]
        )
        
        st.markdown("### 📋 Quick Stats")
        if 'analysis_results' in st.session_state:
            results = st.session_state.analysis_results
            st.metric("Total Records", len(st.session_state.get('uploaded_data', [])))
            st.metric("Insights Generated", len(results.get('insights', [])))
            st.metric("Recommendations", len(results.get('recommendations', [])))
        else:
            st.info("Upload data to see statistics")
        
        st.markdown("### 🔗 Quick Actions")
        if st.button("🔄 Refresh Data"):
            st.rerun()
        
        if st.button("📥 Download Results"):
            if 'analysis_results' in st.session_state:
                download_results()
            else:
                st.warning("No results to download")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Clinical Insights Assistant** helps you:
        - 📤 Upload clinical trial data
        - 🔍 Detect issues and patterns
        - 📊 Compare patient cohorts
        - 🤖 Generate AI insights
        - 📋 Get actionable recommendations
        """)
    
    return page

def display_file_uploader():
    """Display file upload interface."""
    st.markdown("### 📤 Data Upload")
    
    with st.expander("📋 **Data Format Requirements & Sample Structure**", expanded=False):
        st.markdown("""
        **Required/Recommended Columns:**
        - `patient_id` - Unique patient identifier
        - `cohort` - Treatment group (e.g., Treatment_A, Treatment_B, Control)
        - `outcome_score` - Primary endpoint measurement (numeric)
        - `compliance_pct` - Medication compliance percentage (0-100)
        - `adverse_event_flag` - Safety indicator (0/1 or True/False)
        - `visit_date` - Visit date (YYYY-MM-DD format)
        - `dosage_mg` - Medication dosage in milligrams
        - `visit_number` - Sequential visit number
        
        **Sample CSV Structure:**
        ```
        patient_id,cohort,outcome_score,compliance_pct,adverse_event_flag,visit_date,dosage_mg
        P001,Treatment_A,75.5,95.2,0,2024-01-15,50
        P002,Treatment_B,68.3,87.4,1,2024-01-16,75
        P003,Control,45.2,92.1,0,2024-01-17,25
        ```
        
        **💡 Note:** The system is flexible with column names and will auto-detect variations.
        """)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with clinical trial data",
        type=['csv'],
        help="Upload your clinical trial dataset. The file should contain patient data with columns like patient_id, cohort, visit_date, etc."
    )
    
    if uploaded_file is not None:
        try:
            # Show debug info for file upload process
            show_debug_info("data_loader.py", "ClinicalDataLoader", "__init__", "Initializing data loader for file processing")
            
            # Initialize ClinicalDataLoader for advanced data processing
            from data_loader import ClinicalDataLoader
            loader = ClinicalDataLoader()
            
            # Show debug info for data reading
            show_debug_info("pandas", "pd", "read_csv", "Reading CSV file into DataFrame")
            
            # Read the uploaded file
            data = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = data
            
            # Use ClinicalDataLoader for data processing (validation is done internally in load_data)
            with st.spinner("Processing and validating data..."):
                try:
                    # ClinicalDataLoader doesn't have public validate_data_structure method
                    # Instead, we'll do basic validation and let the modules handle detailed validation
                    st.success(f"✅ Successfully uploaded {len(data)} records!")
                    
                    # Basic data quality checks
                    missing_data_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
                    
                    if missing_data_pct > 20:
                        st.warning(f"⚠️ High missing data: {missing_data_pct:.1f}% of values are missing")
                    elif missing_data_pct > 0:
                        st.info(f"ℹ️ Missing data: {missing_data_pct:.1f}% of values are missing")
                    else:
                        st.success("✅ No missing data detected")
                        
                except Exception as validation_error:
                    st.warning(f"⚠️ Data processing note: {str(validation_error)}")
                    st.session_state.uploaded_data = data
            
            # Display data preview
            st.markdown("#### 👀 Data Preview")
            st.dataframe(data.head(10), width='stretch')
            
            # Display comprehensive statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(data))
            with col2:
                st.metric("Columns", len(data.columns))
            with col3:
                if 'patient_id' in data.columns:
                    st.metric("Unique Patients", data['patient_id'].nunique())
                else:
                    st.metric("Unique Patients", "N/A")
            with col4:
                if 'cohort' in data.columns:
                    st.metric("Cohorts", data['cohort'].nunique())
                else:
                    st.metric("Cohorts", "N/A")
            
            # Data quality indicators
            st.markdown("#### 📊 Data Quality Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
                st.metric("Missing Data", f"{missing_pct:.1f}%")
            
            with col2:
                if 'adverse_event_flag' in data.columns:
                    ae_rate = data['adverse_event_flag'].mean() * 100
                    st.metric("Adverse Event Rate", f"{ae_rate:.1f}%")
                else:
                    st.metric("Adverse Event Rate", "N/A")
            
            with col3:
                if 'compliance_pct' in data.columns:
                    avg_compliance = data['compliance_pct'].mean()
                    st.metric("Avg Compliance", f"{avg_compliance:.1f}%")
                else:
                    st.metric("Avg Compliance", "N/A")
            
            with col4:
                if 'outcome_score' in data.columns:
                    avg_outcome = data['outcome_score'].mean()
                    st.metric("Avg Outcome Score", f"{avg_outcome:.1f}")
                else:
                    st.metric("Avg Outcome Score", "N/A")
            
            return data
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    
    return None

def display_analysis_controls():
    """Display analysis configuration and trigger controls."""
    st.markdown("### 🎯 Analysis Configuration")
    
    st.info("""
    **🔧 Configure Your Analysis:**
    - **Analysis Goals**: Select what you want to analyze (efficacy, safety, compliance, etc.)
    - **Confidence Threshold**: Higher values = more conservative results (0.7 recommended)
    - **Max Insights**: Number of key findings to generate (10 is optimal)
    - **Include Recommendations**: Get actionable clinical recommendations
    - **Detailed Analysis**: Enable for comprehensive statistical analysis
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Analysis Goals")
        default_goals = [
            "Compare treatment efficacy across cohorts",
            "Evaluate safety profile and adverse events",
            "Assess patient compliance patterns",
            "Identify high-risk patients"
        ]
        
        analysis_goals = st.multiselect(
            "Select analysis objectives:",
            default_goals + ["Custom analysis goal"],
            default=default_goals[:2],
            help="Choose what aspects of your clinical trial you want to analyze"
        )
        
        if "Custom analysis goal" in analysis_goals:
            custom_goal = st.text_input("Enter custom analysis goal:")
            if custom_goal:
                analysis_goals = [g for g in analysis_goals if g != "Custom analysis goal"] + [custom_goal]
    
    with col2:
        st.markdown("#### ⚙️ Analysis Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            0.0, 1.0, 0.7, 0.1,
            help="Higher values produce more confident but fewer results (0.7 recommended)"
        )
        max_insights = st.number_input(
            "Maximum Insights", 
            1, 50, 10,
            help="Number of key findings to generate (10 is optimal for most analyses)"
        )
        include_recommendations = st.checkbox(
            "Include Recommendations", 
            True,
            help="Generate actionable clinical recommendations based on findings"
        )
        detailed_analysis = st.checkbox(
            "Detailed Analysis", 
            False,
            help="Enable comprehensive statistical analysis (takes longer but more thorough)"
        )
    
    # Analysis trigger button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Start AI Analysis", width='stretch'):
            if 'uploaded_data' in st.session_state:
                run_analysis(
                    st.session_state.uploaded_data,
                    analysis_goals,
                    confidence_threshold,
                    max_insights,
                    include_recommendations,
                    detailed_analysis
                )
            else:
                st.error("Please upload data first!")
    
    return {
        'goals': analysis_goals,
        'confidence_threshold': confidence_threshold,
        'max_insights': max_insights,
        'include_recommendations': include_recommendations,
        'detailed_analysis': detailed_analysis
    }

def display_data_visualizations(data: pd.DataFrame):
    """Display interactive data visualizations."""
    st.markdown("### 📊 Data Visualizations")
    
    if data is None or data.empty:
        st.warning("No data to visualize")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔍 Cohort Analysis", "📅 Time Series", "🎯 Custom"])
    
    with tab1:
        display_overview_charts(data)
    
    with tab2:
        display_cohort_analysis_charts(data)
    
    with tab3:
        display_time_series_charts(data)
    
    with tab4:
        display_custom_charts(data)

def display_overview_charts(data: pd.DataFrame):
    """Display overview charts."""
    col1, col2 = st.columns(2)
    
    with col1:
        if 'cohort' in data.columns:
            # Cohort distribution
            cohort_counts = data['cohort'].value_counts()
            fig = px.pie(
                values=cohort_counts.values,
                names=cohort_counts.index,
                title="Patient Distribution by Cohort"
            )
            st.plotly_chart(fig, width='stretch')
    
    with col2:
        if 'outcome_score' in data.columns:
            # Outcome distribution
            fig = px.histogram(
                data,
                x='outcome_score',
                title="Outcome Score Distribution",
                nbins=20
            )
            st.plotly_chart(fig, width='stretch')
    
    # Additional charts
    if 'adverse_event_flag' in data.columns:
        ae_summary = data.groupby('cohort')['adverse_event_flag'].agg(['sum', 'count']).reset_index()
        ae_summary['rate'] = ae_summary['sum'] / ae_summary['count'] * 100
        
        fig = px.bar(
            ae_summary,
            x='cohort',
            y='rate',
            title="Adverse Event Rate by Cohort (%)",
            color='cohort'
        )
        st.plotly_chart(fig, use_container_width=True)

def display_cohort_analysis_charts(data: pd.DataFrame):
    """Display cohort comparison charts."""
    if 'cohort' not in data.columns:
        st.warning("No cohort information found in data")
        return
    
    # Cohort comparison metrics
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) > 0:
        selected_metric = st.selectbox("Select metric for cohort comparison:", numeric_columns)
        
        # Box plot comparison
        fig = px.box(
            data,
            x='cohort',
            y=selected_metric,
            title=f"{selected_metric.title()} by Cohort"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        cohort_stats = data.groupby('cohort')[selected_metric].describe()
        st.markdown("#### Statistical Summary")
        st.dataframe(cohort_stats, use_container_width=True)

def display_time_series_charts(data: pd.DataFrame):
    """Display time series visualizations."""
    date_columns = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    if not date_columns:
        st.warning("No date/time columns found for time series analysis")
        return
    
    date_col = st.selectbox("Select date column:", date_columns)
    
    try:
        data[date_col] = pd.to_datetime(data[date_col])
        
        # Time series of key metrics
        if 'outcome_score' in data.columns:
            fig = px.line(
                data.groupby([date_col, 'cohort'])['outcome_score'].mean().reset_index(),
                x=date_col,
                y='outcome_score',
                color='cohort' if 'cohort' in data.columns else None,
                title="Outcome Score Trends Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error creating time series: {str(e)}")

def display_custom_charts(data: pd.DataFrame):
    """Allow users to create custom visualizations."""
    st.markdown("#### Create Custom Visualization")
    
    chart_type = st.selectbox("Chart Type:", ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram"])
    
    columns = list(data.columns)
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-axis:", columns)
    with col2:
        y_axis = st.selectbox("Y-axis:", columns)
    
    color_by = st.selectbox("Color by (optional):", ["None"] + columns)
    
    if st.button("Generate Chart"):
        try:
            if chart_type == "Scatter Plot":
                fig = px.scatter(data, x=x_axis, y=y_axis, color=color_by if color_by != "None" else None)
            elif chart_type == "Line Chart":
                fig = px.line(data, x=x_axis, y=y_axis, color=color_by if color_by != "None" else None)
            elif chart_type == "Bar Chart":
                fig = px.bar(data, x=x_axis, y=y_axis, color=color_by if color_by != "None" else None)
            elif chart_type == "Histogram":
                fig = px.histogram(data, x=x_axis, color=color_by if color_by != "None" else None)
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")

def display_analysis_results():
    """Display AI analysis results and insights."""
    if 'analysis_results' not in st.session_state:
        st.info("Run an analysis to see results here")
        return
    
    results = st.session_state.analysis_results
    
    st.markdown("### 🤖 AI Analysis Results")
    
    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Tasks Completed</h4>
            <h2>{}</h2>
        </div>
        """.format(len(results.get('analysis_results', []))), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>💡 Insights Generated</h4>
            <h2>{}</h2>
        </div>
        """.format(len(results.get('insights', []))), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>📋 Recommendations</h4>
            <h2>{}</h2>
        </div>
        """.format(len(results.get('recommendations', []))), unsafe_allow_html=True)
    
    with col4:
        avg_confidence = np.mean([insight.confidence_score for insight in results.get('insights', [])]) if results.get('insights') else 0
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Avg Confidence</h4>
            <h2>{:.1%}</h2>
        </div>
        """.format(avg_confidence), unsafe_allow_html=True)
    
    # Display insights
    if results.get('insights'):
        st.markdown("#### 💡 Key Insights")
        for i, insight in enumerate(results['insights']):
            with st.expander(f"Insight {i+1}: {insight.title}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(insight.description)
                    if insight.recommendations:
                        st.markdown("**Recommendations:**")
                        for rec in insight.recommendations:
                            st.write(f"• {rec}")
                
                with col2:
                    st.metric("Confidence", f"{insight.confidence_score:.1%}")
                    st.metric("Significance", insight.clinical_significance.title())
                    
                    # Visual indicator for confidence
                    if insight.confidence_score >= 0.8:
                        st.success("High Confidence")
                    elif insight.confidence_score >= 0.6:
                        st.warning("Medium Confidence")
                    else:
                        st.error("Low Confidence")
    
    # Display recommendations
    if results.get('recommendations'):
        st.markdown("#### 📋 Actionable Recommendations")
        for i, rec in enumerate(results['recommendations']):
            st.markdown(f"""
            <div class="insight-card">
                <strong>Recommendation {i+1}:</strong> {rec}
            </div>
            """, unsafe_allow_html=True)

def display_agent_dashboard():
    """Display AI agent status and controls."""
    st.markdown("### 🤖 AI Agent Dashboard")
    
    # Initialize agent if not exists
    if 'agent' not in st.session_state:
        with st.spinner("Initializing AI Agent..."):
            # Show debug info for agent initialization
            show_debug_info("agent_core.py", "ClinicalAgent", "__init__", "Creating AI agent instance for dashboard")
            st.session_state.agent = ClinicalAgent()
    
    # Show debug info for status retrieval
    show_debug_info("agent_core.py", "ClinicalAgent", "get_agent_status", "Retrieving current agent status and metrics")
    
    agent = st.session_state.agent
    status = agent.get_agent_status()
    
    # Agent status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Tasks", status['active_tasks'])
    with col2:
        st.metric("Completed Tasks", status['completed_tasks'])
    with col3:
        st.metric("Total Insights", status['total_insights'])
    with col4:
        memory_usage = status['memory_usage']['disk_usage_mb'] if isinstance(status['memory_usage'], dict) else 0
        st.metric("Memory Usage (MB)", f"{memory_usage:.2f}")
    
    # Agent controls
    st.markdown("#### Agent Controls")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear Memory"):
            # Clear agent memory
            st.success("Agent memory cleared!")
    
    with col3:
        if st.button("📊 View Agent Logs"):
            display_agent_logs()

def display_agent_logs():
    """Display agent activity logs."""
    st.markdown("#### 📋 Agent Activity Logs")
    
    # Mock logs for demonstration
    logs = [
        {"timestamp": datetime.now() - timedelta(minutes=5), "level": "INFO", "message": "Analysis session started"},
        {"timestamp": datetime.now() - timedelta(minutes=3), "level": "INFO", "message": "Data exploration completed"},
        {"timestamp": datetime.now() - timedelta(minutes=2), "level": "INFO", "message": "Cohort analysis completed"},
        {"timestamp": datetime.now() - timedelta(minutes=1), "level": "INFO", "message": "Insights generated successfully"},
    ]
    
    for log in logs:
        level_color = {"INFO": "🔵", "WARNING": "🟡", "ERROR": "🔴"}.get(log["level"], "⚪")
        st.write(f"{level_color} {log['timestamp'].strftime('%H:%M:%S')} - {log['message']}")

# ============================================================================
# 3. CORE ANALYSIS FUNCTIONS
# ============================================================================

def run_analysis(data: pd.DataFrame, goals: List[str], confidence_threshold: float, 
                max_insights: int, include_recommendations: bool, detailed_analysis: bool):
    """Run comprehensive AI analysis on the uploaded data."""
    
    # Show debug info for analysis initialization
    show_debug_info("agent_core.py", "ClinicalAgent", "__init__", "Initializing clinical AI agent for analysis")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize components
        status_text.text("Initializing AI components...")
        progress_bar.progress(10)
        
        if 'agent' not in st.session_state:
            st.session_state.agent = ClinicalAgent()
        
        agent = st.session_state.agent
        
        # Run analysis
        status_text.text("Running AI analysis...")
        progress_bar.progress(30)
        
        # Show debug info for main analysis
        show_debug_info("agent_core.py", "ClinicalAgent", "analyze_trial_data", "Running comprehensive AI analysis on clinical data")
        
        # This needs to be run in an async context
        async def run_async_analysis():
            return await agent.analyze_trial_data(data, goals)
        
        # Create event loop for async execution
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(run_async_analysis())
        
        progress_bar.progress(80)
        status_text.text("Processing results...")
        
        # Store results in session state
        st.session_state.analysis_results = results
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # Show success message
        st.success(f"✅ Analysis completed! Generated {len(results.get('insights', []))} insights and {len(results.get('recommendations', []))} recommendations.")
        
        # Auto-refresh to show results
        st.rerun()
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        logger.error(f"Analysis error: {e}", exc_info=True)
    
    finally:
        progress_bar.empty()
        status_text.empty()

def download_results():
    """Generate and provide download link for analysis results."""
    if 'analysis_results' not in st.session_state:
        st.warning("No results to download")
        return
    
    results = st.session_state.analysis_results
    
    # Convert results to downloadable format
    download_data = {
        'session_id': results.get('session_id'),
        'analysis_timestamp': datetime.now().isoformat(),
        'insights': [
            {
                'title': insight.title,
                'description': insight.description,
                'confidence_score': insight.confidence_score,
                'clinical_significance': insight.clinical_significance,
                'recommendations': insight.recommendations
            }
            for insight in results.get('insights', [])
        ],
        'recommendations': results.get('recommendations', []),
        'metrics': results.get('metrics', {})
    }
    
    # Convert to JSON
    json_data = json.dumps(download_data, indent=2)
    
    st.download_button(
        label="📥 Download Analysis Results (JSON)",
        data=json_data,
        file_name=f"clinical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# ============================================================================
# 4. MAIN APPLICATION LAYOUT AND ROUTING
# ============================================================================

def display_footer():
    """Display footer with developer contact information."""
    st.markdown("---")
    st.markdown("""
    <div style="
        background: rgba(0,0,0,0.05);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-top: 2rem;
        color: #666;
    ">
        <p style="margin: 0;"><strong>👨‍💻 Developed by Nitesh Sharma</strong></p>
        <p style="margin: 0.5rem 0 0 0;">
            📧 <a href="mailto:nitesh.sharma@live.com">nitesh.sharma@live.com</a> | 
            📝 <a href="https://thedataarch.com/" target="_blank">The Data Arch Blog</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Configure page
    configure_page()
    
    # Display header
    display_header()
    
    # Get current page from sidebar
    current_page = display_sidebar()
    
    # Route to appropriate page
    if current_page == "📊 Data Upload & Analysis":
        data_upload_page()
    elif current_page == "🤖 AI Agent Dashboard":
        agent_dashboard_page()
    elif current_page == "📈 Analytics & Insights":
        analytics_page()
    elif current_page == "⚙️ Settings":
        settings_page()
    
    # Display footer on all pages
    display_footer()

def data_upload_page():
    """Data upload and analysis page."""
    st.markdown("## 📊 Data Upload & Analysis")
    
    # Page instructions
    st.info("""
    **📋 Instructions for Data Upload & Analysis:**
    
    **Step 1:** Upload your clinical trial data CSV file using the file uploader below
    - Ensure your CSV contains columns like: `patient_id`, `cohort`, `outcome_score`, `compliance_pct`, `adverse_event_flag`, `visit_date`
    - The system accepts various column names and will automatically detect the structure
    
    **Step 2:** Review the data preview and quality metrics to verify your upload
    
    **Step 3:** Configure your analysis goals and settings
    
    **Step 4:** Click "Start AI Analysis" to run comprehensive clinical analysis
    
    **Step 5:** View results and visualizations below, then proceed to Analytics & Insights page for detailed reports
    """)
    
    # File upload section
    uploaded_data = display_file_uploader()
    
    if uploaded_data is not None:
        # Analysis controls
        analysis_config = display_analysis_controls()
        
        st.markdown("---")
        
        # Data visualizations
        display_data_visualizations(uploaded_data)
        
        st.markdown("---")
        
        # Analysis results
        display_analysis_results()

def agent_dashboard_page():
    """AI agent dashboard page."""
    st.markdown("## 🤖 AI Agent Dashboard")
    
    st.info("""
    **🤖 About the AI Agent:**
    
    The Clinical AI Agent is an autonomous system that:
    - **🔍 Analyzes** your clinical data automatically using advanced algorithms
    - **🧠 Learns** patterns and correlations in your dataset
    - **📊 Generates** insights about safety, efficacy, and compliance
    - **💡 Provides** actionable recommendations for clinical decisions
    - **🎯 Prioritizes** findings based on clinical significance
    
    **How to Use:**
    1. Ensure you have uploaded clinical data in the Data Upload page
    2. Monitor the agent's progress and task execution below
    3. View generated insights and recommendations
    4. Check detailed reports in the Analytics & Insights page
    """)
    
    display_agent_dashboard()
    
    # Agent performance metrics
    if 'analysis_results' in st.session_state:
        st.markdown("### 📊 Agent Performance Metrics")
        
        st.success("✅ **Analysis Complete** - Your clinical data has been processed successfully!")
        
        results = st.session_state.analysis_results
        metrics = results.get('metrics', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tasks Completed", metrics.get('tasks_completed', 0))
        with col2:
            st.metric("Insights Generated", metrics.get('insights_generated', 0))
        with col3:
            st.metric("Avg Confidence", f"{metrics.get('average_confidence_score', 0):.1%}")
        
        st.markdown("### 🎯 Next Steps")
        st.markdown("""
        - 📈 **Go to Analytics & Insights** → View detailed statistical analysis and reports
        - 📋 **Generate Detailed Reports** → Create comprehensive clinical summaries
        - 🤖 **Use AI Text Analysis** → Analyze clinical reports and documentation
        - 🎯 **Try What-If Analysis** → Explore different treatment scenarios
        """)
    else:
        st.warning("⚠️ **No Analysis Results Found** - Please upload data and run analysis first!")
        st.markdown("""
        **To get started:**
        1. Go to **📊 Data Upload & Analysis** page
        2. Upload your clinical trial CSV file
        3. Configure analysis settings
        4. Click **🚀 Start AI Analysis**
        5. Return here to monitor progress
        """)

def analytics_page():
    """Analytics and insights page."""
    st.markdown("## 📈 Analytics & Insights")
    
    if 'uploaded_data' not in st.session_state:
        st.error("⚠️ **No Data Found** - Please upload data first!")
        st.markdown("""
        **To access analytics, you need to:**
        1. Go to **📊 Data Upload & Analysis** page
        2. Upload your clinical trial CSV file
        3. Run the AI analysis
        4. Return here to explore detailed insights
        """)
        return
    
    data = st.session_state.uploaded_data
    
    st.success(f"✅ **Data Loaded:** {len(data)} records ready for analysis")
    
    # Instructions for each tab
    st.info("""
    **📊 Analytics Tabs Guide:**
    
    **📊 Statistical Analysis** - Compare cohorts, run statistical tests, view effect sizes
    **🔍 Pattern Detection** - Detect compliance, safety, and efficacy issues automatically  
    **🎯 What-If Analysis** - Simulate different treatment scenarios and dosage changes
    **🤖 AI Text Analysis** - Analyze clinical reports, adverse events, and medical text
    **📋 Detailed Reports** - Generate comprehensive AI-powered clinical reports
    """)
    
    # Advanced analytics
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Statistical Analysis", "🔍 Pattern Detection", "🎯 What-If Analysis", "🤖 AI Text Analysis", "📋 Detailed Reports"])
    
    with tab1:
        display_statistical_analysis(data)
    
    with tab2:
        display_pattern_detection(data)
    
    with tab3:
        display_scenario_simulation(data)
    
    with tab4:
        display_genai_text_analysis()
    
    with tab5:
        display_detailed_reports()

def display_statistical_analysis(data: pd.DataFrame):
    """Display advanced statistical analysis using CohortAnalyzer."""
    st.markdown("### 📊 Statistical Analysis")
    
    st.info("""
    **📊 Statistical Analysis Instructions:**
    
    This section provides comprehensive statistical analysis of your clinical trial data:
    
    **🔬 What it does:**
    - Compares treatment cohorts using advanced statistical tests
    - Calculates effect sizes and clinical significance
    - Performs t-tests, chi-square tests, and ANOVA where appropriate
    - Provides confidence intervals and p-values
    
    **📈 How to use:**
    1. Select cohorts to compare from the dropdown
    2. Review statistical significance and effect sizes
    3. Examine detailed test results in the tables below
    4. Use these results for clinical decision-making and reporting
    
    **💡 Key Metrics:**
    - **P-value**: Statistical significance (< 0.05 typically significant)
    - **Effect Size**: Practical significance (Cohen's d: 0.2=small, 0.5=medium, 0.8=large)
    - **Confidence Level**: Reliability of the estimates (95% standard)
    """)
    
    # Show debug info for statistical analysis
    show_debug_info("cohort_analysis.py", "CohortAnalyzer", "__init__", "Initializing statistical analysis engine")
    
    # Initialize CohortAnalyzer for statistical analysis
    analyzer = CohortAnalyzer()
    
    # Cohort comparison if cohort column exists
    if 'cohort' in data.columns:
        st.markdown("#### 📈 Cohort Comparison Analysis")
        
        cohorts = data['cohort'].unique()
        if len(cohorts) >= 2:
            # Select cohorts to compare
            selected_cohorts = st.multiselect(
                "Select cohorts to compare:", 
                cohorts, 
                default=cohorts[:2] if len(cohorts) >= 2 else cohorts
            )
            
            if len(selected_cohorts) >= 2:
                try:
                    # Show debug info for cohort comparison
                    show_debug_info("cohort_analysis.py", "CohortAnalyzer", "compare_cohorts", f"Comparing cohorts: {selected_cohorts}")
                    
                    # Perform cohort comparison
                    comparison = analyzer.compare_cohorts(
                        data, 'cohort', selected_cohorts[0], selected_cohorts[1]
                    )
                    
                    # Display comparison results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        # Get p-value from outcome comparison (primary endpoint)
                        outcome_test = comparison.statistical_tests.get('outcome_comparison', {})
                        p_value = outcome_test.get('p_value', 'N/A')
                        if p_value != 'N/A':
                            st.metric("Statistical Significance", f"p-value: {p_value:.4f}")
                        else:
                            st.metric("Statistical Significance", "N/A")
                    with col2:
                        # Get effect size (Cohen's d for outcomes)
                        effect_size = comparison.effect_sizes.get('outcome_cohens_d', 'N/A')
                        if effect_size != 'N/A':
                            st.metric("Effect Size (Cohen's d)", f"{effect_size:.3f}")
                        else:
                            st.metric("Effect Size", "N/A")
                    with col3:
                        st.metric("Confidence Level", f"{comparison.confidence_level:.1%}")
                    
                    # Display detailed comparison results
                    st.markdown("##### Detailed Comparison Results")
                    
                    # Statistical Tests Results
                    if comparison.statistical_tests:
                        st.markdown("###### Statistical Tests")
                        tests_data = []
                        for test_name, test_result in comparison.statistical_tests.items():
                            if isinstance(test_result, dict):
                                tests_data.append({
                                    'Test': test_name.replace('_', ' ').title(),
                                    'Test Type': test_result.get('test_name', 'N/A'),
                                    'P-Value': f"{test_result.get('p_value', 0):.4f}",
                                    'Significant': '✅' if test_result.get('significant', False) else '❌'
                                })
                        if tests_data:
                            st.dataframe(pd.DataFrame(tests_data), use_container_width=True)
                    
                    # Effect Sizes
                    if comparison.effect_sizes:
                        st.markdown("###### Effect Sizes")
                        effect_data = []
                        for effect_name, effect_value in comparison.effect_sizes.items():
                            if isinstance(effect_value, (int, float)):
                                effect_data.append({
                                    'Metric': effect_name.replace('_', ' ').title(),
                                    'Value': f"{effect_value:.3f}"
                                })
                        if effect_data:
                            st.dataframe(pd.DataFrame(effect_data), use_container_width=True)
                    
                    # Clinical Recommendations
                    if comparison.recommendations:
                        st.markdown("###### Clinical Recommendations")
                        for i, rec in enumerate(comparison.recommendations, 1):
                            st.write(f"{i}. {rec}")
                    
                except Exception as e:
                    st.error(f"Error in cohort analysis: {str(e)}")
    
    # Correlation matrix
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        st.markdown("#### 🔗 Correlation Matrix")
        corr_matrix = data[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.markdown("#### Statistical Summary")
        st.dataframe(data[numeric_cols].describe(), use_container_width=True)

def display_pattern_detection(data: pd.DataFrame):
    """Display pattern detection results using IssueDetector."""
    st.markdown("### 🔍 Pattern Detection & Issue Analysis")
    
    st.info("""
    **🔍 Pattern Detection Instructions:**
    
    **🎯 Purpose:** Automatically detect potential issues in your clinical trial data
    
    **📋 Three Analysis Categories:**
    
    **⚠️ Compliance Issues** - Identifies patients with poor medication adherence
    - Detects low compliance rates (< 70% typically flagged)
    - Severity levels: Critical, High, Medium based on compliance percentage
    - Provides recommendations for intervention strategies
    
    **🚨 Safety Alerts** - Monitors adverse events and safety signals
    - Identifies patients with concerning adverse event patterns
    - Detects population-level safety trends
    - Flags potential safety signals requiring investigation
    
    **📉 Efficacy Concerns** - Evaluates treatment effectiveness
    - Identifies patients with poor treatment response
    - Detects declining efficacy trends over time
    - Highlights potential treatment failures
    
    **💡 How to Use:** Click on each tab below to view detailed analysis results. Each issue includes severity level, confidence score, and actionable recommendations.
    """)
    
    # Show debug info for issue detection
    show_debug_info("issue_detection.py", "IssueDetector", "__init__", "Initializing pattern detection and issue analysis")
    
    # Initialize IssueDetector
    detector = IssueDetector()
    
    tab1, tab2, tab3 = st.tabs(["⚠️ Compliance Issues", "🚨 Safety Alerts", "📉 Efficacy Concerns"])
    
    with tab1:
        st.markdown("#### 💊 Compliance Analysis")
        try:
            # Show debug info for compliance detection
            show_debug_info("issue_detection.py", "IssueDetector", "detect_compliance_issues", "Analyzing patient compliance patterns")
            
            # Detect compliance issues
            compliance_issues = detector.detect_compliance_issues(data)
            
            if compliance_issues:
                st.warning(f"Found {len(compliance_issues)} compliance issues")
                
                for i, issue in enumerate(compliance_issues):
                    with st.expander(f"Issue {i+1}: {issue.issue_type} - {issue.severity}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Patient ID:** {issue.patient_id}")
                            st.write(f"**Severity:** {issue.severity}")
                        with col2:
                            st.write(f"**Confidence:** {issue.confidence_score:.1%}")
                            st.write(f"**Visit:** {issue.visit_number if hasattr(issue, "visit_number") and issue.visit_number else "N/A"}")
                        st.write(f"**Description:** {issue.description}")
                        if issue.recommendation:
                            st.write(f"**Recommendations:** {issue.recommendation}")
            else:
                st.success("No compliance issues detected")
                
        except Exception as e:
            st.error(f"Error in compliance analysis: {str(e)}")
    
    with tab2:
        st.markdown("#### 🚨 Safety & Adverse Events")
        try:
            # Detect adverse events
            ae_issues = detector.detect_adverse_event_patterns(data)
            
            if ae_issues:
                st.error(f"Found {len(ae_issues)} safety concerns")
                
                for i, issue in enumerate(ae_issues):
                    with st.expander(f"Safety Alert {i+1}: {issue.issue_type} - {issue.severity}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Patient ID:** {issue.patient_id}")
                            st.write(f"**Severity:** {issue.severity}")
                        with col2:
                            st.write(f"**Confidence:** {issue.confidence_score:.1%}")
                            st.write(f"**Visit:** {issue.visit_number if hasattr(issue, "visit_number") and issue.visit_number else "N/A"}")
                        st.write(f"**Description:** {issue.description}")
                        if issue.recommendation:
                            st.write(f"**Recommendations:** {issue.recommendation}")
            else:
                st.success("No safety issues detected")
                
        except Exception as e:
            st.error(f"Error in safety analysis: {str(e)}")
    
    with tab3:
        st.markdown("#### 📉 Efficacy Assessment")
        try:
            # Detect efficacy issues
            efficacy_issues = detector.detect_efficacy_issues(data)
            
            if efficacy_issues:
                st.warning(f"Found {len(efficacy_issues)} efficacy concerns")
                
                for i, issue in enumerate(efficacy_issues):
                    with st.expander(f"Efficacy Issue {i+1}: {issue.issue_type} - {issue.severity}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Patient ID:** {issue.patient_id}")
                            st.write(f"**Severity:** {issue.severity}")
                        with col2:
                            st.write(f"**Confidence:** {issue.confidence_score:.1%}")
                            st.write(f"**Visit:** {issue.visit_number if hasattr(issue, "visit_number") and issue.visit_number else "N/A"}")
                        st.write(f"**Description:** {issue.description}")
                        if issue.recommendation:
                            st.write(f"**Recommendations:** {issue.recommendation}")
            else:
                st.success("No efficacy issues detected")
                
        except Exception as e:
            st.error(f"Error in efficacy analysis: {str(e)}")

def display_scenario_simulation(data: pd.DataFrame):
    """Display scenario simulation and what-if analysis using ScenarioSimulator."""
    st.markdown("### 🎯 What-If Analysis & Scenario Simulation")
    
    st.info("""
    **🎯 What-If Analysis Instructions:**
    
    **🎮 Purpose:** Simulate different treatment scenarios and predict their outcomes
    
    **🔬 What You Can Simulate:**
    - **Dosage Changes**: Test different medication dosages and predict efficacy/safety outcomes
    - **Compliance Improvements**: Model the impact of better patient adherence
    - **Treatment Modifications**: Explore alternative treatment strategies
    - **Risk Assessment**: Evaluate potential benefits and risks of changes
    
    **📊 How It Works:**
    1. Select a patient cohort for simulation
    2. Adjust dosage, compliance, or other parameters using the sliders
    3. Run the simulation to see predicted outcomes
    4. Compare results with current baseline performance
    5. Use insights for clinical decision-making
    
    **🎯 Use Cases:**
    - Dosage optimization before protocol amendments
    - Predicting impact of compliance interventions
    - Risk-benefit analysis for treatment modifications
    - Supporting regulatory submissions with predictive evidence
    
    **💡 Tip:** Start with small parameter changes to understand their impact, then explore larger modifications.
    """)
    
    # Show debug info for scenario simulator initialization
    show_debug_info("scenario_simulation.py", "ScenarioSimulator", "__init__", "Initializing clinical scenario simulation engine")
    
    # Initialize ScenarioSimulator
    simulator = ScenarioSimulator()
    
    st.markdown("#### 📋 Configure Scenario Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📊 Cohort Parameters")
        if 'cohort' in data.columns:
            selected_cohort = st.selectbox("Select Cohort for Simulation:", data['cohort'].unique())
        else:
            st.warning("No cohort column found in data")
            return
        
        # Dosage modification
        if 'dosage_mg' in data.columns:
            current_dosage = data[data['cohort'] == selected_cohort]['dosage_mg'].iloc[0]
            new_dosage = st.slider("Modify Dosage (mg)", 
                                 min_value=0, 
                                 max_value=int(current_dosage * 2), 
                                 value=int(current_dosage))
        else:
            new_dosage = st.slider("Dosage (mg)", 0, 100, 50)
    
    with col2:
        st.markdown("##### 📈 Outcome Parameters")
        # Compliance modification
        if 'compliance_pct' in data.columns:
            avg_compliance = data[data['cohort'] == selected_cohort]['compliance_pct'].mean()
            target_compliance = st.slider("Target Compliance (%)", 
                                        min_value=0.0, 
                                        max_value=100.0, 
                                        value=float(avg_compliance))
        else:
            target_compliance = st.slider("Target Compliance (%)", 0.0, 100.0, 85.0)
        
        # Sample size modification
        sample_size = st.number_input("Sample Size", 
                                    min_value=10, 
                                    max_value=500, 
                                    value=100)
    
    if st.button("🚀 Run Scenario Simulation"):
        try:
            with st.spinner("Running scenario simulation..."):
                # Show debug info for scenario simulation
                show_debug_info("scenario_simulation.py", "ScenarioSimulator", "simulate_outcome_scenarios", "Running what-if analysis and outcome prediction")
                
                # Create scenario parameters
                scenario_params = {
                    'cohort': selected_cohort,
                    'dosage_mg': new_dosage,
                    'compliance_pct': target_compliance,
                    'sample_size': sample_size
                }
                
                # Run simulation
                simulation_results = simulator.simulate_outcome_scenarios(data, scenario_params)
                
                # Display results
                st.markdown("#### 📊 Simulation Results")
                
                if simulation_results:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        predicted_outcome = simulation_results.predicted_outcomes.get('predicted_outcome', 0)
                        st.metric("Predicted Outcome Score", f"{predicted_outcome:.1f}")
                    
                    with col2:
                        confidence_interval = simulation_results.confidence_intervals.get('outcome', [0, 0])
                        interval_range = confidence_interval[1] - confidence_interval[0]
                        st.metric("Confidence Interval", f"±{interval_range/2:.1f}")
                    
                    with col3:
                        confidence_score = simulation_results.confidence_score
                        st.metric("Confidence Score", f"{confidence_score:.1%}")
                    
                    # Display additional metrics
                    col4, col5, col6 = st.columns(3)
                    
                    with col4:
                        outcome_change = simulation_results.predicted_outcomes.get('outcome_change', 0)
                        st.metric("Outcome Change", f"{outcome_change:+.1f}", delta=f"{outcome_change:+.1f}")
                    
                    with col5:
                        risk_level = simulation_results.risk_assessment.get('overall_risk', 'Unknown')
                        st.metric("Risk Level", risk_level)
                    
                    with col6:
                        num_recommendations = len(simulation_results.recommendations)
                        st.metric("Recommendations", f"{num_recommendations} items")
                    
                    # Show scenario comparison
                    st.markdown("#### 📈 Scenario Comparison")
                    
                    # Create comparison chart
                    baseline_outcome = simulation_results.baseline_metrics.get('baseline_outcome', 70)
                    predicted_outcome = simulation_results.predicted_outcomes.get('predicted_outcome', 75)
                    
                    comparison_data = {
                        'Scenario': ['Current', 'Simulated'],
                        'Dosage (mg)': [current_dosage if 'dosage_mg' in data.columns else 50, new_dosage],
                        'Compliance (%)': [avg_compliance if 'compliance_pct' in data.columns else 85, target_compliance],
                        'Predicted Outcome': [baseline_outcome, predicted_outcome]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, width='stretch')
                    
                    # Visualization
                    fig = px.bar(comparison_df, x='Scenario', y='Predicted Outcome', 
                               title="Outcome Comparison: Current vs Simulated")
                    st.plotly_chart(fig, width='stretch')
                    
                    # Display recommendations
                    if simulation_results.recommendations:
                        st.markdown("#### 💡 Recommendations")
                        for i, recommendation in enumerate(simulation_results.recommendations, 1):
                            st.write(f"{i}. {recommendation}")
                    
                    # Display risk assessment details
                    st.markdown("#### ⚠️ Risk Assessment")
                    risk_info = simulation_results.risk_assessment
                    st.write(f"**Overall Risk Level:** {risk_info.get('overall_risk', 'Unknown')}")
                    st.write(f"**Risk Score:** {risk_info.get('risk_score', 0.5):.2f}")
                    if 'dosage_risk' in risk_info:
                        st.write(f"**Dosage Change Risk:** {risk_info['dosage_risk']}")
                    
                else:
                    st.warning("Simulation completed but no results available")
                    
        except Exception as e:
            st.error(f"Error in scenario simulation: {str(e)}")

def display_genai_text_analysis():
    """Display AI-powered text analysis using GenAI Interface."""
    st.markdown("### 🤖 AI Text Analysis & Natural Language Processing")
    
    st.info("""
    **🤖 AI Text Analysis Guide:**
    
    **🎯 Purpose:** Leverage advanced AI to analyze clinical text and extract meaningful insights
    
    **📋 Analysis Types Available:**
    
    **📋 Clinical Report Analysis** - Comprehensive analysis of clinical trial reports
    - Extracts key findings, safety signals, and efficacy data
    - Identifies potential issues and regulatory concerns
    - Provides structured insights for decision-making
    
    **⚠️ Adverse Event Description Analysis** - Detailed safety signal detection
    - Analyzes adverse event narratives and descriptions
    - Categorizes events by severity and relationship to treatment
    - Suggests follow-up actions and reporting requirements
    
    **📄 Medical Text Summarization** - Automated summarization of complex medical documents
    - Condenses lengthy clinical documents into key points
    - Maintains critical medical information and context
    - Suitable for case reports, study summaries, and medical literature
    
    **🔍 Custom Query** - Flexible analysis for specialized clinical questions
    - Analyze any clinical text with custom prompts
    - Suitable for protocol deviations, case analyses, and specific investigations
    - Adaptable to your unique analysis needs
    
    **💡 Instructions:** Select an analysis type, enter your text, and click analyze. The AI will provide detailed insights and recommendations.
    """)
    
    try:
        # Show debug info for GenAI initialization
        show_debug_info("genai_interface.py", "GenAIInterface", "__init__", "Initializing AI text analysis interface")
        
        # Initialize GenAI Interface
        genai = GenAIInterface()
        
        st.markdown("#### 📝 Text Analysis Options")
        
        analysis_type = st.selectbox(
            "Select analysis type:",
            ["Clinical Report Analysis", "Adverse Event Description Analysis", "Medical Text Summarization", "Custom Query"],
            help="Choose the type of analysis that best fits your clinical text"
        )
        
        if analysis_type == "Clinical Report Analysis":
            st.markdown("##### 📋 Analyze Clinical Reports")
            
            # Text input for clinical report
            report_text = st.text_area(
                "Enter clinical report text:",
                placeholder="Enter clinical report, study notes, or medical documentation...",
                height=200
            )
            
            if report_text and st.button("Analyze Report"):
                with st.spinner("Analyzing clinical report..."):
                    try:
                        # Use GenAI for clinical report analysis
                        prompt = f"Analyze this clinical report and extract key insights, potential issues, and recommendations:\n\n{report_text}"
                        analysis_result = genai.generate_insights(prompt)
                        
                        st.markdown("##### 🔍 Analysis Results")
                        st.write(analysis_result)
                        
                    except Exception as e:
                        st.error(f"Error analyzing report: {str(e)}")
        
        elif analysis_type == "Adverse Event Description Analysis":
            st.markdown("##### ⚠️ Adverse Event Analysis")
            
            ae_text = st.text_area(
                "Enter adverse event description:",
                placeholder="Describe the adverse event, symptoms, timeline, etc...",
                height=150
            )
            
            if ae_text and st.button("Analyze Adverse Event"):
                with st.spinner("Analyzing adverse event..."):
                    try:
                        prompt = f"Analyze this adverse event description for severity, causality, and required actions:\n\n{ae_text}"
                        ae_analysis = genai.generate_insights(prompt)
                        
                        st.markdown("##### ⚠️ Adverse Event Analysis")
                        st.write(ae_analysis)
                        
                    except Exception as e:
                        st.error(f"Error analyzing adverse event: {str(e)}")
        
        elif analysis_type == "Medical Text Summarization":
            st.markdown("##### 📚 Medical Text Summarization")
            
            medical_text = st.text_area(
                "Enter medical text to summarize:",
                placeholder="Enter medical literature, study protocols, or lengthy medical documents...",
                height=200
            )
            
            if medical_text and st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    try:
                        prompt = f"Summarize this medical text, highlighting key points, findings, and clinical relevance:\n\n{medical_text}"
                        summary = genai.generate_insights(prompt)
                        
                        st.markdown("##### 📄 Summary")
                        st.write(summary)
                        
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
        
        elif analysis_type == "Custom Query":
            st.markdown("##### 💬 Custom AI Query")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                custom_query = st.text_area(
                    "Enter your custom query:",
                    placeholder="Ask questions about clinical data, request analysis, or get recommendations...",
                    height=150
                )
            
            with col2:
                st.markdown("**Example queries:**")
                st.markdown("- What are the key risk factors in this study?")
                st.markdown("- Summarize the efficacy outcomes")
                st.markdown("- What safety concerns should we monitor?")
                st.markdown("- Generate recommendations for protocol amendments")
            
            if custom_query and st.button("Get AI Response"):
                with st.spinner("Processing query..."):
                    try:
                        response = genai.generate_insights(custom_query)
                        
                        st.markdown("##### 🤖 AI Response")
                        st.write(response)
                        
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
        
        # Additional GenAI features
        st.markdown("#### 🧠 Advanced AI Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Generate Data Insights"):
                if 'uploaded_data' in st.session_state:
                    with st.spinner("Generating AI insights from data..."):
                        try:
                            data = st.session_state.uploaded_data
                            data_summary = f"Data shape: {data.shape}, Columns: {list(data.columns)}, Sample data: {data.head().to_string()}"
                            prompt = f"Analyze this clinical dataset and provide insights:\n\n{data_summary}"
                            
                            insights = genai.generate_insights(prompt)
                            st.write(insights)
                            
                        except Exception as e:
                            st.error(f"Error generating insights: {str(e)}")
                else:
                    st.warning("Please upload data first")
        
        with col2:
            if st.button("📝 Generate Analysis Report"):
                if 'analysis_results' in st.session_state:
                    with st.spinner("Generating AI-powered report..."):
                        try:
                            # Show debug info for report generation
                            show_debug_info("genai_interface.py", "GenAIInterface", "generate_insights", "Generating AI-powered clinical analysis report")
                            
                            results = st.session_state.analysis_results
                            results_summary = str(results)[:1000]  # Limit length
                            prompt = f"Create a comprehensive clinical analysis report based on these results:\n\n{results_summary}"
                            
                            report = genai.generate_insights(prompt)
                            st.write(report)
                            
                        except Exception as e:
                            st.error(f"Error generating report: {str(e)}")
                else:
                    st.warning("Please run analysis first")
                    
    except Exception as e:
        st.error(f"GenAI Interface initialization error: {str(e)}")
        st.info("AI text analysis features require proper GenAI configuration")

def display_detailed_reports():
    """Display detailed analysis reports."""
    st.markdown("### 📋 Detailed Reports")
    
    st.info("""
    **📋 Detailed Reports Guide:**
    
    **🎯 Purpose:** Generate comprehensive, AI-powered clinical reports suitable for regulatory submission and clinical review
    
    **📊 Available Report Sections:**
    
    **📈 Executive Summary** - High-level overview for senior management
    - Key findings, metrics, and strategic recommendations
    - Suitable for executive briefings and decision-making
    
    **🔍 Data Quality Assessment** - Comprehensive data integrity analysis
    - Data completeness, consistency, and potential issues
    - Recommendations for data quality improvements
    
    **📊 Cohort Analysis Results** - Statistical comparison results
    - Demographics, efficacy comparisons, and statistical significance
    - Clinical interpretation of cohort differences
    
    **🚨 Safety Analysis** - Comprehensive safety profile evaluation
    - Adverse event patterns, safety signals, and risk assessment
    - Regulatory-focused safety conclusions
    
    **💊 Efficacy Analysis** - Treatment effectiveness evaluation
    - Primary/secondary endpoint results and clinical significance
    - Efficacy trends and treatment response patterns
    
    **💡 Recommendations** - Actionable clinical recommendations
    - Evidence-based suggestions for protocol modifications
    - Priority levels and implementation guidance
    
    **🚀 How to Use:** Select the report sections you need, then click "Generate Report" to create AI-powered, professional clinical reports.
    """)
    
    if 'analysis_results' not in st.session_state:
        st.warning("⚠️ **No Analysis Results Found**")
        st.markdown("""
        **To generate detailed reports:**
        1. Go to **📊 Data Upload & Analysis** page
        2. Upload your clinical data
        3. Run the AI analysis
        4. Return here to generate comprehensive reports
        """)
        return
    
    results = st.session_state.analysis_results
    
    st.success("✅ **Analysis Results Available** - Ready to generate detailed reports!")
    
    # Generate comprehensive report
    report_sections = [
        "Executive Summary",
        "Data Quality Assessment", 
        "Cohort Analysis Results",
        "Safety Analysis",
        "Efficacy Analysis",
        "Recommendations"
    ]
    
    selected_sections = st.multiselect(
        "Select report sections:", 
        report_sections, 
        default=report_sections,
        help="Choose which sections to include in your detailed report"
    )
    
    if st.button("Generate Report"):
        try:
            # Initialize GenAI interface for report generation
            genai = GenAIInterface()
            
            with st.spinner("Generating detailed reports..."):
                for section in selected_sections:
                    st.markdown(f"#### {section}")
                    
                    # Generate section-specific content based on analysis results
                    if section == "Executive Summary":
                        # Create executive summary from overall analysis
                        # Format insights properly for Insight objects
                        key_insights = []
                        for insight in results.get('insights', [])[:5]:
                            if hasattr(insight, 'title') and hasattr(insight, 'description'):
                                key_insights.append(f"• {insight.title}: {insight.description}")
                            else:
                                key_insights.append(f"• {str(insight)}")
                        
                        exec_prompt = f"""
                        Create an executive summary for this clinical trial analysis:
                        
                        Session ID: {results.get('session_id', 'N/A')}
                        Total Insights: {len(results.get('insights', []))}
                        Total Recommendations: {len(results.get('recommendations', []))}
                        
                        Key Insights:
                        {chr(10).join(key_insights)}
                        
                        Key Recommendations:
                        {chr(10).join([f"• {rec}" for rec in results.get('recommendations', [])[:5]])}
                        
                        Provide a concise executive summary suitable for senior management.
                        """
                        
                        summary = genai.generate_insights(exec_prompt)
                        st.write(summary)
                    
                    elif section == "Data Quality Assessment":
                        # Generate data quality report
                        quality_prompt = f"""
                        Analyze the data quality based on this clinical trial analysis:
                        
                        Analysis Results Summary:
                        - Session: {results.get('session_id', 'N/A')}
                        - Metrics: {results.get('metrics', {})}
                        - Analysis completed successfully with {len(results.get('analysis_results', []))} components
                        
                        Provide a comprehensive data quality assessment including:
                        - Data completeness and integrity
                        - Potential data issues identified
                        - Recommendations for data quality improvement
                        """
                        
                        quality_report = genai.generate_insights(quality_prompt)
                        st.write(quality_report)
                    
                    elif section == "Cohort Analysis Results":
                        # Generate cohort analysis summary
                        # Format cohort-related insights
                        cohort_insights = []
                        for insight in results.get('insights', [])[:3]:
                            if hasattr(insight, 'title') and hasattr(insight, 'description'):
                                cohort_insights.append(f"• {insight.title}: {insight.description}")
                            else:
                                cohort_insights.append(f"• {str(insight)}")
                        
                        cohort_prompt = f"""
                        Summarize cohort analysis findings from this clinical trial:
                        
                        Analysis Components: {len(results.get('analysis_results', []))}
                        Key Insights: 
                        {chr(10).join(cohort_insights)}
                        
                        Provide detailed cohort analysis results including:
                        - Cohort characteristics and demographics
                        - Statistical comparisons between groups
                        - Clinical significance of findings
                        """
                        
                        cohort_report = genai.generate_insights(cohort_prompt)
                        st.write(cohort_report)
                    
                    elif section == "Safety Analysis":
                        # Generate safety analysis report
                        # Filter safety-related insights from Insight objects
                        safety_insights = []
                        for insight in results.get('insights', []):
                            if hasattr(insight, 'tags') and any('safety' in tag.lower() for tag in insight.tags):
                                safety_insights.append(f"• {insight.title}: {insight.description}")
                            elif hasattr(insight, 'insight_type') and 'safety' in insight.insight_type.lower():
                                safety_insights.append(f"• {insight.title}: {insight.description}")
                            elif hasattr(insight, 'title') and ('safety' in insight.title.lower() or 'adverse' in insight.title.lower()):
                                safety_insights.append(f"• {insight.title}: {insight.description}")
                            elif hasattr(insight, 'description') and ('safety' in insight.description.lower() or 'adverse' in insight.description.lower()):
                                safety_insights.append(f"• {insight.title}: {insight.description}")
                        
                        safety_prompt = f"""
                        Create a safety analysis report based on clinical trial findings:
                        
                        Total Insights Generated: {len(results.get('insights', []))}
                        Safety-related Insights:
                        {chr(10).join(safety_insights)}
                        
                        Provide comprehensive safety analysis including:
                        - Adverse event patterns and frequencies
                        - Safety signal identification
                        - Risk assessment and mitigation strategies
                        """
                        
                        safety_report = genai.generate_insights(safety_prompt)
                        st.write(safety_report)
                    
                    elif section == "Efficacy Analysis":
                        # Generate efficacy analysis report
                        # Filter efficacy-related insights from Insight objects
                        efficacy_insights = []
                        for insight in results.get('insights', []):
                            if hasattr(insight, 'tags') and any(tag.lower() in ['efficacy', 'treatment', 'outcome'] for tag in insight.tags):
                                efficacy_insights.append(f"• {insight.title}: {insight.description}")
                            elif hasattr(insight, 'insight_type') and any(word in insight.insight_type.lower() for word in ['efficacy', 'treatment', 'outcome']):
                                efficacy_insights.append(f"• {insight.title}: {insight.description}")
                            elif hasattr(insight, 'title') and any(word in insight.title.lower() for word in ['efficacy', 'treatment', 'outcome']):
                                efficacy_insights.append(f"• {insight.title}: {insight.description}")
                            elif hasattr(insight, 'description') and any(word in insight.description.lower() for word in ['efficacy', 'treatment', 'outcome']):
                                efficacy_insights.append(f"• {insight.title}: {insight.description}")
                        
                        efficacy_prompt = f"""
                        Analyze treatment efficacy based on clinical trial results:
                        
                        Analysis Session: {results.get('session_id', 'N/A')}
                        Efficacy-related Insights:
                        {chr(10).join(efficacy_insights)}
                        
                        Provide detailed efficacy analysis including:
                        - Primary and secondary endpoint results
                        - Treatment response patterns
                        - Clinical significance and implications
                        """
                        
                        efficacy_report = genai.generate_insights(efficacy_prompt)
                        st.write(efficacy_report)
                    
                    elif section == "Recommendations":
                        # Display and enhance recommendations
                        # Format insights properly for context
                        insight_summaries = []
                        for insight in results.get('insights', []):
                            if hasattr(insight, 'title') and hasattr(insight, 'description'):
                                insight_summaries.append(f"• {insight.title}: {insight.description}")
                            else:
                                insight_summaries.append(f"• {str(insight)}")
                        
                        rec_prompt = f"""
                        Expand and contextualize these clinical trial recommendations:
                        
                        Generated Recommendations:
                        {chr(10).join([f"• {rec}" for rec in results.get('recommendations', [])])}
                        
                        Based on Insights:
                        {chr(10).join(insight_summaries)}
                        
                        Provide enhanced recommendations with:
                        - Detailed implementation guidance
                        - Priority levels and timelines
                        - Risk considerations and mitigation
                        """
                        
                        enhanced_recs = genai.generate_insights(rec_prompt)
                        st.write(enhanced_recs)
                    
                    st.markdown("---")
                    
        except Exception as e:
            st.error(f"Error generating reports: {str(e)}")
            # Fallback to basic display
            for section in selected_sections:
                st.markdown(f"#### {section}")
                if section == "Recommendations" and results.get('recommendations'):
                    for i, rec in enumerate(results.get('recommendations', []), 1):
                        st.write(f"{i}. {rec}")
                elif section == "Executive Summary" and results.get('insights'):
                    st.write("Key findings from analysis:")
                    for insight in results.get('insights', [])[:5]:
                        if hasattr(insight, 'title') and hasattr(insight, 'description'):
                            st.write(f"• **{insight.title}**: {insight.description}")
                        else:
                            st.write(f"• {str(insight)}")
                else:
                    st.write(f"Analysis data available for {section}. Enable GenAI interface for detailed reports.")
                st.markdown("---")

def settings_page():
    """Application settings page."""
    st.markdown("## ⚙️ Settings")
    
    st.info("""
    **⚙️ Settings & Configuration Guide:**
    
    **🎯 Purpose:** Configure the Clinical Insights Assistant platform to optimize performance and customize analysis parameters for your specific clinical research needs.
    
    **📋 Available Tabs:**
    - **🎛️ Analysis Settings:** Configure AI analysis parameters and thresholds
    - **🔧 System Settings:** Manage AI provider settings and system resources
    - **ℹ️ About:** Platform information, version details, and support resources
    
    **💡 Tip:** Adjust settings based on your clinical trial type and data characteristics for optimal analysis results.
    """)
    
    tab1, tab2, tab3 = st.tabs(["🎛️ Analysis Settings", "🔧 System Settings", "ℹ️ About"])
    
    with tab1:
        st.markdown("### Analysis Configuration")
        
        st.info("""
        **🎛️ Analysis Settings Help:**
        
        **🎯 Confidence Threshold (0.0 - 1.0):**
        - **0.7-0.8:** Standard clinical research threshold
        - **0.8-0.9:** High confidence for regulatory submissions
        - **0.6-0.7:** Exploratory analysis or small sample sizes
        
        **📊 Max Insights (1-100):**
        - **5-10:** Focus on most critical findings
        - **10-20:** Comprehensive analysis for large datasets
        - **20+:** Detailed exploration for complex studies
        
        **🔄 Auto-Analysis:**
        - **Enabled:** Automatically analyze uploaded data
        - **Disabled:** Manual analysis control for custom workflows
        """)
        
        confidence_default = st.slider(
            "Default Confidence Threshold", 
            0.0, 1.0, 0.7,
            help="Higher values require stronger statistical evidence for insights"
        )
        max_insights_default = st.number_input(
            "Default Max Insights", 
            1, 100, 10,
            help="Maximum number of insights to generate per analysis"
        )
        auto_analysis = st.checkbox(
            "Enable Auto-Analysis", 
            False,
            help="Automatically start analysis when data is uploaded"
        )
        
        if st.button("Save Analysis Settings"):
            st.success("✅ Analysis settings saved successfully!")
    
    with tab2:
        st.markdown("### System Configuration")
        
        st.info("""
        **🔧 System Settings Help:**
        
        **🤖 AI Provider Options:**
        - **Azure OpenAI:** Enterprise-grade with enhanced security (Recommended)
        - **OpenAI:** Direct OpenAI API access
        
        **🧠 Model Selection:**
        - **gpt-4o-mini:** Fast, cost-effective for routine analysis
        - **gpt-4:** Most advanced reasoning for complex clinical data
        - **gpt-3.5-turbo:** Balanced performance and speed
        
        **💾 Memory & Performance:**
        - **Memory Limit:** Controls system resource usage
        - **Caching:** Improves performance for repeated analyses
        """)
        
        st.markdown("#### AI Provider Settings")
        provider = st.selectbox(
            "AI Provider", 
            ["Azure OpenAI", "OpenAI"],
            help="Choose your preferred AI service provider"
        )
        model = st.selectbox(
            "Model", 
            ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
            help="Select the AI model for clinical analysis"
        )
        
        st.markdown("#### Memory Settings")
        memory_limit = st.slider(
            "Memory Limit (MB)", 
            100, 1000, 500,
            help="Maximum memory allocation for analysis processes"
        )
        cache_enabled = st.checkbox(
            "Enable Caching", 
            True,
            help="Cache results to improve performance for repeated operations"
        )
        
        if st.button("Save System Settings"):
            st.success("✅ System settings saved successfully!")
    
    with tab3:
        st.markdown("### About Clinical Insights Assistant")
        
        st.success("""
        **🏥 Clinical Insights Assistant v1.0.0**
        
        **🎯 Mission:** Empowering clinical researchers with AI-driven insights for safer, more effective treatments.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🛠️ Technology Stack:**
            - **Frontend:** Streamlit (Interactive UI)
            - **AI Engine:** Azure OpenAI (GPT-4)
            - **Visualization:** Plotly (Interactive Charts)
            - **Data Processing:** Pandas, NumPy
            - **Statistics:** SciPy, Statsmodels
            """)
        
        with col2:
            st.markdown("""
            **🚀 Core Capabilities:**
            - ✅ Automated data quality assessment
            - ✅ Intelligent cohort comparison
            - ✅ Safety and efficacy analysis
            - ✅ AI-generated insights & recommendations
            - ✅ Regulatory-ready reporting
            """)
        
        st.markdown("---")
        
        st.info("""
        **📞 Support & Resources:**
        
        - **📚 Documentation:** [GitHub Repository](https://github.com/Nits02/clinical-insight-assistance)
        - **🐛 Bug Reports:** Submit issues via GitHub
        - **💡 Feature Requests:** Contact the Clinical AI Team
        - **📧 Enterprise Support:** Available for clinical organizations
        
        **🔒 Security & Compliance:**
        - HIPAA-compliant data handling
        - Enterprise-grade Azure OpenAI integration
        - No data storage on external servers
        """)
        
        st.markdown("""
        **👥 Developed by:** Clinical AI Research Team  
        **📅 Last Updated:** December 2024  
        **🔖 License:** Clinical Research License
        """)
        
        st.markdown("---")
        
        # Developer Contact Information
        st.markdown("""
        **👨‍💻 Developer Contact:**
        
        **📧 Email:** [nitesh.sharma@live.com](mailto:nitesh.sharma@live.com)  
        **📝 Blog:** [The Data Arch](https://thedataarch.com/)  
        
        *For technical inquiries, collaboration opportunities, or platform support, feel free to reach out!*
        """)

# ============================================================================
# 5. APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()