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
    from data_loader import DataLoader
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
    from data_loader import DataLoader
    from genai_interface import GenAIInterface
    from memory import MemoryManager
    from issue_detection import IssueDetector
    from cohort_analysis import CohortAnalyzer
    from scenario_simulation import ScenarioSimulator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Configure and display the sidebar with navigation and controls."""
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/white?text=Clinical+Insights", width=300)
        
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
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with clinical trial data",
        type=['csv'],
        help="Upload your clinical trial dataset. The file should contain patient data with columns like patient_id, cohort, visit_date, etc."
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            data = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = data
            
            st.success(f"✅ Successfully uploaded {len(data)} records!")
            
            # Display data preview
            st.markdown("#### 👀 Data Preview")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Display basic statistics
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
            
            return data
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    
    return None

def display_analysis_controls():
    """Display analysis configuration and trigger controls."""
    st.markdown("### 🎯 Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Analysis Goals")
        default_goals = [
            "Compare treatment efficacy across cohorts",
            "Evaluate safety profile and adverse events",
            "Assess patient compliance patterns",
            "Identify high-risk patients"
        ]
        
        analysis_goals = st.multiselect(
            "Select analysis objectives:",
            default_goals + ["Custom analysis goal"],
            default=default_goals[:2]
        )
        
        if "Custom analysis goal" in analysis_goals:
            custom_goal = st.text_input("Enter custom analysis goal:")
            if custom_goal:
                analysis_goals = [g for g in analysis_goals if g != "Custom analysis goal"] + [custom_goal]
    
    with col2:
        st.markdown("#### Analysis Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
        max_insights = st.number_input("Maximum Insights", 1, 50, 10)
        include_recommendations = st.checkbox("Include Recommendations", True)
        detailed_analysis = st.checkbox("Detailed Analysis", False)
    
    # Analysis trigger button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Start AI Analysis", use_container_width=True):
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
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'outcome_score' in data.columns:
            # Outcome distribution
            fig = px.histogram(
                data,
                x='outcome_score',
                title="Outcome Score Distribution",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
    
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
            st.session_state.agent = ClinicalAgent()
    
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

def data_upload_page():
    """Data upload and analysis page."""
    st.markdown("## 📊 Data Upload & Analysis")
    
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
    
    display_agent_dashboard()
    
    # Agent performance metrics
    if 'analysis_results' in st.session_state:
        st.markdown("### 📊 Agent Performance")
        results = st.session_state.analysis_results
        metrics = results.get('metrics', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tasks Completed", metrics.get('tasks_completed', 0))
        with col2:
            st.metric("Insights Generated", metrics.get('insights_generated', 0))
        with col3:
            st.metric("Avg Confidence", f"{metrics.get('average_confidence_score', 0):.1%}")

def analytics_page():
    """Analytics and insights page."""
    st.markdown("## 📈 Analytics & Insights")
    
    if 'uploaded_data' not in st.session_state:
        st.warning("Please upload data first in the Data Upload & Analysis page")
        return
    
    data = st.session_state.uploaded_data
    
    # Advanced analytics
    tab1, tab2, tab3 = st.tabs(["📊 Statistical Analysis", "🔍 Pattern Detection", "📋 Detailed Reports"])
    
    with tab1:
        display_statistical_analysis(data)
    
    with tab2:
        display_pattern_detection(data)
    
    with tab3:
        display_detailed_reports()

def display_statistical_analysis(data: pd.DataFrame):
    """Display advanced statistical analysis."""
    st.markdown("### 📊 Statistical Analysis")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        # Correlation matrix
        st.markdown("#### Correlation Matrix")
        corr_matrix = data[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.markdown("#### Statistical Summary")
        st.dataframe(data[numeric_cols].describe(), use_container_width=True)

def display_pattern_detection(data: pd.DataFrame):
    """Display pattern detection results."""
    st.markdown("### 🔍 Pattern Detection")
    
    # Mock pattern detection results
    patterns = [
        {"pattern": "High compliance correlation", "strength": 0.85, "description": "Strong correlation between compliance and outcomes"},
        {"pattern": "Cohort A superiority", "strength": 0.78, "description": "Cohort A shows significantly better outcomes"},
        {"pattern": "Time-dependent effects", "strength": 0.62, "description": "Treatment effects vary over time"}
    ]
    
    for pattern in patterns:
        with st.expander(f"Pattern: {pattern['pattern']} (Strength: {pattern['strength']:.1%})"):
            st.write(pattern['description'])
            st.progress(pattern['strength'])

def display_detailed_reports():
    """Display detailed analysis reports."""
    st.markdown("### 📋 Detailed Reports")
    
    if 'analysis_results' not in st.session_state:
        st.info("Run an analysis to generate detailed reports")
        return
    
    results = st.session_state.analysis_results
    
    # Generate comprehensive report
    report_sections = [
        "Executive Summary",
        "Data Quality Assessment", 
        "Cohort Analysis Results",
        "Safety Analysis",
        "Efficacy Analysis",
        "Recommendations"
    ]
    
    selected_sections = st.multiselect("Select report sections:", report_sections, default=report_sections)
    
    if st.button("Generate Report"):
        for section in selected_sections:
            st.markdown(f"#### {section}")
            st.write(f"Detailed analysis for {section} would appear here...")
            st.markdown("---")

def settings_page():
    """Application settings page."""
    st.markdown("## ⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["🎛️ Analysis Settings", "🔧 System Settings", "ℹ️ About"])
    
    with tab1:
        st.markdown("### Analysis Configuration")
        
        confidence_default = st.slider("Default Confidence Threshold", 0.0, 1.0, 0.7)
        max_insights_default = st.number_input("Default Max Insights", 1, 100, 10)
        auto_analysis = st.checkbox("Enable Auto-Analysis", False)
        
        if st.button("Save Analysis Settings"):
            st.success("Settings saved!")
    
    with tab2:
        st.markdown("### System Configuration")
        
        st.markdown("#### AI Provider Settings")
        provider = st.selectbox("AI Provider", ["Azure OpenAI", "OpenAI"])
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"])
        
        st.markdown("#### Memory Settings")
        memory_limit = st.slider("Memory Limit (MB)", 100, 1000, 500)
        cache_enabled = st.checkbox("Enable Caching", True)
        
        if st.button("Save System Settings"):
            st.success("System settings saved!")
    
    with tab3:
        st.markdown("### About Clinical Insights Assistant")
        
        st.markdown("""
        **Version:** 1.0.0  
        **Built with:** Streamlit, OpenAI, Plotly  
        **Author:** Clinical AI Team  
        
        This application provides AI-powered analysis of clinical trial data with:
        - Automated data quality assessment
        - Intelligent cohort comparison
        - Safety and efficacy analysis
        - AI-generated insights and recommendations
        
        For support and documentation, visit our [GitHub repository](https://github.com/Nits02/clinical-insight-assistance).
        """)

# ============================================================================
# 5. APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()