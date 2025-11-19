"""
Agent Core Module for Clinical Insights Assistant

This module implements the agentic AI system that autonomously explores findings,
generates insights, and recommends next steps for clinical trial analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import asyncio
from enum import Enum

try:
    from .genai_interface import GenAIInterface
    from .issue_detection import IssueDetector
    from .cohort_analysis import CohortAnalyzer
    from .scenario_simulation import ScenarioSimulator
    from .memory import MemoryManager
except ImportError:
    # Fallback for direct execution
    from genai_interface import GenAIInterface
    from issue_detection import IssueDetector
    from cohort_analysis import CohortAnalyzer
    from scenario_simulation import ScenarioSimulator
    from memory import MemoryManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Enumeration for task priorities."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Enumeration for task statuses."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentTask:
    """Data class representing an agent task."""
    task_id: str
    task_type: str
    description: str
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    parameters: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    dependencies: List[str] = None
    estimated_duration: Optional[int] = None  # in minutes


@dataclass
class Insight:
    """Data class representing a clinical insight."""
    insight_id: str
    insight_type: str
    title: str
    description: str
    confidence_score: float
    clinical_significance: str
    supporting_evidence: List[Dict[str, Any]]
    recommendations: List[str]
    created_at: datetime
    patient_ids: List[str]
    cohorts: List[str]
    tags: List[str]


class ClinicalAgent:
    """
    Autonomous agent for clinical trial analysis and insights generation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Clinical Agent.
        
        Args:
            config (Dict, optional): Configuration dictionary for the agent.
        """
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.genai = GenAIInterface()
        self.issue_detector = IssueDetector()
        self.cohort_analyzer = CohortAnalyzer()
        self.scenario_simulator = ScenarioSimulator()
        self.memory = MemoryManager()
        
        # Agent state
        self.task_queue = []
        self.active_tasks = {}
        self.completed_tasks = []
        self.insights = []
        self.exploration_history = []
        
        # Performance metrics
        self.metrics = {
            'tasks_completed': 0,
            'insights_generated': 0,
            'recommendations_made': 0,
            'analysis_sessions': 0,
            'average_confidence_score': 0.0
        }
        
        logger.info("Clinical Agent initialized")
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration for the agent.
        
        Returns:
            Dict: Default configuration parameters.
        """
        return {
            # Agent behavior parameters
            'agent_config': {
                'max_concurrent_tasks': 3,
                'task_timeout_minutes': 30,
                'exploration_depth': 3,
                'min_confidence_threshold': 0.6,
                'auto_exploration_enabled': True,
                'insight_generation_interval': 300  # seconds
            },
            
            # Analysis thresholds
            'analysis_thresholds': {
                'significant_difference_threshold': 5.0,
                'high_risk_threshold': 0.3,
                'low_compliance_threshold': 80.0,
                'high_adverse_event_rate': 0.15,
                'outcome_improvement_threshold': 3.0
            },
            
            # Exploration parameters
            'exploration_config': {
                'follow_up_probability': 0.7,
                'cross_analysis_probability': 0.5,
                'simulation_probability': 0.6,
                'max_exploration_branches': 5,
                'exploration_time_limit': 1800  # seconds
            },
            
            # Insight generation parameters
            'insight_config': {
                'min_supporting_evidence': 2,
                'confidence_weight_factors': {
                    'data_quality': 0.3,
                    'statistical_significance': 0.4,
                    'clinical_relevance': 0.3
                },
                'insight_categories': [
                    'safety_concern',
                    'efficacy_finding',
                    'compliance_issue',
                    'cohort_difference',
                    'dosage_optimization',
                    'protocol_recommendation'
                ]
            }
        }
    
    async def analyze_trial_data(self, data: pd.DataFrame, analysis_goals: List[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive autonomous analysis of trial data.
        
        Args:
            data (pd.DataFrame): Clinical trial data.
            analysis_goals (List[str], optional): Specific analysis goals.
            
        Returns:
            Dict[str, Any]: Analysis results and insights.
        """
        logger.info("Starting autonomous trial data analysis")
        
        # Initialize analysis session
        session_id = f"ANALYSIS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metrics['analysis_sessions'] += 1
        
        # Store data in memory
        self.memory.store_data('trial_data', data)
        
        # Create initial analysis tasks
        initial_tasks = self._create_initial_analysis_tasks(data, analysis_goals)
        
        # Add tasks to queue
        for task in initial_tasks:
            await self._add_task(task)
        
        # Execute analysis workflow
        analysis_results = await self._execute_analysis_workflow(session_id)
        
        # Generate final insights
        final_insights = await self._generate_session_insights(session_id, analysis_results)
        
        # Update metrics
        self._update_metrics()
        
        logger.info(f"Analysis session {session_id} completed")
        
        return {
            'session_id': session_id,
            'analysis_results': analysis_results,
            'insights': final_insights,
            'recommendations': self._generate_session_recommendations(final_insights),
            'metrics': self.metrics.copy(),
            'exploration_paths': self.exploration_history[-10:]  # Last 10 exploration paths
        }
    
    def analyze_trial_data_sync(self, data: pd.DataFrame, analysis_goals: List[str] = None) -> Dict[str, Any]:
        """
        Synchronous wrapper for analyze_trial_data to avoid blocking Streamlit UI.
        
        Args:
            data (pd.DataFrame): Clinical trial data.
            analysis_goals (List[str], optional): Specific analysis goals.
            
        Returns:
            Dict[str, Any]: Analysis results and insights.
        """
        import asyncio
        import threading
        
        result = {}
        exception = None
        
        def run_async_in_thread():
            nonlocal result, exception
            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Run the async analysis
                    result = loop.run_until_complete(self.analyze_trial_data(data, analysis_goals))
                except Exception as e:
                    exception = e
                finally:
                    loop.close()
            except Exception as e:
                exception = e
        
        # Run in a separate thread to avoid event loop conflicts
        thread = threading.Thread(target=run_async_in_thread)
        thread.daemon = True  # Allow main thread to exit
        thread.start()
        thread.join(timeout=300)  # 5 minute timeout
        
        if thread.is_alive():
            # Thread is still running, analysis timed out
            raise TimeoutError("Analysis timed out after 5 minutes")
        
        if exception:
            raise exception
        
        return result
    
    def _create_initial_analysis_tasks(self, data: pd.DataFrame, analysis_goals: List[str] = None) -> List[AgentTask]:
        """
        Create initial analysis tasks based on data characteristics.
        
        Args:
            data (pd.DataFrame): Clinical trial data.
            analysis_goals (List[str], optional): Specific analysis goals.
            
        Returns:
            List[AgentTask]: List of initial tasks.
        """
        tasks = []
        current_time = datetime.now()
        
        # Basic data exploration task
        tasks.append(AgentTask(
            task_id=f"EXPLORE_{current_time.strftime('%H%M%S')}_001",
            task_type="data_exploration",
            description="Perform initial data exploration and quality assessment",
            priority=TaskPriority.HIGH,
            status=TaskStatus.PENDING,
            created_at=current_time,
            updated_at=current_time,
            parameters={'data_shape': data.shape, 'columns': list(data.columns)},
            estimated_duration=5
        ))
        
        # Issue detection task
        tasks.append(AgentTask(
            task_id=f"ISSUES_{current_time.strftime('%H%M%S')}_002",
            task_type="issue_detection",
            description="Detect compliance, safety, and efficacy issues",
            priority=TaskPriority.HIGH,
            status=TaskStatus.PENDING,
            created_at=current_time,
            updated_at=current_time,
            parameters={'detection_types': ['compliance', 'safety', 'efficacy']},
            estimated_duration=10
        ))
        
        # Cohort analysis task (if cohorts exist)
        if 'cohort' in data.columns:
            unique_cohorts = data['cohort'].unique()
            if len(unique_cohorts) >= 2:
                tasks.append(AgentTask(
                    task_id=f"COHORT_{current_time.strftime('%H%M%S')}_003",
                    task_type="cohort_analysis",
                    description=f"Compare cohorts: {', '.join(unique_cohorts)}",
                    priority=TaskPriority.MEDIUM,
                    status=TaskStatus.PENDING,
                    created_at=current_time,
                    updated_at=current_time,
                    parameters={'cohorts': list(unique_cohorts)},
                    estimated_duration=15
                ))
        
        # Patient-level analysis task
        if 'patient_id' in data.columns:
            patient_count = data['patient_id'].nunique()
            if patient_count > 0:
                tasks.append(AgentTask(
                    task_id=f"PATIENT_{current_time.strftime('%H%M%S')}_004",
                    task_type="patient_analysis",
                    description=f"Analyze individual patient patterns ({patient_count} patients)",
                    priority=TaskPriority.MEDIUM,
                    status=TaskStatus.PENDING,
                    created_at=current_time,
                    updated_at=current_time,
                    parameters={'patient_count': patient_count},
                    estimated_duration=20
                ))
        
        # Goal-specific tasks
        if analysis_goals:
            for i, goal in enumerate(analysis_goals):
                tasks.append(AgentTask(
                    task_id=f"GOAL_{current_time.strftime('%H%M%S')}_{i+5:03d}",
                    task_type="goal_analysis",
                    description=f"Analyze specific goal: {goal}",
                    priority=TaskPriority.HIGH,
                    status=TaskStatus.PENDING,
                    created_at=current_time,
                    updated_at=current_time,
                    parameters={'goal': goal},
                    estimated_duration=15
                ))
        
        return tasks
    
    async def _add_task(self, task: AgentTask):
        """
        Add a task to the queue.
        
        Args:
            task (AgentTask): Task to add.
        """
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)
        logger.info(f"Added task: {task.task_id} - {task.description}")
    
    async def _execute_analysis_workflow(self, session_id: str) -> Dict[str, Any]:
        """
        Execute the analysis workflow by processing tasks.
        
        Args:
            session_id (str): Analysis session identifier.
            
        Returns:
            Dict[str, Any]: Analysis results.
        """
        results = {}
        max_concurrent = self.config['agent_config']['max_concurrent_tasks']
        
        while self.task_queue or self.active_tasks:
            # Start new tasks if capacity allows
            while (len(self.active_tasks) < max_concurrent and 
                   self.task_queue and 
                   len([t for t in self.active_tasks.values() if t.status == TaskStatus.IN_PROGRESS]) < max_concurrent):
                
                task = self.task_queue.pop(0)
                task.status = TaskStatus.IN_PROGRESS
                task.updated_at = datetime.now()
                self.active_tasks[task.task_id] = task
                
                # Execute task asynchronously
                asyncio.create_task(self._execute_task(task))
            
            # Check for completed tasks
            completed_task_ids = []
            for task_id, task in self.active_tasks.items():
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    completed_task_ids.append(task_id)
                    
                    if task.status == TaskStatus.COMPLETED:
                        results[task_id] = task.results
                        self.completed_tasks.append(task)
                        
                        # Generate follow-up tasks based on results
                        follow_up_tasks = await self._generate_follow_up_tasks(task)
                        for follow_up_task in follow_up_tasks:
                            await self._add_task(follow_up_task)
            
            # Remove completed tasks from active tasks
            for task_id in completed_task_ids:
                del self.active_tasks[task_id]
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.1)
        
        return results
    
    async def _execute_task(self, task: AgentTask):
        """
        Execute a specific task.
        
        Args:
            task (AgentTask): Task to execute.
        """
        try:
            logger.info(f"Executing task: {task.task_id}")
            
            if task.task_type == "data_exploration":
                task.results = await self._execute_data_exploration(task)
            elif task.task_type == "issue_detection":
                task.results = await self._execute_issue_detection(task)
            elif task.task_type == "cohort_analysis":
                task.results = await self._execute_cohort_analysis(task)
            elif task.task_type == "patient_analysis":
                task.results = await self._execute_patient_analysis(task)
            elif task.task_type == "scenario_simulation":
                task.results = await self._execute_scenario_simulation(task)
            elif task.task_type == "goal_analysis":
                task.results = await self._execute_goal_analysis(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            task.status = TaskStatus.COMPLETED
            task.updated_at = datetime.now()
            self.metrics['tasks_completed'] += 1
            
            logger.info(f"Task completed: {task.task_id}")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.updated_at = datetime.now()
            logger.error(f"Task failed: {task.task_id} - {str(e)}")
    
    async def _execute_data_exploration(self, task: AgentTask) -> Dict[str, Any]:
        """
        Execute data exploration task.
        
        Args:
            task (AgentTask): Data exploration task.
            
        Returns:
            Dict[str, Any]: Exploration results.
        """
        data = self.memory.get_data('trial_data')
        
        # Basic data statistics
        exploration_results = {
            'data_shape': data.shape,
            'columns': list(data.columns),
            'missing_data': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict(),
            'numeric_summary': data.describe().to_dict() if len(data.select_dtypes(include=[np.number]).columns) > 0 else {},
            'categorical_summary': {}
        }
        
        # Categorical data summary
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            exploration_results['categorical_summary'][col] = data[col].value_counts().to_dict()
        
        # Data quality assessment
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        exploration_results['data_quality'] = {
            'completeness': 1 - (missing_cells / total_cells),
            'total_records': len(data),
            'missing_percentage': (missing_cells / total_cells) * 100
        }
        
        return exploration_results
    
    async def _execute_issue_detection(self, task: AgentTask) -> Dict[str, Any]:
        """
        Execute issue detection task.
        
        Args:
            task (AgentTask): Issue detection task.
            
        Returns:
            Dict[str, Any]: Issue detection results.
        """
        data = self.memory.get_data('trial_data')
        detection_types = task.parameters.get('detection_types', ['compliance', 'safety', 'efficacy'])
        
        results = {}
        
        for detection_type in detection_types:
            if detection_type == 'compliance':
                results['compliance_issues'] = self.issue_detector.detect_compliance_issues(data)
            elif detection_type == 'safety':
                # Use detect_adverse_event_patterns for safety issues
                results['safety_issues'] = self.issue_detector.detect_adverse_event_patterns(data)
            elif detection_type == 'efficacy':
                results['efficacy_issues'] = self.issue_detector.detect_efficacy_issues(data)
        
        # Generate summary
        total_issues = sum(len(issues) for issues in results.values())
        results['summary'] = {
            'total_issues_detected': total_issues,
            'issue_types': list(results.keys()),
            'severity_distribution': self._analyze_issue_severity(results)
        }
        
        return results
    
    async def _execute_cohort_analysis(self, task: AgentTask) -> Dict[str, Any]:
        """
        Execute cohort analysis task.
        
        Args:
            task (AgentTask): Cohort analysis task.
            
        Returns:
            Dict[str, Any]: Cohort analysis results.
        """
        data = self.memory.get_data('trial_data')
        cohorts = task.parameters.get('cohorts', [])
        
        results = {}
        
        # Pairwise cohort comparisons
        for i in range(len(cohorts)):
            for j in range(i + 1, len(cohorts)):
                cohort_a, cohort_b = cohorts[i], cohorts[j]
                comparison_key = f"{cohort_a}_vs_{cohort_b}"
                
                try:
                    comparison_result = self.cohort_analyzer.compare_cohorts(
                        data, 'cohort', cohort_a, cohort_b
                    )
                    results[comparison_key] = asdict(comparison_result)
                except Exception as e:
                    logger.warning(f"Cohort comparison failed: {cohort_a} vs {cohort_b} - {str(e)}")
                    results[comparison_key] = {'error': str(e)}
        
        # Overall cohort summary
        results['cohort_summary'] = {
            'total_cohorts': len(cohorts),
            'comparisons_performed': len([k for k in results.keys() if 'vs' in k and 'error' not in results[k]]),
            'significant_differences': len([k for k in results.keys() if 'vs' in k and 
                                          'error' not in results[k] and 
                                          any(test.get('significant', False) for test in results[k].get('statistical_tests', {}).values())])
        }
        
        return results
    
    async def _execute_patient_analysis(self, task: AgentTask) -> Dict[str, Any]:
        """
        Execute patient-level analysis task.
        
        Args:
            task (AgentTask): Patient analysis task.
            
        Returns:
            Dict[str, Any]: Patient analysis results.
        """
        data = self.memory.get_data('trial_data')
        patient_count = task.parameters.get('patient_count', 0)
        
        results = {
            'patient_profiles': {},
            'outlier_patients': [],
            'high_risk_patients': [],
            'response_patterns': {}
        }
        
        # Analyze individual patients (sample if too many)
        patients_to_analyze = data['patient_id'].unique()
        if len(patients_to_analyze) > 20:
            patients_to_analyze = np.random.choice(patients_to_analyze, 20, replace=False)
        
        for patient_id in patients_to_analyze:
            patient_data = data[data['patient_id'] == patient_id]
            
            profile = {
                'patient_id': patient_id,
                'total_visits': len(patient_data),
                'duration_days': (patient_data['visit_date'].max() - patient_data['visit_date'].min()).days if 'visit_date' in patient_data.columns else 0
            }
            
            # Add outcome metrics if available
            if 'outcome_score' in patient_data.columns:
                profile['outcome_metrics'] = {
                    'mean_outcome': float(patient_data['outcome_score'].mean()),
                    'outcome_trend': float(patient_data['outcome_score'].iloc[-1] - patient_data['outcome_score'].iloc[0]) if len(patient_data) > 1 else 0,
                    'outcome_variability': float(patient_data['outcome_score'].std())
                }
            
            # Add compliance metrics if available
            if 'compliance_pct' in patient_data.columns:
                profile['compliance_metrics'] = {
                    'mean_compliance': float(patient_data['compliance_pct'].mean()),
                    'compliance_variability': float(patient_data['compliance_pct'].std())
                }
            
            # Add adverse event metrics if available
            if 'adverse_event_flag' in patient_data.columns:
                profile['safety_metrics'] = {
                    'total_adverse_events': int(patient_data['adverse_event_flag'].sum()),
                    'adverse_event_rate': float(patient_data['adverse_event_flag'].mean())
                }
            
            results['patient_profiles'][patient_id] = profile
            
            # Identify outliers and high-risk patients
            if 'outcome_score' in patient_data.columns:
                mean_outcome = patient_data['outcome_score'].mean()
                if mean_outcome < 50:  # Low outcome threshold
                    results['outlier_patients'].append({
                        'patient_id': patient_id,
                        'reason': 'low_outcome',
                        'value': mean_outcome
                    })
            
            if 'adverse_event_flag' in patient_data.columns:
                ae_rate = patient_data['adverse_event_flag'].mean()
                if ae_rate > 0.2:  # High AE rate threshold
                    results['high_risk_patients'].append({
                        'patient_id': patient_id,
                        'reason': 'high_adverse_events',
                        'value': ae_rate
                    })
        
        return results
    
    async def _execute_scenario_simulation(self, task: AgentTask) -> Dict[str, Any]:
        """
        Execute scenario simulation task.
        
        Args:
            task (AgentTask): Scenario simulation task.
            
        Returns:
            Dict[str, Any]: Simulation results.
        """
        data = self.memory.get_data('trial_data')
        simulation_params = task.parameters
        
        patient_id = simulation_params.get('patient_id')
        scenario_type = simulation_params.get('scenario_type', 'dosage_adjustment')
        
        if scenario_type == 'dosage_adjustment':
            current_dosage = simulation_params.get('current_dosage', 50)
            proposed_dosage = simulation_params.get('proposed_dosage', 75)
            
            simulation_result = self.scenario_simulator.simulate_dosage_adjustment(
                data, patient_id, current_dosage, proposed_dosage
            )
            
            return asdict(simulation_result)
        
        # Add other scenario types as needed
        return {'error': f'Unsupported scenario type: {scenario_type}'}
    
    async def _execute_goal_analysis(self, task: AgentTask) -> Dict[str, Any]:
        """
        Execute goal-specific analysis task.
        
        Args:
            task (AgentTask): Goal analysis task.
            
        Returns:
            Dict[str, Any]: Goal analysis results.
        """
        data = self.memory.get_data('trial_data')
        goal = task.parameters.get('goal', '')
        
        # Use GenAI to interpret and analyze the goal
        goal_analysis_prompt = f"""
        Analyze the following clinical trial goal and provide specific analysis recommendations:
        
        Goal: {goal}
        
        Data columns available: {list(data.columns)}
        Data shape: {data.shape}
        
        Provide:
        1. Specific metrics to calculate
        2. Statistical tests to perform
        3. Visualizations to create
        4. Key insights to look for
        """
        
        # Use generate_clinical_insights method instead
        analysis_plan = self.genai.generate_clinical_insights(
            {'goal': goal, 'columns': list(data.columns), 'shape': data.shape},
            {'analysis_type': 'goal_analysis'}
        )
        
        # Execute basic analysis based on goal keywords
        results = {
            'goal': goal,
            'analysis_plan': analysis_plan,
            'findings': {}
        }
        
        # Keyword-based analysis
        if 'safety' in goal.lower():
            if 'adverse_event_flag' in data.columns:
                results['findings']['adverse_event_rate'] = float(data['adverse_event_flag'].mean())
                results['findings']['total_adverse_events'] = int(data['adverse_event_flag'].sum())
        
        if 'efficacy' in goal.lower() or 'outcome' in goal.lower():
            if 'outcome_score' in data.columns:
                results['findings']['mean_outcome'] = float(data['outcome_score'].mean())
                results['findings']['outcome_improvement'] = float(data.groupby('patient_id')['outcome_score'].apply(
                    lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0
                ).mean()) if 'patient_id' in data.columns else 0
        
        if 'compliance' in goal.lower() or 'adherence' in goal.lower():
            if 'compliance_pct' in data.columns:
                results['findings']['mean_compliance'] = float(data['compliance_pct'].mean())
                results['findings']['low_compliance_rate'] = float((data['compliance_pct'] < 80).mean())
        
        return results
    
    async def _generate_follow_up_tasks(self, completed_task: AgentTask) -> List[AgentTask]:
        """
        Generate follow-up tasks based on completed task results.
        
        Args:
            completed_task (AgentTask): Completed task.
            
        Returns:
            List[AgentTask]: List of follow-up tasks.
        """
        follow_up_tasks = []
        current_time = datetime.now()
        
        if not self.config['agent_config']['auto_exploration_enabled']:
            return follow_up_tasks
        
        # Generate follow-ups based on task type and results
        if completed_task.task_type == "issue_detection":
            results = completed_task.results
            
            # If significant issues found, create simulation tasks
            if results.get('summary', {}).get('total_issues_detected', 0) > 0:
                for issue_type, issues in results.items():
                    if issue_type != 'summary' and isinstance(issues, list) and len(issues) > 0:
                        # Create simulation task for addressing issues
                        follow_up_tasks.append(AgentTask(
                            task_id=f"SIM_{current_time.strftime('%H%M%S')}_{len(self.task_queue) + 1:03d}",
                            task_type="scenario_simulation",
                            description=f"Simulate interventions for {issue_type}",
                            priority=TaskPriority.MEDIUM,
                            status=TaskStatus.PENDING,
                            created_at=current_time,
                            updated_at=current_time,
                            parameters={
                                'scenario_type': 'intervention_simulation',
                                'issue_type': issue_type,
                                'issues': issues[:3]  # Limit to top 3 issues
                            },
                            estimated_duration=20
                        ))
        
        elif completed_task.task_type == "cohort_analysis":
            results = completed_task.results
            
            # If significant differences found, explore further
            significant_comparisons = [k for k in results.keys() if 'vs' in k and 
                                     'error' not in results[k] and 
                                     any(test.get('significant', False) for test in results[k].get('statistical_tests', {}).values())]
            
            if significant_comparisons:
                follow_up_tasks.append(AgentTask(
                    task_id=f"DEEP_{current_time.strftime('%H%M%S')}_{len(self.task_queue) + 1:03d}",
                    task_type="deep_cohort_analysis",
                    description="Deep dive into significant cohort differences",
                    priority=TaskPriority.MEDIUM,
                    status=TaskStatus.PENDING,
                    created_at=current_time,
                    updated_at=current_time,
                    parameters={
                        'significant_comparisons': significant_comparisons,
                        'analysis_depth': 'detailed'
                    },
                    estimated_duration=25
                ))
        
        # Limit follow-up tasks to prevent infinite exploration
        max_follow_ups = self.config['exploration_config']['max_exploration_branches']
        return follow_up_tasks[:max_follow_ups]
    
    async def _generate_session_insights(self, session_id: str, analysis_results: Dict[str, Any]) -> List[Insight]:
        """
        Generate insights from analysis session results.
        
        Args:
            session_id (str): Analysis session identifier.
            analysis_results (Dict[str, Any]): Analysis results.
            
        Returns:
            List[Insight]: Generated insights.
        """
        insights = []
        current_time = datetime.now()
        
        # Analyze results and generate insights
        for task_id, results in analysis_results.items():
            task = next((t for t in self.completed_tasks if t.task_id == task_id), None)
            if not task:
                continue
            
            if task.task_type == "issue_detection":
                insights.extend(self._generate_issue_insights(results, current_time))
            elif task.task_type == "cohort_analysis":
                insights.extend(self._generate_cohort_insights(results, current_time))
            elif task.task_type == "patient_analysis":
                insights.extend(self._generate_patient_insights(results, current_time))
        
        # Store insights in memory
        for insight in insights:
            self.insights.append(insight)
            self.memory.store_insight(insight)
        
        self.metrics['insights_generated'] += len(insights)
        
        return insights
    
    def _generate_issue_insights(self, results: Dict[str, Any], timestamp: datetime) -> List[Insight]:
        """
        Generate insights from issue detection results.
        
        Args:
            results (Dict[str, Any]): Issue detection results.
            timestamp (datetime): Timestamp for insights.
            
        Returns:
            List[Insight]: Generated insights.
        """
        insights = []
        
        total_issues = results.get('summary', {}).get('total_issues_detected', 0)
        
        if total_issues > 0:
            insight = Insight(
                insight_id=f"ISSUE_INSIGHT_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                insight_type="safety_concern",
                title=f"Multiple Issues Detected in Trial Data",
                description=f"Analysis identified {total_issues} potential issues across compliance, safety, and efficacy domains.",
                confidence_score=0.8,
                clinical_significance="high" if total_issues > 10 else "medium",
                supporting_evidence=[
                    {
                        'type': 'issue_summary',
                        'data': results.get('summary', {})
                    }
                ],
                recommendations=[
                    "Conduct detailed review of identified issues",
                    "Implement targeted interventions for high-priority issues",
                    "Increase monitoring frequency for affected patients"
                ],
                created_at=timestamp,
                patient_ids=[],
                cohorts=[],
                tags=['issues', 'safety', 'compliance', 'efficacy']
            )
            insights.append(insight)
        
        return insights
    
    def _generate_cohort_insights(self, results: Dict[str, Any], timestamp: datetime) -> List[Insight]:
        """
        Generate insights from cohort analysis results.
        
        Args:
            results (Dict[str, Any]): Cohort analysis results.
            timestamp (datetime): Timestamp for insights.
            
        Returns:
            List[Insight]: Generated insights.
        """
        insights = []
        
        significant_differences = results.get('cohort_summary', {}).get('significant_differences', 0)
        
        if significant_differences > 0:
            insight = Insight(
                insight_id=f"COHORT_INSIGHT_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                insight_type="cohort_difference",
                title=f"Significant Differences Between Cohorts",
                description=f"Statistical analysis revealed {significant_differences} significant differences between treatment cohorts.",
                confidence_score=0.85,
                clinical_significance="high",
                supporting_evidence=[
                    {
                        'type': 'cohort_comparison',
                        'data': results.get('cohort_summary', {})
                    }
                ],
                recommendations=[
                    "Investigate causes of cohort differences",
                    "Consider protocol modifications for underperforming cohorts",
                    "Analyze baseline characteristics for confounding factors"
                ],
                created_at=timestamp,
                patient_ids=[],
                cohorts=[],
                tags=['cohorts', 'efficacy', 'comparison']
            )
            insights.append(insight)
        
        return insights
    
    def _generate_patient_insights(self, results: Dict[str, Any], timestamp: datetime) -> List[Insight]:
        """
        Generate insights from patient analysis results.
        
        Args:
            results (Dict[str, Any]): Patient analysis results.
            timestamp (datetime): Timestamp for insights.
            
        Returns:
            List[Insight]: Generated insights.
        """
        insights = []
        
        high_risk_patients = len(results.get('high_risk_patients', []))
        outlier_patients = len(results.get('outlier_patients', []))
        
        if high_risk_patients > 0 or outlier_patients > 0:
            insight = Insight(
                insight_id=f"PATIENT_INSIGHT_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                insight_type="safety_concern",
                title=f"High-Risk and Outlier Patients Identified",
                description=f"Analysis identified {high_risk_patients} high-risk patients and {outlier_patients} outlier patients requiring attention.",
                confidence_score=0.75,
                clinical_significance="medium" if (high_risk_patients + outlier_patients) < 5 else "high",
                supporting_evidence=[
                    {
                        'type': 'patient_analysis',
                        'data': {
                            'high_risk_count': high_risk_patients,
                            'outlier_count': outlier_patients
                        }
                    }
                ],
                recommendations=[
                    "Review individual patient cases for high-risk patients",
                    "Consider dose adjustments or additional monitoring",
                    "Investigate common factors among outlier patients"
                ],
                created_at=timestamp,
                patient_ids=[p['patient_id'] for p in results.get('high_risk_patients', [])],
                cohorts=[],
                tags=['patients', 'safety', 'outliers']
            )
            insights.append(insight)
        
        return insights
    
    def _generate_session_recommendations(self, insights: List[Insight]) -> List[str]:
        """
        Generate overall session recommendations based on insights.
        
        Args:
            insights (List[Insight]): Generated insights.
            
        Returns:
            List[str]: Session recommendations.
        """
        recommendations = []
        
        # Aggregate recommendations from insights
        all_recommendations = []
        for insight in insights:
            all_recommendations.extend(insight.recommendations)
        
        # Remove duplicates and prioritize
        unique_recommendations = list(set(all_recommendations))
        
        # Add high-level recommendations based on insight patterns
        safety_insights = [i for i in insights if i.insight_type == 'safety_concern']
        efficacy_insights = [i for i in insights if i.insight_type == 'efficacy_finding']
        cohort_insights = [i for i in insights if i.insight_type == 'cohort_difference']
        
        if safety_insights:
            recommendations.append("Prioritize safety monitoring and risk mitigation strategies")
        
        if efficacy_insights:
            recommendations.append("Focus on optimizing treatment protocols for improved efficacy")
        
        if cohort_insights:
            recommendations.append("Investigate and address cohort-specific differences")
        
        # Add general recommendations
        if len(insights) > 5:
            recommendations.append("Consider comprehensive protocol review given multiple findings")
        
        recommendations.extend(unique_recommendations[:5])  # Top 5 specific recommendations
        
        self.metrics['recommendations_made'] += len(recommendations)
        
        return recommendations
    
    def _analyze_issue_severity(self, issues_dict: Dict[str, List]) -> Dict[str, int]:
        """
        Analyze severity distribution of detected issues.
        
        Args:
            issues_dict (Dict[str, List]): Dictionary of issues by type.
            
        Returns:
            Dict[str, int]: Severity distribution.
        """
        severity_counts = {'low': 0, 'medium': 0, 'high': 0}
        
        for issue_type, issues in issues_dict.items():
            if issue_type == 'summary':
                continue
                
            for issue in issues:
                # Handle both IssueAlert objects and dictionaries
                if hasattr(issue, 'severity'):
                    severity = issue.severity
                else:
                    severity = issue.get('severity', 'medium')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return severity_counts
    
    def _update_metrics(self):
        """Update agent performance metrics."""
        if self.insights:
            total_confidence = sum(insight.confidence_score for insight in self.insights)
            self.metrics['average_confidence_score'] = total_confidence / len(self.insights)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get current agent status and metrics.
        
        Returns:
            Dict[str, Any]: Agent status information.
        """
        return {
            'active_tasks': len(self.active_tasks),
            'queued_tasks': len(self.task_queue),
            'completed_tasks': len(self.completed_tasks),
            'total_insights': len(self.insights),
            'metrics': self.metrics.copy(),
            'memory_usage': self.memory.get_memory_stats(),
            'last_activity': datetime.now().isoformat()
        }


def main():
    """
    Main function for testing the Clinical Agent.
    """
    import asyncio
    
    async def test_agent():
        # Create sample data
        np.random.seed(42)
        data = []
        
        for i in range(100):
            patient_id = f"P{(i // 10) + 1:03d}"
            cohort = 'A' if i < 50 else 'B'
            
            # Cohort A has better outcomes but more adverse events
            if cohort == 'A':
                outcome = np.random.normal(85, 8)
                compliance = np.random.normal(90, 5)
                adverse_event = 1 if np.random.random() < 0.15 else 0
            else:  # Cohort B
                outcome = np.random.normal(78, 10)
                compliance = np.random.normal(85, 8)
                adverse_event = 1 if np.random.random() < 0.08 else 0
            
            data.append({
                'patient_id': patient_id,
                'trial_day': (i % 10) + 1,
                'dosage_mg': 50,
                'compliance_pct': np.clip(compliance, 0, 100),
                'adverse_event_flag': adverse_event,
                'outcome_score': np.clip(outcome, 0, 100),
                'cohort': cohort,
                'visit_date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)
            })
        
        df = pd.DataFrame(data)
        
        # Initialize and test agent
        agent = ClinicalAgent()
        
        # Run analysis
        analysis_goals = ["Assess safety profile", "Compare cohort efficacy"]
        results = await agent.analyze_trial_data(df, analysis_goals)
        
        print(f"Analysis completed:")
        print(f"Session ID: {results['session_id']}")
        print(f"Insights generated: {len(results['insights'])}")
        print(f"Recommendations: {len(results['recommendations'])}")
        
        # Print agent status
        status = agent.get_agent_status()
        print(f"\nAgent Status:")
        print(f"Completed tasks: {status['completed_tasks']}")
        print(f"Total insights: {status['total_insights']}")
        print(f"Average confidence: {status['metrics']['average_confidence_score']:.2f}")
    
    # Run the test
    asyncio.run(test_agent())


if __name__ == "__main__":
    main()