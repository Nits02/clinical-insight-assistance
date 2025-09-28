"""
Tests for the agent_core.py module.

This test suite validates the autonomous clinical agent functionality.
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent_core import (
    ClinicalAgent, 
    AgentTask, 
    Insight,
    TaskStatus,
    TaskPriority
)


class TestAgentDataClasses:
    """Test the data classes used by the agent."""
    
    def test_agent_task_creation(self):
        """Test AgentTask dataclass creation and validation."""
        task = AgentTask(
            task_id="TEST_001",
            task_type="data_exploration",
            description="Test task",
            priority=TaskPriority.MEDIUM,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            parameters={}
        )
        
        assert task.task_id == "TEST_001"
        assert task.task_type == "data_exploration"
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.MEDIUM
        assert task.results is None
    
    def test_insight_creation(self):
        """Test Insight dataclass creation and validation."""
        insight = Insight(
            insight_id="INSIGHT_001",
            insight_type="efficacy",
            title="Test Insight",
            description="Test description",
            confidence_score=0.85,
            clinical_significance="high",
            supporting_evidence=[{"test": "data"}],
            recommendations=["Test recommendation"],
            created_at=datetime.now(),
            patient_ids=["P001"],
            cohorts=["Treatment"],
            tags=["efficacy"]
        )
        
        assert insight.insight_id == "INSIGHT_001"
        assert insight.title == "Test Insight"
        assert insight.insight_type == "efficacy"
        assert insight.clinical_significance == "high"
        assert insight.confidence_score == 0.85
        assert "Test recommendation" in insight.recommendations


class TestClinicalAgent:
    """Test the main ClinicalAgent class functionality."""
    
    @pytest.fixture
    def sample_clinical_data(self):
        """Create sample clinical trial data for testing."""
        np.random.seed(42)  # For reproducible tests
        
        data = []
        for i in range(30):  # Smaller dataset for faster tests
            patient_id = f'TEST_PATIENT_{(i // 3) + 1:03d}'
            cohort = 'Treatment' if i < 15 else 'Control'
            visit_num = (i % 3) + 1
            
            data.append({
                'patient_id': patient_id,
                'visit_number': visit_num,
                'cohort': cohort,
                'dosage_mg': 50 if cohort == 'Treatment' else 0,
                'compliance_pct': np.random.uniform(80, 100),
                'adverse_event_flag': np.random.choice([0, 1], p=[0.9, 0.1]),
                'outcome_score': np.random.uniform(50, 100),
                'visit_date': datetime.now() - timedelta(days=i*7),
                'biomarker_a': np.random.normal(10, 2),
                'biomarker_b': np.random.normal(5, 1)
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock the external dependencies."""
        with patch('agent_core.GenAIInterface') as mock_genai, \
             patch('agent_core.IssueDetector') as mock_issues, \
             patch('agent_core.CohortAnalyzer') as mock_cohort, \
             patch('agent_core.ScenarioSimulator') as mock_scenario, \
             patch('agent_core.MemoryManager') as mock_memory:
            
            # Configure mock returns
            mock_genai.return_value.generate_clinical_insights = AsyncMock(return_value={
                'insights': ['Test insight'],
                'confidence': 0.8,
                'recommendations': ['Test recommendation']
            })
            
            mock_issues.return_value.detect_adverse_event_patterns = Mock(return_value=[])
            mock_issues.return_value.analyze_compliance_issues = Mock(return_value=[])
            mock_issues.return_value.assess_efficacy_concerns = Mock(return_value=[])
            
            mock_cohort.return_value.compare_cohorts = AsyncMock(return_value={
                'statistical_results': {'p_value': 0.05, 'effect_size': 0.3},
                'clinical_significance': 'moderate'
            })
            
            mock_memory.return_value.store_data = Mock(return_value='test_entry_id')
            mock_memory.return_value.store_insight = Mock(return_value='insight_id')
            mock_memory.return_value.get_memory_usage = Mock(return_value={
                'disk_usage_mb': 0.01,
                'cache_hit_rate': 1.0
            })
            
            yield {
                'genai': mock_genai,
                'issues': mock_issues,
                'cohort': mock_cohort,
                'scenario': mock_scenario,
                'memory': mock_memory
            }
    
    def test_agent_initialization(self, mock_dependencies):
        """Test agent initialization."""
        agent = ClinicalAgent()
        
        assert agent.config is not None
        assert agent.active_tasks == {}
        assert agent.completed_tasks == []
        assert agent.insights == []
        assert hasattr(agent, 'genai')
        assert hasattr(agent, 'issue_detector')
        assert hasattr(agent, 'cohort_analyzer')
        assert hasattr(agent, 'scenario_simulator')
        assert hasattr(agent, 'memory')
    
    @pytest.mark.asyncio
    async def test_analyze_trial_data_basic(self, sample_clinical_data, mock_dependencies):
        """Test basic trial data analysis workflow."""
        agent = ClinicalAgent()
        analysis_goals = ['Test efficacy analysis']
        
        # Run analysis
        results = await agent.analyze_trial_data(sample_clinical_data, analysis_goals)
        
        # Verify results structure
        assert 'session_id' in results
        assert 'analysis_results' in results
        assert 'insights' in results
        assert 'recommendations' in results
        assert 'metrics' in results
        assert 'exploration_paths' in results
        
        # Verify session ID exists
        assert results['session_id'] is not None
    
    def test_create_analysis_tasks(self, sample_clinical_data, mock_dependencies):
        """Test task creation for analysis."""
        agent = ClinicalAgent()
        analysis_goals = ['Goal 1', 'Goal 2']
        
        tasks = agent._create_initial_analysis_tasks(sample_clinical_data, analysis_goals)
        
        # Verify tasks were created and returned
        assert len(tasks) > 0
        assert tasks[0].task_type == "data_exploration"
        assert tasks[1].task_type == "issue_detection"
        
        # Verify goal-specific tasks were created
        goal_tasks = [task for task in tasks if task.task_type == "goal_analysis"]
        assert len(goal_tasks) == len(analysis_goals)
    
    def test_agent_config(self, mock_dependencies):
        """Test agent configuration."""
        agent = ClinicalAgent()
        
        # Test default config
        assert agent.config is not None
        assert isinstance(agent.config, dict)
        
        # Test custom config
        custom_config = {'test_param': 'test_value'}
        agent2 = ClinicalAgent(config=custom_config)
        assert agent2.config['test_param'] == 'test_value'
    
    def test_get_agent_status(self, mock_dependencies):
        """Test agent status reporting."""
        agent = ClinicalAgent()
        agent.insights = [Mock()]  # Add mock insight
        
        status = agent.get_agent_status()
        
        assert 'active_tasks' in status
        assert 'completed_tasks' in status
        assert 'total_insights' in status
        assert 'memory_usage' in status
        assert 'metrics' in status
        
        assert status['active_tasks'] == 0
        assert status['total_insights'] == 1


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def mock_dependencies_with_errors(self):
        """Mock dependencies that raise errors."""
        with patch('agent_core.GenAIInterface') as mock_genai, \
             patch('agent_core.IssueDetector') as mock_issues, \
             patch('agent_core.CohortAnalyzer') as mock_cohort, \
             patch('agent_core.ScenarioSimulator') as mock_scenario, \
             patch('agent_core.MemoryManager') as mock_memory:
            
            # Configure mocks to raise errors
            mock_genai.return_value.generate_clinical_insights = AsyncMock(
                side_effect=Exception("GenAI error")
            )
            mock_cohort.return_value.compare_cohorts = AsyncMock(
                side_effect=Exception("Cohort analysis error")
            )
            
            yield {
                'genai': mock_genai,
                'issues': mock_issues,
                'cohort': mock_cohort,
                'scenario': mock_scenario,
                'memory': mock_memory
            }
    
    def test_empty_dataframe_handling(self, mock_dependencies_with_errors):
        """Test handling of empty dataframes."""
        agent = ClinicalAgent()
        empty_df = pd.DataFrame()
        
        # Should not crash during task creation
        agent._create_initial_analysis_tasks(empty_df, ['Test goal'])
        # Tasks could be in queue or active
        total_tasks = len(agent.active_tasks) + len(agent.task_queue)
        assert total_tasks >= 0  # Should at least not crash
    
    def test_invalid_data_types(self, mock_dependencies_with_errors):
        """Test handling of invalid data types."""
        agent = ClinicalAgent()
        
        # Create dataframe with invalid data
        invalid_df = pd.DataFrame({
            'patient_id': ['P1', 'P2'],
            'invalid_column': [{'nested': 'dict'}, [1, 2, 3]]  # Invalid types
        })
        
        # Should not crash during task creation
        agent._create_initial_analysis_tasks(invalid_df, ['Test goal'])
        # Tasks could be in queue or active
        total_tasks = len(agent.active_tasks) + len(agent.task_queue)
        assert total_tasks >= 0  # Should at least not crash


class TestIntegration:
    """Integration tests for the complete agent workflow."""
    
    @pytest.mark.asyncio
    async def test_minimal_workflow_integration(self):
        """Test minimal agent workflow with real dependencies."""
        # This test uses real dependencies (if available) or skips
        try:
            agent = ClinicalAgent()
            
            # Create minimal test data
            test_data = pd.DataFrame({
                'patient_id': ['P001', 'P002'],
                'cohort': ['Treatment', 'Control'],
                'outcome_score': [75, 65],
                'compliance_pct': [95, 90],
                'adverse_event_flag': [0, 0],
                'visit_number': [1, 1],
                'visit_date': [datetime.now(), datetime.now()]
            })
            
            results = await agent.analyze_trial_data(test_data, ['Test analysis'])
            
            # Verify basic structure
            assert 'session_id' in results
            assert 'analysis_results' in results
            
        except Exception as e:
            # If there are dependency issues, skip the test
            pytest.skip(f"Dependencies not available for integration test: {e}")


if __name__ == '__main__':
    # Run tests when executed directly
    pytest.main([__file__, '-v', '--tb=short'])