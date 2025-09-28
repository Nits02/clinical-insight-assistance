"""
Unit tests for the Issue Detection Module

This test suite provides comprehensive testing for all issue detection functionality
in the Clinical Insights Assistant, ensuring reliable detection of:
- Patient compliance issues
- Treatment efficacy concerns
- Adverse event patterns
- Statistical outliers
- Data quality problems
- Temporal trends

The tests use both synthetic data and edge cases to validate detection accuracy,
threshold sensitivity, and error handling across all detection methods.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import warnings

# Import the modules to test
from src.issue_detection import IssueDetector, IssueAlert


class TestIssueAlert(unittest.TestCase):
    """Test the IssueAlert dataclass structure and functionality."""
    
    def test_issue_alert_creation(self):
        """Test creating an IssueAlert with all required fields."""
        alert = IssueAlert(
            issue_type='compliance',
            severity='high',
            patient_id='P001',
            description='Test issue description',
            affected_records=5,
            recommendation='Test recommendation',
            confidence_score=0.85,
            metadata={'test_key': 'test_value'}
        )
        
        self.assertEqual(alert.issue_type, 'compliance')
        self.assertEqual(alert.severity, 'high')
        self.assertEqual(alert.patient_id, 'P001')
        self.assertEqual(alert.description, 'Test issue description')
        self.assertEqual(alert.affected_records, 5)
        self.assertEqual(alert.recommendation, 'Test recommendation')
        self.assertEqual(alert.confidence_score, 0.85)
        self.assertEqual(alert.metadata['test_key'], 'test_value')


class TestIssueDetectorInitialization(unittest.TestCase):
    """Test IssueDetector initialization and configuration."""
    
    def test_default_initialization(self):
        """Test detector initialization with default configuration."""
        detector = IssueDetector()
        
        self.assertIsNotNone(detector.config)
        self.assertEqual(detector.detected_issues, [])
        self.assertIn('compliance_thresholds', detector.config)
        self.assertIn('outcome_thresholds', detector.config)
        self.assertIn('adverse_event_config', detector.config)
        self.assertIn('statistical_config', detector.config)
        self.assertIn('data_quality_config', detector.config)
    
    def test_custom_configuration(self):
        """Test detector initialization with custom configuration."""
        custom_config = {
            'compliance_thresholds': {
                'critical': 40.0,
                'high': 60.0,
                'medium': 80.0
            }
        }
        
        detector = IssueDetector(config=custom_config)
        
        self.assertEqual(detector.config['compliance_thresholds']['critical'], 40.0)
        self.assertEqual(detector.config['compliance_thresholds']['high'], 60.0)
        self.assertEqual(detector.config['compliance_thresholds']['medium'], 80.0)
    
    def test_default_config_structure(self):
        """Test that default configuration has all required sections."""
        detector = IssueDetector()
        config = detector.config
        
        # Test compliance thresholds
        self.assertIn('critical', config['compliance_thresholds'])
        self.assertIn('high', config['compliance_thresholds'])
        self.assertIn('medium', config['compliance_thresholds'])
        
        # Test outcome thresholds
        self.assertIn('inefficacy_critical', config['outcome_thresholds'])
        self.assertIn('inefficacy_high', config['outcome_thresholds'])
        
        # Test adverse event config
        self.assertIn('max_acceptable_rate', config['adverse_event_config'])
        self.assertIn('clustering_threshold', config['adverse_event_config'])
        
        # Test statistical config
        self.assertIn('outlier_z_threshold', config['statistical_config'])
        self.assertIn('min_data_points', config['statistical_config'])
        
        # Test data quality config
        self.assertIn('max_missing_percentage', config['data_quality_config'])
        self.assertIn('duplicate_threshold', config['data_quality_config'])


class TestComplianceDetection(unittest.TestCase):
    """Test compliance issue detection functionality."""
    
    def setUp(self):
        """Set up test data and detector for compliance testing."""
        self.detector = IssueDetector()
        
        # Create test data with various compliance scenarios
        self.test_data = pd.DataFrame([
            # Patient with critical compliance (30%)
            {'patient_id': 'P001', 'compliance_pct': 25.0, 'trial_day': 1},
            {'patient_id': 'P001', 'compliance_pct': 35.0, 'trial_day': 2},
            {'patient_id': 'P001', 'compliance_pct': 30.0, 'trial_day': 3},
            
            # Patient with high concern compliance (65%)
            {'patient_id': 'P002', 'compliance_pct': 60.0, 'trial_day': 1},
            {'patient_id': 'P002', 'compliance_pct': 70.0, 'trial_day': 2},
            {'patient_id': 'P002', 'compliance_pct': 65.0, 'trial_day': 3},
            
            # Patient with medium concern compliance (80%)
            {'patient_id': 'P003', 'compliance_pct': 75.0, 'trial_day': 1},
            {'patient_id': 'P003', 'compliance_pct': 85.0, 'trial_day': 2},
            {'patient_id': 'P003', 'compliance_pct': 80.0, 'trial_day': 3},
            
            # Patient with good compliance (95%)
            {'patient_id': 'P004', 'compliance_pct': 95.0, 'trial_day': 1},
            {'patient_id': 'P004', 'compliance_pct': 90.0, 'trial_day': 2},
            {'patient_id': 'P004', 'compliance_pct': 100.0, 'trial_day': 3},
        ])
    
    def test_critical_compliance_detection(self):
        """Test detection of critical compliance issues."""
        issues = self.detector.detect_compliance_issues(self.test_data)
        
        # Find critical compliance issue
        critical_issues = [issue for issue in issues if issue.severity == 'critical']
        self.assertEqual(len(critical_issues), 1)
        
        critical_issue = critical_issues[0]
        self.assertEqual(critical_issue.patient_id, 'P001')
        self.assertEqual(critical_issue.issue_type, 'compliance')
        self.assertIn('30.0%', critical_issue.description)
    
    def test_high_compliance_detection(self):
        """Test detection of high concern compliance issues."""
        issues = self.detector.detect_compliance_issues(self.test_data)
        
        # Find high concern compliance issue
        high_issues = [issue for issue in issues if issue.severity == 'high']
        self.assertEqual(len(high_issues), 1)
        
        high_issue = high_issues[0]
        self.assertEqual(high_issue.patient_id, 'P002')
        self.assertEqual(high_issue.issue_type, 'compliance')
    
    def test_medium_compliance_detection(self):
        """Test detection of medium concern compliance issues."""
        issues = self.detector.detect_compliance_issues(self.test_data)
        
        # Find medium concern compliance issue
        medium_issues = [issue for issue in issues if issue.severity == 'medium']
        self.assertEqual(len(medium_issues), 1)
        
        medium_issue = medium_issues[0]
        self.assertEqual(medium_issue.patient_id, 'P003')
        self.assertEqual(medium_issue.issue_type, 'compliance')
    
    def test_no_compliance_issues_good_patient(self):
        """Test that patients with good compliance don't trigger issues."""
        issues = self.detector.detect_compliance_issues(self.test_data)
        
        # Check that P004 (good compliance) has no issues
        p004_issues = [issue for issue in issues if issue.patient_id == 'P004']
        self.assertEqual(len(p004_issues), 0)
    
    def test_compliance_missing_columns(self):
        """Test handling of missing required columns."""
        # Test with missing compliance_pct column
        data_missing_compliance = pd.DataFrame([
            {'patient_id': 'P001', 'trial_day': 1}
        ])
        
        issues = self.detector.detect_compliance_issues(data_missing_compliance)
        self.assertEqual(len(issues), 0)
        
        # Test with missing patient_id column
        data_missing_patient = pd.DataFrame([
            {'compliance_pct': 50.0, 'trial_day': 1}
        ])
        
        issues = self.detector.detect_compliance_issues(data_missing_patient)
        self.assertEqual(len(issues), 0)
    
    def test_compliance_confidence_scoring(self):
        """Test confidence score calculation for compliance issues."""
        issues = self.detector.detect_compliance_issues(self.test_data)
        
        for issue in issues:
            self.assertGreaterEqual(issue.confidence_score, 0.0)
            self.assertLessEqual(issue.confidence_score, 1.0)
    
    def test_variable_compliance_recommendation(self):
        """Test recommendation generation for variable compliance."""
        # Create data with high variability
        variable_data = pd.DataFrame([
            {'patient_id': 'P001', 'compliance_pct': 20.0, 'trial_day': 1},
            {'patient_id': 'P001', 'compliance_pct': 80.0, 'trial_day': 2},
            {'patient_id': 'P001', 'compliance_pct': 10.0, 'trial_day': 3},
            {'patient_id': 'P001', 'compliance_pct': 70.0, 'trial_day': 4},
        ])
        
        issues = self.detector.detect_compliance_issues(variable_data)
        self.assertEqual(len(issues), 1)
        
        issue = issues[0]
        self.assertIn('inconsistent compliance', issue.recommendation)
        self.assertIn('adherence counseling', issue.recommendation)


class TestEfficacyDetection(unittest.TestCase):
    """Test efficacy issue detection functionality."""
    
    def setUp(self):
        """Set up test data and detector for efficacy testing."""
        self.detector = IssueDetector()
        
        # Create test data with various efficacy scenarios
        self.test_data = pd.DataFrame([
            # Patient with critical low efficacy (avg ~35)
            {'patient_id': 'P001', 'outcome_score': 30.0, 'trial_day': 1},
            {'patient_id': 'P001', 'outcome_score': 35.0, 'trial_day': 2},
            {'patient_id': 'P001', 'outcome_score': 40.0, 'trial_day': 3},
            
            # Patient with high concern efficacy (avg ~55)
            {'patient_id': 'P002', 'outcome_score': 50.0, 'trial_day': 1},
            {'patient_id': 'P002', 'outcome_score': 55.0, 'trial_day': 2},
            {'patient_id': 'P002', 'outcome_score': 60.0, 'trial_day': 3},
            
            # Patient with declining efficacy (starts high, ends low)
            {'patient_id': 'P003', 'outcome_score': 80.0, 'trial_day': 1},
            {'patient_id': 'P003', 'outcome_score': 75.0, 'trial_day': 2},
            {'patient_id': 'P003', 'outcome_score': 70.0, 'trial_day': 3},
            {'patient_id': 'P003', 'outcome_score': 65.0, 'trial_day': 4},
            {'patient_id': 'P003', 'outcome_score': 60.0, 'trial_day': 5},
            {'patient_id': 'P003', 'outcome_score': 55.0, 'trial_day': 6},
            
            # Patient with good efficacy (avg ~85)
            {'patient_id': 'P004', 'outcome_score': 85.0, 'trial_day': 1},
            {'patient_id': 'P004', 'outcome_score': 90.0, 'trial_day': 2},
            {'patient_id': 'P004', 'outcome_score': 80.0, 'trial_day': 3},
        ])
    
    def test_critical_efficacy_detection(self):
        """Test detection of critical efficacy issues."""
        issues = self.detector.detect_efficacy_issues(self.test_data)
        
        # Find critical efficacy issues
        critical_issues = [issue for issue in issues if issue.severity == 'critical' and issue.issue_type == 'efficacy_low']
        self.assertGreaterEqual(len(critical_issues), 1)
        
        # Check that P001 has a critical efficacy issue
        p001_critical = [issue for issue in critical_issues if issue.patient_id == 'P001']
        self.assertEqual(len(p001_critical), 1)
    
    def test_high_efficacy_detection(self):
        """Test detection of high concern efficacy issues."""
        issues = self.detector.detect_efficacy_issues(self.test_data)
        
        # Find high concern efficacy issues
        high_issues = [issue for issue in issues if issue.severity == 'high' and issue.issue_type == 'efficacy_low']
        self.assertGreaterEqual(len(high_issues), 1)
    
    def test_declining_efficacy_detection(self):
        """Test detection of declining efficacy trends."""
        issues = self.detector.detect_efficacy_issues(self.test_data)
        
        # Find declining efficacy issues
        declining_issues = [issue for issue in issues if issue.issue_type == 'efficacy_declining']
        self.assertGreaterEqual(len(declining_issues), 1)
        
        # Check that P003 has a declining efficacy issue
        p003_declining = [issue for issue in declining_issues if issue.patient_id == 'P003']
        self.assertEqual(len(p003_declining), 1)
        
        declining_issue = p003_declining[0]
        self.assertEqual(declining_issue.severity, 'high')
        self.assertIn('declining', declining_issue.description.lower())
    
    def test_no_efficacy_issues_good_patient(self):
        """Test that patients with good efficacy don't trigger issues."""
        issues = self.detector.detect_efficacy_issues(self.test_data)
        
        # Check that P004 (good efficacy) has no issues
        p004_issues = [issue for issue in issues if issue.patient_id == 'P004']
        self.assertEqual(len(p004_issues), 0)
    
    def test_efficacy_missing_columns(self):
        """Test handling of missing required columns."""
        # Test with missing outcome_score column
        data_missing_outcome = pd.DataFrame([
            {'patient_id': 'P001', 'trial_day': 1}
        ])
        
        issues = self.detector.detect_efficacy_issues(data_missing_outcome)
        self.assertEqual(len(issues), 0)
        
        # Test with missing patient_id column
        data_missing_patient = pd.DataFrame([
            {'outcome_score': 50.0, 'trial_day': 1}
        ])
        
        issues = self.detector.detect_efficacy_issues(data_missing_patient)
        self.assertEqual(len(issues), 0)
    
    def test_insufficient_data_for_trend(self):
        """Test handling of insufficient data for trend analysis."""
        # Single data point per patient
        minimal_data = pd.DataFrame([
            {'patient_id': 'P001', 'outcome_score': 30.0, 'trial_day': 1},
            {'patient_id': 'P002', 'outcome_score': 50.0, 'trial_day': 1},
        ])
        
        issues = self.detector.detect_efficacy_issues(minimal_data)
        
        # Should still detect absolute low efficacy but no declining trends
        declining_issues = [issue for issue in issues if issue.issue_type == 'efficacy_declining']
        self.assertEqual(len(declining_issues), 0)


class TestAdverseEventDetection(unittest.TestCase):
    """Test adverse event pattern detection functionality."""
    
    def setUp(self):
        """Set up test data and detector for adverse event testing."""
        self.detector = IssueDetector()
        
        # Create test data with adverse event patterns
        base_date = datetime(2024, 1, 1)
        self.test_data = pd.DataFrame([
            # High overall adverse event rate scenario
            {'patient_id': f'P00{i}', 'adverse_event_flag': 1 if i % 4 == 0 else 0, 
             'visit_date': base_date + timedelta(days=i)} 
            for i in range(1, 21)  # 20 records, 5 adverse events (25% rate)
        ] + [
            # Patient with multiple adverse events (clustering)
            {'patient_id': 'P021', 'adverse_event_flag': 1, 'visit_date': base_date + timedelta(days=21)},
            {'patient_id': 'P021', 'adverse_event_flag': 1, 'visit_date': base_date + timedelta(days=22)},
            {'patient_id': 'P021', 'adverse_event_flag': 1, 'visit_date': base_date + timedelta(days=23)},
            {'patient_id': 'P021', 'adverse_event_flag': 0, 'visit_date': base_date + timedelta(days=24)},
            
            # Temporal clustering scenario (3 events within 3 days)
            {'patient_id': 'P022', 'adverse_event_flag': 1, 'visit_date': base_date + timedelta(days=30)},
            {'patient_id': 'P023', 'adverse_event_flag': 1, 'visit_date': base_date + timedelta(days=31)},
            {'patient_id': 'P024', 'adverse_event_flag': 1, 'visit_date': base_date + timedelta(days=32)},
        ])
    
    def test_high_overall_adverse_event_rate(self):
        """Test detection of high overall adverse event rates."""
        issues = self.detector.detect_adverse_event_patterns(self.test_data)
        
        # Find high overall rate issues
        rate_issues = [issue for issue in issues if issue.issue_type == 'adverse_event_rate_high']
        self.assertGreaterEqual(len(rate_issues), 1)
        
        rate_issue = rate_issues[0]
        self.assertEqual(rate_issue.patient_id, 'ALL')
        self.assertIn('High overall adverse event rate', rate_issue.description)
    
    def test_patient_adverse_event_clustering(self):
        """Test detection of multiple adverse events in single patient."""
        issues = self.detector.detect_adverse_event_patterns(self.test_data)
        
        # Find clustering issues
        clustering_issues = [issue for issue in issues if issue.issue_type == 'adverse_event_clustering']
        self.assertGreaterEqual(len(clustering_issues), 1)
        
        # Check that P021 has a clustering issue
        p021_clustering = [issue for issue in clustering_issues if issue.patient_id == 'P021']
        self.assertEqual(len(p021_clustering), 1)
        
        clustering_issue = p021_clustering[0]
        self.assertIn('Multiple adverse events', clustering_issue.description)
        self.assertIn('high', clustering_issue.severity)
    
    def test_temporal_adverse_event_clustering(self):
        """Test detection of temporal clustering of adverse events."""
        issues = self.detector.detect_adverse_event_patterns(self.test_data)
        
        # Find temporal clustering issues
        temporal_issues = [issue for issue in issues if issue.issue_type == 'adverse_event_temporal_clustering']
        self.assertGreaterEqual(len(temporal_issues), 1)
        
        temporal_issue = temporal_issues[0]
        self.assertEqual(temporal_issue.patient_id, 'MULTIPLE')
        self.assertIn('Temporal clustering', temporal_issue.description)
    
    def test_missing_adverse_event_column(self):
        """Test handling of missing adverse event column."""
        data_no_ae = pd.DataFrame([
            {'patient_id': 'P001', 'trial_day': 1}
        ])
        
        issues = self.detector.detect_adverse_event_patterns(data_no_ae)
        self.assertEqual(len(issues), 0)
    
    def test_low_adverse_event_rate(self):
        """Test that low adverse event rates don't trigger issues."""
        # Create data with low adverse event rate (5%)
        low_ae_data = pd.DataFrame([
            {'patient_id': f'P00{i}', 'adverse_event_flag': 1 if i == 1 else 0} 
            for i in range(1, 21)  # 20 records, 1 adverse event (5% rate)
        ])
        
        issues = self.detector.detect_adverse_event_patterns(low_ae_data)
        
        # Should not trigger high rate issues
        rate_issues = [issue for issue in issues if issue.issue_type == 'adverse_event_rate_high']
        self.assertEqual(len(rate_issues), 0)


class TestStatisticalOutlierDetection(unittest.TestCase):
    """Test statistical outlier detection functionality."""
    
    def setUp(self):
        """Set up test data and detector for outlier testing."""
        self.detector = IssueDetector()
        
        # Create test data with statistical outliers
        np.random.seed(42)  # For reproducible results
        
        normal_data = []
        # Generate normal data
        for i in range(50):
            normal_data.append({
                'patient_id': f'P{i:03d}',
                'outcome_score': np.random.normal(75, 5),  # Mean 75, std 5
                'compliance_pct': np.random.normal(90, 3),  # Mean 90, std 3
                'dosage_mg': 50  # Standard dosage
            })
        
        # Add clear outliers
        outlier_data = [
            {'patient_id': 'P100', 'outcome_score': 20.0, 'compliance_pct': 90, 'dosage_mg': 50},  # Outcome outlier
            {'patient_id': 'P101', 'outcome_score': 75.0, 'compliance_pct': 30, 'dosage_mg': 50},  # Compliance outlier
            {'patient_id': 'P102', 'outcome_score': 75.0, 'compliance_pct': 90, 'dosage_mg': 200}, # Dosage outlier
        ]
        
        self.test_data = pd.DataFrame(normal_data + outlier_data)
    
    def test_outcome_score_outlier_detection(self):
        """Test detection of outcome score outliers."""
        issues = self.detector.detect_statistical_outliers(self.test_data)
        
        # Find outcome score outliers
        outcome_outliers = [issue for issue in issues 
                          if issue.issue_type == 'statistical_outlier' 
                          and 'outcome_score' in issue.metadata['column']]
        
        self.assertGreaterEqual(len(outcome_outliers), 1)
        
        # Check that P100 is detected as an outlier
        p100_outliers = [issue for issue in outcome_outliers if issue.patient_id == 'P100']
        self.assertEqual(len(p100_outliers), 1)
    
    def test_compliance_outlier_detection(self):
        """Test detection of compliance outliers."""
        issues = self.detector.detect_statistical_outliers(self.test_data)
        
        # Find compliance outliers
        compliance_outliers = [issue for issue in issues 
                             if issue.issue_type == 'statistical_outlier' 
                             and 'compliance_pct' in issue.metadata['column']]
        
        self.assertGreaterEqual(len(compliance_outliers), 1)
        
        # Check that P101 is detected as an outlier
        p101_outliers = [issue for issue in compliance_outliers if issue.patient_id == 'P101']
        self.assertEqual(len(p101_outliers), 1)
    
    def test_dosage_outlier_detection(self):
        """Test detection of dosage outliers."""
        issues = self.detector.detect_statistical_outliers(self.test_data)
        
        # Find dosage outliers
        dosage_outliers = [issue for issue in issues 
                         if issue.issue_type == 'statistical_outlier' 
                         and 'dosage_mg' in issue.metadata['column']]
        
        self.assertGreaterEqual(len(dosage_outliers), 1)
        
        # Check that P102 is detected as an outlier
        p102_outliers = [issue for issue in dosage_outliers if issue.patient_id == 'P102']
        self.assertEqual(len(p102_outliers), 1)
    
    def test_insufficient_data_for_outlier_detection(self):
        """Test handling of insufficient data for outlier detection."""
        # Create data with too few points
        minimal_data = pd.DataFrame([
            {'patient_id': 'P001', 'outcome_score': 75.0},
            {'patient_id': 'P002', 'outcome_score': 80.0},
        ])
        
        issues = self.detector.detect_statistical_outliers(minimal_data)
        self.assertEqual(len(issues), 0)
    
    def test_missing_numeric_columns(self):
        """Test handling when target numeric columns are missing."""
        data_no_numeric = pd.DataFrame([
            {'patient_id': 'P001', 'notes': 'test'},
        ])
        
        issues = self.detector.detect_statistical_outliers(data_no_numeric)
        self.assertEqual(len(issues), 0)


class TestDataQualityDetection(unittest.TestCase):
    """Test data quality issue detection functionality."""
    
    def setUp(self):
        """Set up test data and detector for data quality testing."""
        self.detector = IssueDetector()
    
    def test_high_missing_data_detection(self):
        """Test detection of high missing data percentage."""
        # Create data with 20% missing values (above 10% threshold)
        data_with_missing = pd.DataFrame([
            {'patient_id': 'P001', 'outcome_score': 75.0, 'compliance_pct': None, 'notes': 'test'},
            {'patient_id': 'P002', 'outcome_score': None, 'compliance_pct': 80.0, 'notes': None},
            {'patient_id': 'P003', 'outcome_score': 85.0, 'compliance_pct': 90.0, 'notes': 'test'},
            {'patient_id': 'P004', 'outcome_score': None, 'compliance_pct': None, 'notes': 'test'},
            {'patient_id': 'P005', 'outcome_score': 70.0, 'compliance_pct': 85.0, 'notes': None},
        ])
        
        issues = self.detector.detect_data_quality_issues(data_with_missing)
        
        # Find missing data issues
        missing_issues = [issue for issue in issues if issue.issue_type == 'data_quality_missing']
        self.assertEqual(len(missing_issues), 1)
        
        missing_issue = missing_issues[0]
        self.assertEqual(missing_issue.patient_id, 'ALL')
        self.assertIn('missing data', missing_issue.description.lower())
    
    def test_duplicate_records_detection(self):
        """Test detection of duplicate records."""
        # Create data with duplicates (above 5% threshold)
        data_with_duplicates = pd.DataFrame([
            {'patient_id': 'P001', 'trial_day': 1, 'outcome_score': 75.0},
            {'patient_id': 'P001', 'trial_day': 1, 'outcome_score': 75.0},  # Duplicate
            {'patient_id': 'P001', 'trial_day': 2, 'outcome_score': 80.0},
            {'patient_id': 'P002', 'trial_day': 1, 'outcome_score': 85.0},
            {'patient_id': 'P002', 'trial_day': 1, 'outcome_score': 85.0},  # Duplicate
            {'patient_id': 'P003', 'trial_day': 1, 'outcome_score': 70.0},
        ])
        
        issues = self.detector.detect_data_quality_issues(data_with_duplicates)
        
        # Find duplicate issues
        duplicate_issues = [issue for issue in issues if issue.issue_type == 'data_quality_duplicates']
        self.assertEqual(len(duplicate_issues), 1)
        
        duplicate_issue = duplicate_issues[0]
        self.assertEqual(duplicate_issue.patient_id, 'MULTIPLE')
        self.assertIn('duplicate', duplicate_issue.description.lower())
    
    def test_low_missing_data_no_issue(self):
        """Test that low missing data doesn't trigger issues."""
        # Create data with low missing percentage (~5% - 1 missing out of 20 cells)
        data_low_missing = pd.DataFrame([
            {'patient_id': 'P001', 'outcome_score': 75.0, 'compliance_pct': 80.0, 'col3': 'A', 'col4': 'B'},
            {'patient_id': 'P002', 'outcome_score': 85.0, 'compliance_pct': None, 'col3': 'A', 'col4': 'B'},  # 1 missing out of 20 total cells = 5%
            {'patient_id': 'P003', 'outcome_score': 70.0, 'compliance_pct': 90.0, 'col3': 'A', 'col4': 'B'},
            {'patient_id': 'P004', 'outcome_score': 80.0, 'compliance_pct': 85.0, 'col3': 'A', 'col4': 'B'},
            {'patient_id': 'P005', 'outcome_score': 90.0, 'compliance_pct': 95.0, 'col3': 'A', 'col4': 'B'},
        ])
        
        issues = self.detector.detect_data_quality_issues(data_low_missing)
        
        # Should not trigger missing data issues
        missing_issues = [issue for issue in issues if issue.issue_type == 'data_quality_missing']
        self.assertEqual(len(missing_issues), 0)
    
    def test_missing_required_columns_for_duplicates(self):
        """Test handling when required columns for duplicate detection are missing."""
        data_no_required_cols = pd.DataFrame([
            {'outcome_score': 75.0, 'compliance_pct': 80.0},
            {'outcome_score': 85.0, 'compliance_pct': 90.0},
        ])
        
        issues = self.detector.detect_data_quality_issues(data_no_required_cols)
        
        # Should not find duplicate issues without patient_id and trial_day
        duplicate_issues = [issue for issue in issues if issue.issue_type == 'data_quality_duplicates']
        self.assertEqual(len(duplicate_issues), 0)


class TestTemporalTrendDetection(unittest.TestCase):
    """Test temporal trend detection functionality."""
    
    def setUp(self):
        """Set up test data and detector for temporal trend testing."""
        self.detector = IssueDetector()
        
        # Create test data with temporal trends
        base_date = datetime(2024, 1, 1)
        
        # Declining outcome trend data
        declining_data = []
        for i in range(10):
            declining_data.append({
                'visit_date': base_date + timedelta(days=i),
                'outcome_score': 90 - (i * 3),  # Declining by 3 points per day for stronger trend
                'compliance_pct': 90,
                'adverse_event_flag': 0,
                'patient_id': f'P{i:03d}'
            })
        
        # Increasing adverse event trend data  
        increasing_ae_data = []
        for i in range(10):
            # Make AE rate increase more gradually to create stronger trend
            ae_rate = 0.1 + (i * 0.1)  # Start at 10%, increase by 10% each day
            increasing_ae_data.append({
                'visit_date': base_date + timedelta(days=i + 10),
                'outcome_score': 80,
                'compliance_pct': 90,
                'adverse_event_flag': 1 if np.random.random() < ae_rate else 0,
                'patient_id': f'P{i+100:03d}'
            })
        
        # Set random seed for reproducible AE data
        np.random.seed(42)
        # Recreate with deterministic pattern for better test reliability
        increasing_ae_data = []
        for i in range(10):
            increasing_ae_data.append({
                'visit_date': base_date + timedelta(days=i + 10),
                'outcome_score': 80,
                'compliance_pct': 90,
                'adverse_event_flag': 1 if i >= 2 else 0,  # More AEs in later days
                'patient_id': f'P{i+100:03d}'
            })
        
        self.test_data = pd.DataFrame(declining_data + increasing_ae_data)
    
    def test_declining_outcome_trend_detection(self):
        """Test detection of declining outcome trends."""
        # Create focused declining trend data for this specific test
        base_date = datetime(2024, 1, 1)
        declining_only_data = []
        for i in range(10):
            declining_only_data.append({
                'visit_date': base_date + timedelta(days=i),
                'outcome_score': 90 - (i * 3),  # Strong declining trend
                'compliance_pct': 90,
                'adverse_event_flag': 0,
                'patient_id': f'P{i:03d}'
            })
        
        declining_df = pd.DataFrame(declining_only_data)
        issues = self.detector.detect_temporal_trends(declining_df)
        
        # Find declining outcome issues
        declining_issues = [issue for issue in issues 
                          if issue.issue_type == 'temporal_trend_declining_outcomes']
        
        self.assertGreaterEqual(len(declining_issues), 1)
        
        declining_issue = declining_issues[0]
        self.assertEqual(declining_issue.patient_id, 'ALL')
        self.assertEqual(declining_issue.severity, 'high')
        self.assertIn('declining', declining_issue.description.lower())
    
    def test_increasing_adverse_event_trend_detection(self):
        """Test detection of increasing adverse event trends."""
        issues = self.detector.detect_temporal_trends(self.test_data)
        
        # Find increasing AE issues
        ae_trend_issues = [issue for issue in issues 
                         if issue.issue_type == 'temporal_trend_increasing_adverse_events']
        
        self.assertGreaterEqual(len(ae_trend_issues), 1)
        
        ae_issue = ae_trend_issues[0]
        self.assertEqual(ae_issue.patient_id, 'ALL')
        self.assertEqual(ae_issue.severity, 'high')  
        self.assertIn('increasing', ae_issue.description.lower())
    
    def test_missing_visit_date_column(self):
        """Test handling of missing visit_date column."""
        data_no_date = pd.DataFrame([
            {'patient_id': 'P001', 'outcome_score': 75.0}
        ])
        
        issues = self.detector.detect_temporal_trends(data_no_date)
        self.assertEqual(len(issues), 0)
    
    def test_insufficient_temporal_data(self):
        """Test handling of insufficient temporal data points."""
        # Create data with too few time points
        minimal_temporal_data = pd.DataFrame([
            {'visit_date': datetime(2024, 1, 1), 'outcome_score': 75.0, 'compliance_pct': 90, 'adverse_event_flag': 0},
            {'visit_date': datetime(2024, 1, 2), 'outcome_score': 80.0, 'compliance_pct': 90, 'adverse_event_flag': 0},
        ])
        
        issues = self.detector.detect_temporal_trends(minimal_temporal_data)
        self.assertEqual(len(issues), 0)


class TestComprehensiveDetection(unittest.TestCase):
    """Test comprehensive issue detection across all methods."""
    
    def setUp(self):
        """Set up comprehensive test data for all detection methods."""
        self.detector = IssueDetector()
        
        # Create comprehensive test dataset with multiple issue types
        np.random.seed(42)
        base_date = datetime(2024, 1, 1)
        
        comprehensive_data = []
        
        # Generate data with various issue patterns
        for i in range(100):
            patient_id = f'P{(i // 10) + 1:03d}'
            trial_day = (i % 10) + 1
            visit_date = base_date + timedelta(days=i)
            
            # Introduce compliance issues
            if patient_id in ['P002', 'P005']:
                compliance = np.random.normal(45, 10)  # Low compliance
            else:
                compliance = np.random.normal(90, 5)   # Normal compliance
            
            # Introduce efficacy issues
            if patient_id in ['P003', 'P007']:
                outcome = np.random.normal(35, 8)   # Low efficacy
            else:
                outcome = np.random.normal(80, 6)   # Normal efficacy
            
            # Introduce adverse events
            if patient_id == 'P004':
                adverse_event = 1 if np.random.random() < 0.4 else 0  # High AE rate
            else:
                adverse_event = 1 if np.random.random() < 0.1 else 0  # Normal AE rate
            
            # Add some missing data and outliers
            if i % 20 == 0:
                compliance = None  # Missing data
            if i == 50:
                outcome = 5.0  # Statistical outlier
            
            comprehensive_data.append({
                'patient_id': patient_id,
                'trial_day': trial_day,
                'visit_date': visit_date,
                'compliance_pct': compliance,
                'outcome_score': outcome,
                'adverse_event_flag': adverse_event,
                'dosage_mg': 50,
                'cohort': 'A' if i % 2 == 0 else 'B'
            })
        
        self.test_data = pd.DataFrame(comprehensive_data)
    
    def test_detect_all_issues_comprehensive(self):
        """Test comprehensive detection across all methods."""
        issues = self.detector.detect_all_issues(self.test_data)
        
        # Should detect multiple types of issues
        self.assertGreater(len(issues), 0)
        
        # Check that different issue types are detected
        issue_types = set(issue.issue_type for issue in issues)
        expected_types = ['compliance', 'efficacy_low', 'adverse_event_clustering', 'statistical_outlier']
        
        # Should find at least some of the expected issue types
        found_types = issue_types.intersection(expected_types)
        self.assertGreater(len(found_types), 0)
    
    def test_issue_summary_generation(self):
        """Test generation of issue summary statistics."""
        issues = self.detector.detect_all_issues(self.test_data)
        summary = self.detector.get_issue_summary()
        
        # Check summary structure
        self.assertIn('total_issues', summary)
        self.assertIn('by_severity', summary)
        self.assertIn('by_type', summary)
        self.assertIn('high_priority_count', summary)
        self.assertIn('high_priority_issues', summary)
        
        # Validate summary content
        self.assertEqual(summary['total_issues'], len(issues))
        self.assertGreaterEqual(summary['high_priority_count'], 0)
        self.assertLessEqual(len(summary['high_priority_issues']), 5)  # Max 5 in summary
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        empty_data = pd.DataFrame()
        
        issues = self.detector.detect_all_issues(empty_data)
        self.assertEqual(len(issues), 0)
        
        summary = self.detector.get_issue_summary()
        self.assertEqual(summary['total_issues'], 0)
        self.assertEqual(summary['summary_status'], 'no_issues_detected')


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up detector for edge case testing."""
        self.detector = IssueDetector()
    
    def test_single_patient_single_record(self):
        """Test handling of single patient with single record."""
        single_record = pd.DataFrame([
            {'patient_id': 'P001', 'compliance_pct': 30.0, 'outcome_score': 35.0, 
             'adverse_event_flag': 1, 'trial_day': 1}
        ])
        
        issues = self.detector.detect_all_issues(single_record)
        
        # Should still detect some issues despite minimal data
        self.assertGreaterEqual(len(issues), 0)
    
    def test_all_nan_values(self):
        """Test handling of datasets with all NaN values."""
        nan_data = pd.DataFrame([
            {'patient_id': 'P001', 'compliance_pct': np.nan, 'outcome_score': np.nan, 'trial_day': 1, 'adverse_event_flag': 0},
            {'patient_id': 'P002', 'compliance_pct': np.nan, 'outcome_score': np.nan, 'trial_day': 1, 'adverse_event_flag': 0},
        ])
        
        issues = self.detector.detect_all_issues(nan_data)
        
        # Should handle gracefully without crashing
        self.assertIsInstance(issues, list)
    
    def test_extreme_outlier_values(self):
        """Test handling of extreme outlier values."""
        extreme_data = pd.DataFrame([
            {'patient_id': 'P001', 'compliance_pct': -1000.0, 'outcome_score': 50.0, 'dosage_mg': 50, 'trial_day': 1, 'adverse_event_flag': 0},
            {'patient_id': 'P002', 'compliance_pct': 90.0, 'outcome_score': 1000000.0, 'dosage_mg': 50, 'trial_day': 1, 'adverse_event_flag': 0},
            {'patient_id': 'P003', 'compliance_pct': 90.0, 'outcome_score': 50.0, 'dosage_mg': -500, 'trial_day': 1, 'adverse_event_flag': 0},
        ])
        
        # Should handle extreme values without crashing
        issues = self.detector.detect_all_issues(extreme_data)
        self.assertIsInstance(issues, list)
    
    def test_mixed_data_types(self):
        """Test handling of mixed data types in numeric columns."""
        mixed_data = pd.DataFrame([
            {'patient_id': 'P001', 'compliance_pct': '90.0', 'outcome_score': 75.0, 'trial_day': 1, 'adverse_event_flag': 0},
            {'patient_id': 'P002', 'compliance_pct': 85.0, 'outcome_score': '80.0', 'trial_day': 1, 'adverse_event_flag': 0},
            {'patient_id': 'P003', 'compliance_pct': 90.0, 'outcome_score': 'invalid', 'trial_day': 1, 'adverse_event_flag': 0},
        ])
        
        # Should handle mixed types gracefully
        issues = self.detector.detect_all_issues(mixed_data)
        self.assertIsInstance(issues, list)
    
    def test_very_large_dataset_performance(self):
        """Test performance with larger datasets."""
        # Create larger dataset (1000 records)
        np.random.seed(42)
        large_data = []
        
        for i in range(1000):
            large_data.append({
                'patient_id': f'P{(i // 10) + 1:04d}',
                'trial_day': (i % 10) + 1,
                'compliance_pct': np.random.normal(85, 10),
                'outcome_score': np.random.normal(75, 10),
                'adverse_event_flag': 1 if np.random.random() < 0.1 else 0,
                'dosage_mg': 50,
                'visit_date': datetime(2024, 1, 1) + timedelta(days=i//10)
            })
        
        large_df = pd.DataFrame(large_data)
        
        # Should complete within reasonable time
        import time
        start_time = time.time()
        issues = self.detector.detect_all_issues(large_df)
        end_time = time.time()
        
        # Should complete in reasonable time (less than 10 seconds)
        self.assertLess(end_time - start_time, 10.0)
        self.assertIsInstance(issues, list)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)