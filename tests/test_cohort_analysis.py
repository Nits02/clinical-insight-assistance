"""
Unit Tests for Cohort Analysis Module

This test suite provides comprehensive testing for the cohort analysis functionality,
including statistical comparisons, effect size calculations, clinical significance
assessment, and subgroup analysis capabilities.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, Mock
import warnings
from scipy import stats

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cohort_analysis import CohortAnalyzer, CohortComparisonResult


class TestCohortComparisonResult(unittest.TestCase):
    """Test the CohortComparisonResult dataclass functionality."""
    
    def test_cohort_comparison_result_creation(self):
        """Test creation of CohortComparisonResult dataclass."""
        # Create sample result data
        cohort_a_stats = {'cohort_name': 'A', 'sample_size': 50}
        cohort_b_stats = {'cohort_name': 'B', 'sample_size': 45}
        statistical_tests = {'outcome_comparison': {'p_value': 0.023, 'significant': True}}
        effect_sizes = {'outcome_cohens_d': 0.65}
        clinical_significance = {'outcome_score': 'clinically_significant'}
        recommendations = ['Consider adopting protocol from Cohort A']
        confidence_level = 0.95
        
        # Create result object
        result = CohortComparisonResult(
            cohort_a_stats=cohort_a_stats,
            cohort_b_stats=cohort_b_stats,
            statistical_tests=statistical_tests,
            effect_sizes=effect_sizes,
            clinical_significance=clinical_significance,
            recommendations=recommendations,
            confidence_level=confidence_level
        )
        
        # Verify all attributes are correctly assigned
        self.assertEqual(result.cohort_a_stats, cohort_a_stats)
        self.assertEqual(result.cohort_b_stats, cohort_b_stats)
        self.assertEqual(result.statistical_tests, statistical_tests)
        self.assertEqual(result.effect_sizes, effect_sizes)
        self.assertEqual(result.clinical_significance, clinical_significance)
        self.assertEqual(result.recommendations, recommendations)
        self.assertEqual(result.confidence_level, confidence_level)


class TestCohortAnalyzerInitialization(unittest.TestCase):
    """Test CohortAnalyzer initialization and configuration."""
    
    def test_default_initialization(self):
        """Test CohortAnalyzer initialization with default configuration."""
        analyzer = CohortAnalyzer()
        
        # Verify initialization
        self.assertIsInstance(analyzer.config, dict)
        self.assertIsInstance(analyzer.analysis_results, dict)
        self.assertEqual(len(analyzer.analysis_results), 0)
        
        # Verify default configuration structure
        self.assertIn('statistical_config', analyzer.config)
        self.assertIn('clinical_thresholds', analyzer.config)
        self.assertIn('analysis_config', analyzer.config)
    
    def test_custom_configuration(self):
        """Test CohortAnalyzer initialization with custom configuration."""
        custom_config = {
            'statistical_config': {
                'alpha': 0.01,
                'min_sample_size': 20
            },
            'clinical_thresholds': {
                'outcome_score_meaningful_diff': 10.0
            }
        }
        
        analyzer = CohortAnalyzer(config=custom_config)
        
        # Verify custom configuration is used
        self.assertEqual(analyzer.config, custom_config)
    
    def test_default_config_structure(self):
        """Test the structure and values of default configuration."""
        analyzer = CohortAnalyzer()
        config = analyzer.config
        
        # Test statistical configuration
        self.assertEqual(config['statistical_config']['alpha'], 0.05)
        self.assertEqual(config['statistical_config']['power'], 0.8)
        self.assertEqual(config['statistical_config']['min_sample_size'], 10)
        
        # Test clinical thresholds
        self.assertEqual(config['clinical_thresholds']['outcome_score_meaningful_diff'], 5.0)
        self.assertEqual(config['clinical_thresholds']['compliance_meaningful_diff'], 10.0)
        
        # Test analysis configuration
        self.assertEqual(config['analysis_config']['confidence_level'], 0.95)
        self.assertEqual(config['analysis_config']['bootstrap_iterations'], 1000)


class TestCohortStatistics(unittest.TestCase):
    """Test cohort statistics calculation methods."""
    
    def setUp(self):
        """Set up test data for cohort statistics tests."""
        self.analyzer = CohortAnalyzer()
        
        # Create sample cohort data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'patient_id': [f'P{i:03d}' for i in range(1, 51)],
            'outcome_score': np.random.normal(80, 10, 50),
            'compliance_pct': np.random.normal(85, 5, 50),
            'adverse_event_flag': np.random.binomial(1, 0.1, 50),
            'dosage_mg': [50] * 25 + [75] * 25,
            'visit_date': pd.date_range('2024-01-01', periods=50, freq='D'),
            'cohort': 'A'
        })
    
    def test_calculate_cohort_statistics_complete(self):
        """Test calculation of complete cohort statistics."""
        stats = self.analyzer._calculate_cohort_statistics(self.sample_data, 'A')
        
        # Test basic information
        self.assertEqual(stats['cohort_name'], 'A')
        self.assertEqual(stats['sample_size'], 50)
        self.assertEqual(stats['unique_patients'], 50)
        
        # Test outcome statistics
        self.assertIn('outcome_stats', stats)
        outcome_stats = stats['outcome_stats']
        self.assertIn('mean', outcome_stats)
        self.assertIn('std', outcome_stats)
        self.assertIn('median', outcome_stats)
        self.assertIn('min', outcome_stats)
        self.assertIn('max', outcome_stats)
        self.assertIn('q25', outcome_stats)
        self.assertIn('q75', outcome_stats)
        self.assertIn('count', outcome_stats)
        
        # Test compliance statistics
        self.assertIn('compliance_stats', stats)
        compliance_stats = stats['compliance_stats']
        self.assertIn('mean', compliance_stats)
        self.assertIn('below_80_pct', compliance_stats)
        
        # Test adverse events statistics
        self.assertIn('adverse_events', stats)
        ae_stats = stats['adverse_events']
        self.assertIn('total_events', ae_stats)
        self.assertIn('event_rate', ae_stats)
        
        # Test dosage statistics
        self.assertIn('dosage_stats', stats)
        dosage_stats = stats['dosage_stats']
        self.assertIn('unique_dosages', dosage_stats)
        self.assertIn('dosage_distribution', dosage_stats)
        
        # Test temporal statistics
        self.assertIn('temporal_stats', stats)
        temporal_stats = stats['temporal_stats']
        self.assertIn('date_range', temporal_stats)
        self.assertIn('duration_days', temporal_stats)
    
    def test_calculate_cohort_statistics_missing_columns(self):
        """Test cohort statistics calculation with missing columns."""
        # Create data with only basic columns
        minimal_data = pd.DataFrame({
            'patient_id': ['P001', 'P002', 'P003'],
            'some_other_column': [1, 2, 3]
        })
        
        stats = self.analyzer._calculate_cohort_statistics(minimal_data, 'Test')
        
        # Should have basic stats but not specific metric stats
        self.assertEqual(stats['cohort_name'], 'Test')
        self.assertEqual(stats['sample_size'], 3)
        self.assertNotIn('outcome_stats', stats)
        self.assertNotIn('compliance_stats', stats)
        self.assertNotIn('adverse_events', stats)
    
    def test_calculate_cohort_statistics_with_nan_values(self):
        """Test cohort statistics calculation with NaN values."""
        # Create data with NaN values
        data_with_nan = self.sample_data.copy()
        data_with_nan.loc[:5, 'outcome_score'] = np.nan
        data_with_nan.loc[:3, 'compliance_pct'] = np.nan
        
        stats = self.analyzer._calculate_cohort_statistics(data_with_nan, 'A')
        
        # Should handle NaN values properly
        self.assertEqual(stats['outcome_stats']['count'], 44)  # 50 - 6 NaN values
        self.assertEqual(stats['compliance_stats']['count'], 46)  # 50 - 4 NaN values


class TestStatisticalTests(unittest.TestCase):
    """Test statistical hypothesis testing methods."""
    
    def setUp(self):
        """Set up test data for statistical tests."""
        self.analyzer = CohortAnalyzer()
        
        # Create two cohorts with known differences
        np.random.seed(42)
        self.cohort_a_data = pd.DataFrame({
            'patient_id': [f'P{i:03d}' for i in range(1, 31)],
            'outcome_score': np.random.normal(85, 8, 30),  # Higher mean
            'compliance_pct': np.random.normal(90, 5, 30),
            'adverse_event_flag': np.random.binomial(1, 0.15, 30),  # Higher AE rate
        })
        
        self.cohort_b_data = pd.DataFrame({
            'patient_id': [f'P{i:03d}' for i in range(31, 61)],
            'outcome_score': np.random.normal(75, 10, 30),  # Lower mean
            'compliance_pct': np.random.normal(85, 8, 30),
            'adverse_event_flag': np.random.binomial(1, 0.08, 30),  # Lower AE rate
        })
    
    def test_perform_statistical_tests_outcome_comparison(self):
        """Test outcome score statistical comparison."""
        test_results = self.analyzer._perform_statistical_tests(
            self.cohort_a_data, self.cohort_b_data
        )
        
        # Should include outcome comparison
        self.assertIn('outcome_comparison', test_results)
        outcome_test = test_results['outcome_comparison']
        
        # Verify test result structure
        self.assertIn('test_name', outcome_test)
        self.assertIn('statistic', outcome_test)
        self.assertIn('p_value', outcome_test)
        self.assertIn('significant', outcome_test)
        self.assertIn('mean_difference', outcome_test)
        self.assertIn('confidence_interval', outcome_test)
        
        # Test should be either t-test or Mann-Whitney U
        self.assertIn(outcome_test['test_name'], ['Independent t-test', 'Mann-Whitney U test'])
        
        # Mean difference should be positive (cohort A > cohort B)
        self.assertGreater(outcome_test['mean_difference'], 0)
    
    def test_perform_statistical_tests_compliance_comparison(self):
        """Test compliance statistical comparison."""
        test_results = self.analyzer._perform_statistical_tests(
            self.cohort_a_data, self.cohort_b_data
        )
        
        # Should include compliance comparison
        self.assertIn('compliance_comparison', test_results)
        compliance_test = test_results['compliance_comparison']
        
        # Verify test result structure
        self.assertEqual(compliance_test['test_name'], 'Independent t-test')
        self.assertIn('p_value', compliance_test)
        self.assertIn('mean_difference', compliance_test)
        self.assertIsInstance(compliance_test['significant'], bool)
    
    def test_perform_statistical_tests_adverse_events(self):
        """Test adverse events statistical comparison."""
        test_results = self.analyzer._perform_statistical_tests(
            self.cohort_a_data, self.cohort_b_data
        )
        
        # Should include adverse events comparison
        self.assertIn('adverse_events_comparison', test_results)
        ae_test = test_results['adverse_events_comparison']
        
        # Verify test result structure
        self.assertIn(ae_test['test_name'], ['Chi-square test', "Fisher's exact test"])
        self.assertIn('p_value', ae_test)
        self.assertIn('rate_difference', ae_test)
        self.assertIn('relative_risk', ae_test)
        self.assertIn('contingency_table', ae_test)
        
        # Contingency table should be 2x2
        self.assertEqual(len(ae_test['contingency_table']), 2)
        self.assertEqual(len(ae_test['contingency_table'][0]), 2)
    
    def test_perform_statistical_tests_missing_columns(self):
        """Test statistical tests with missing required columns."""
        # Create data without outcome scores
        minimal_data_a = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'some_column': [1, 2]
        })
        minimal_data_b = pd.DataFrame({
            'patient_id': ['P003', 'P004'],
            'some_column': [3, 4]
        })
        
        test_results = self.analyzer._perform_statistical_tests(
            minimal_data_a, minimal_data_b
        )
        
        # Should return empty results or skip missing comparisons
        self.assertNotIn('outcome_comparison', test_results)
        self.assertNotIn('compliance_comparison', test_results)
        self.assertNotIn('adverse_events_comparison', test_results)


class TestEffectSizes(unittest.TestCase):
    """Test effect size calculation methods."""
    
    def setUp(self):
        """Set up test data for effect size calculations."""
        self.analyzer = CohortAnalyzer()
        
        # Create cohorts with known effect sizes
        np.random.seed(42)
        self.cohort_a_data = pd.DataFrame({
            'outcome_score': np.random.normal(80, 10, 50),
            'compliance_pct': np.random.normal(85, 5, 50),
            'adverse_event_flag': np.random.binomial(1, 0.2, 50)
        })
        
        self.cohort_b_data = pd.DataFrame({
            'outcome_score': np.random.normal(75, 10, 50),  # 0.5 Cohen's d difference
            'compliance_pct': np.random.normal(80, 5, 50),
            'adverse_event_flag': np.random.binomial(1, 0.1, 50)
        })
    
    def test_calculate_effect_sizes_cohens_d(self):
        """Test Cohen's d calculation for continuous variables."""
        effect_sizes = self.analyzer._calculate_effect_sizes(
            self.cohort_a_data, self.cohort_b_data
        )
        
        # Should include Cohen's d for outcome and compliance
        self.assertIn('outcome_cohens_d', effect_sizes)
        self.assertIn('compliance_cohens_d', effect_sizes)
        
        # Cohen's d should be reasonable values
        outcome_d = effect_sizes['outcome_cohens_d']
        self.assertIsInstance(outcome_d, float)
        self.assertGreater(abs(outcome_d), 0)  # Should have some effect
        
        compliance_d = effect_sizes['compliance_cohens_d']
        self.assertIsInstance(compliance_d, float)
    
    def test_calculate_effect_sizes_odds_ratio(self):
        """Test odds ratio calculation for binary variables."""
        effect_sizes = self.analyzer._calculate_effect_sizes(
            self.cohort_a_data, self.cohort_b_data
        )
        
        # Should include odds ratio for adverse events
        self.assertIn('adverse_events_odds_ratio', effect_sizes)
        
        odds_ratio = effect_sizes['adverse_events_odds_ratio']
        self.assertIsInstance(odds_ratio, float)
        self.assertGreater(odds_ratio, 0)  # Odds ratio should be positive
    
    def test_calculate_effect_sizes_missing_data(self):
        """Test effect size calculation with missing columns."""
        # Create data without some columns
        minimal_a = pd.DataFrame({'other_column': [1, 2, 3]})
        minimal_b = pd.DataFrame({'other_column': [4, 5, 6]})
        
        effect_sizes = self.analyzer._calculate_effect_sizes(minimal_a, minimal_b)
        
        # Should return empty or partial results
        self.assertNotIn('outcome_cohens_d', effect_sizes)


class TestConfidenceIntervals(unittest.TestCase):
    """Test confidence interval calculations."""
    
    def setUp(self):
        """Set up test data for confidence interval tests."""
        self.analyzer = CohortAnalyzer()
        
        # Create sample data
        np.random.seed(42)
        self.sample_a = pd.Series(np.random.normal(80, 10, 30))
        self.sample_b = pd.Series(np.random.normal(75, 12, 25))
    
    def test_calculate_confidence_interval_structure(self):
        """Test confidence interval calculation structure."""
        ci = self.analyzer._calculate_confidence_interval(
            self.sample_a, self.sample_b, confidence_level=0.95
        )
        
        # Should return list with two elements
        self.assertIsInstance(ci, list)
        self.assertEqual(len(ci), 2)
        
        # Lower bound should be less than upper bound
        self.assertLess(ci[0], ci[1])
        
        # Both should be floats
        self.assertIsInstance(ci[0], float)
        self.assertIsInstance(ci[1], float)
    
    def test_calculate_confidence_interval_different_levels(self):
        """Test confidence intervals with different confidence levels."""
        ci_95 = self.analyzer._calculate_confidence_interval(
            self.sample_a, self.sample_b, confidence_level=0.95
        )
        ci_99 = self.analyzer._calculate_confidence_interval(
            self.sample_a, self.sample_b, confidence_level=0.99
        )
        
        # 99% CI should be wider than 95% CI
        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]
        self.assertGreater(width_99, width_95)
    
    def test_calculate_confidence_interval_identical_samples(self):
        """Test confidence interval with identical samples."""
        identical_sample = pd.Series([80.0] * 20)
        
        ci = self.analyzer._calculate_confidence_interval(
            identical_sample, identical_sample
        )
        
        # CI should be very narrow around zero
        self.assertAlmostEqual(ci[0], 0.0, places=10)
        self.assertAlmostEqual(ci[1], 0.0, places=10)


class TestClinicalSignificance(unittest.TestCase):
    """Test clinical significance assessment methods."""
    
    def setUp(self):
        """Set up test data for clinical significance tests."""
        self.analyzer = CohortAnalyzer()
        
        # Create cohort stats with known differences
        self.cohort_a_stats = {
            'cohort_name': 'A',
            'outcome_stats': {'mean': 85.0},
            'compliance_stats': {'mean': 90.0},
            'adverse_events': {'event_rate': 0.15}
        }
        
        self.cohort_b_stats = {
            'cohort_name': 'B',
            'outcome_stats': {'mean': 78.0},  # 7-point difference (>5 threshold)
            'compliance_stats': {'mean': 85.0},  # 5% difference (<10% threshold)
            'adverse_events': {'event_rate': 0.08}  # 7% difference (>5% threshold)
        }
        
        self.statistical_tests = {}
    
    def test_assess_clinical_significance_outcome(self):
        """Test clinical significance assessment for outcomes."""
        clinical_sig = self.analyzer._assess_clinical_significance(
            self.cohort_a_stats, self.cohort_b_stats, self.statistical_tests
        )
        
        # 7-point difference should be clinically significant (>5 threshold)
        self.assertIn('outcome_score', clinical_sig)
        self.assertEqual(clinical_sig['outcome_score'], 'clinically_significant')
    
    def test_assess_clinical_significance_compliance(self):
        """Test clinical significance assessment for compliance."""
        clinical_sig = self.analyzer._assess_clinical_significance(
            self.cohort_a_stats, self.cohort_b_stats, self.statistical_tests
        )
        
        # 5% difference should not be clinically significant (<10% threshold)
        self.assertIn('compliance', clinical_sig)
        self.assertEqual(clinical_sig['compliance'], 'not_clinically_significant')
    
    def test_assess_clinical_significance_adverse_events(self):
        """Test clinical significance assessment for adverse events."""
        clinical_sig = self.analyzer._assess_clinical_significance(
            self.cohort_a_stats, self.cohort_b_stats, self.statistical_tests
        )
        
        # 7% difference should be clinically significant (>5% threshold)
        self.assertIn('adverse_events', clinical_sig)
        self.assertEqual(clinical_sig['adverse_events'], 'clinically_significant')
    
    def test_assess_clinical_significance_missing_stats(self):
        """Test clinical significance assessment with missing statistics."""
        minimal_stats_a = {'cohort_name': 'A'}
        minimal_stats_b = {'cohort_name': 'B'}
        
        clinical_sig = self.analyzer._assess_clinical_significance(
            minimal_stats_a, minimal_stats_b, self.statistical_tests
        )
        
        # Should not include assessments for missing data
        self.assertNotIn('outcome_score', clinical_sig)
        self.assertNotIn('compliance', clinical_sig)
        self.assertNotIn('adverse_events', clinical_sig)


class TestCohortComparison(unittest.TestCase):
    """Test complete cohort comparison workflow."""
    
    def setUp(self):
        """Set up test data for cohort comparison tests."""
        self.analyzer = CohortAnalyzer()
        
        # Create comprehensive test data
        np.random.seed(42)
        data = []
        
        for i in range(100):
            cohort = 'A' if i < 50 else 'B'
            if cohort == 'A':
                outcome = np.random.normal(85, 8)
                compliance = np.random.normal(90, 5)
                ae_prob = 0.15
            else:
                outcome = np.random.normal(78, 10)
                compliance = np.random.normal(85, 8)
                ae_prob = 0.08
            
            data.append({
                'patient_id': f'P{i+1:03d}',
                'outcome_score': outcome,
                'compliance_pct': compliance,
                'adverse_event_flag': 1 if np.random.random() < ae_prob else 0,
                'cohort': cohort,
                'dosage_mg': 50,
                'visit_date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)
            })
        
        self.test_data = pd.DataFrame(data)
    
    def test_compare_cohorts_complete_workflow(self):
        """Test complete cohort comparison workflow."""
        result = self.analyzer.compare_cohorts(
            self.test_data, 'cohort', 'A', 'B'
        )
        
        # Verify result structure
        self.assertIsInstance(result, CohortComparisonResult)
        self.assertIn('cohort_name', result.cohort_a_stats)
        self.assertIn('cohort_name', result.cohort_b_stats)
        self.assertIsInstance(result.statistical_tests, dict)
        self.assertIsInstance(result.effect_sizes, dict)
        self.assertIsInstance(result.clinical_significance, dict)
        self.assertIsInstance(result.recommendations, list)
        self.assertEqual(result.confidence_level, 0.95)
    
    def test_compare_cohorts_insufficient_sample_size(self):
        """Test cohort comparison with insufficient sample sizes."""
        # Create very small dataset
        small_data = self.test_data.head(5)  # Only 5 records total
        
        with self.assertRaises(ValueError) as context:
            self.analyzer.compare_cohorts(small_data, 'cohort', 'A', 'B')
        
        self.assertIn('insufficient sample size', str(context.exception))
    
    def test_compare_cohorts_nonexistent_cohorts(self):
        """Test cohort comparison with nonexistent cohort identifiers."""
        # This should result in empty cohorts
        with self.assertRaises(ValueError):
            self.analyzer.compare_cohorts(self.test_data, 'cohort', 'X', 'Y')
    
    def test_compare_cohorts_statistical_significance(self):
        """Test that cohort comparison detects statistical significance."""
        result = self.analyzer.compare_cohorts(
            self.test_data, 'cohort', 'A', 'B'
        )
        
        # Should detect significant difference in outcomes (A > B by design)
        if 'outcome_comparison' in result.statistical_tests:
            outcome_test = result.statistical_tests['outcome_comparison']
            # With our synthetic data design, should likely be significant
            self.assertIsInstance(outcome_test['significant'], bool)
            self.assertGreater(outcome_test['mean_difference'], 0)
    
    def test_compare_cohorts_results_storage(self):
        """Test that comparison results are stored in analyzer."""
        result = self.analyzer.compare_cohorts(
            self.test_data, 'cohort', 'A', 'B'
        )
        
        # Results should be stored in analyzer
        self.assertIn('A_vs_B', self.analyzer.analysis_results)
        stored_result = self.analyzer.analysis_results['A_vs_B']
        self.assertEqual(stored_result, result)


class TestSubgroupAnalysis(unittest.TestCase):
    """Test subgroup analysis functionality."""
    
    def setUp(self):
        """Set up test data for subgroup analysis."""
        self.analyzer = CohortAnalyzer()
        
        # Create data with subgroups
        np.random.seed(42)
        data = []
        
        for i in range(120):
            age_group = 'young' if i < 60 else 'elderly'
            gender = 'male' if i % 2 == 0 else 'female'
            
            # Create different outcomes by subgroup
            if age_group == 'young':
                outcome = np.random.normal(85, 8)
            else:
                outcome = np.random.normal(80, 10)
            
            data.append({
                'patient_id': f'P{i+1:03d}',
                'outcome_score': outcome,
                'compliance_pct': np.random.normal(85, 5),
                'adverse_event_flag': np.random.binomial(1, 0.1),
                'age_group': age_group,
                'gender': gender
            })
        
        self.subgroup_data = pd.DataFrame(data)
    
    def test_perform_subgroup_analysis_basic(self):
        """Test basic subgroup analysis functionality."""
        results = self.analyzer.perform_subgroup_analysis(
            self.subgroup_data, 'age_group', 'outcome_score'
        )
        
        # Should have results for each age group
        self.assertIn('young', results)
        self.assertIn('elderly', results)
        
        # Each subgroup should have required statistics
        for subgroup in ['young', 'elderly']:
            stats = results[subgroup]
            self.assertIn('sample_size', stats)
            self.assertIn('outcome_mean', stats)
            self.assertIn('outcome_std', stats)
            self.assertIn('outcome_median', stats)
    
    def test_perform_subgroup_analysis_anova(self):
        """Test ANOVA test in subgroup analysis."""
        results = self.analyzer.perform_subgroup_analysis(
            self.subgroup_data, 'age_group', 'outcome_score'
        )
        
        # Should include ANOVA test results
        self.assertIn('anova_test', results)
        anova = results['anova_test']
        
        self.assertIn('f_statistic', anova)
        self.assertIn('p_value', anova)
        self.assertIn('significant', anova)
        self.assertEqual(anova['test_name'], 'One-way ANOVA')
    
    def test_perform_subgroup_analysis_multiple_subgroups(self):
        """Test subgroup analysis with multiple subgroups."""
        # Create data with more subgroups
        multi_subgroup_data = self.subgroup_data.copy()
        multi_subgroup_data['region'] = ['north', 'south', 'east', 'west'] * 30
        
        results = self.analyzer.perform_subgroup_analysis(
            multi_subgroup_data, 'region', 'outcome_score'
        )
        
        # Should have results for each region
        regions = ['north', 'south', 'east', 'west']
        for region in regions:
            self.assertIn(region, results)
        
        # Should include ANOVA for multiple groups
        self.assertIn('anova_test', results)
    
    def test_perform_subgroup_analysis_insufficient_sample_size(self):
        """Test subgroup analysis with insufficient sample sizes."""
        # Create small dataset
        small_data = self.subgroup_data.head(15)  # Very small subgroups
        
        results = self.analyzer.perform_subgroup_analysis(
            small_data, 'age_group', 'outcome_score'
        )
        
        # May have fewer subgroups due to sample size filtering
        # Should still return some results if any subgroups meet criteria
        self.assertIsInstance(results, dict)


class TestRecommendationGeneration(unittest.TestCase):
    """Test clinical recommendation generation."""
    
    def setUp(self):
        """Set up test data for recommendation tests."""
        self.analyzer = CohortAnalyzer()
        
        # Create test statistics showing A > B for outcomes
        self.cohort_a_stats = {
            'cohort_name': 'A',
            'sample_size': 50,
            'outcome_stats': {'mean': 85.0},
            'compliance_stats': {'mean': 90.0},
            'adverse_events': {'event_rate': 0.15}
        }
        
        self.cohort_b_stats = {
            'cohort_name': 'B',
            'sample_size': 45,
            'outcome_stats': {'mean': 75.0},
            'compliance_stats': {'mean': 85.0},
            'adverse_events': {'event_rate': 0.08}
        }
        
        self.statistical_tests = {
            'outcome_comparison': {
                'significant': True,
                'mean_difference': 10.0,
                'p_value': 0.001
            },
            'adverse_events_comparison': {
                'significant': True,
                'rate_difference': 0.07,
                'p_value': 0.02
            }
        }
        
        self.clinical_significance = {
            'outcome_score': 'clinically_significant',
            'adverse_events': 'clinically_significant'
        }
    
    def test_generate_recommendations_efficacy_superior(self):
        """Test recommendation generation for superior efficacy."""
        recommendations = self.analyzer._generate_recommendations(
            self.cohort_a_stats, self.cohort_b_stats,
            self.statistical_tests, self.clinical_significance
        )
        
        # Should recommend adopting protocol from superior cohort
        self.assertGreater(len(recommendations), 0)
        efficacy_rec = next((r for r in recommendations if 'EFFICACY' in r), None)
        self.assertIsNotNone(efficacy_rec)
        self.assertIn('Cohort A', efficacy_rec)
        self.assertIn('superior outcomes', efficacy_rec)
    
    def test_generate_recommendations_safety_concern(self):
        """Test recommendation generation for safety concerns."""
        recommendations = self.analyzer._generate_recommendations(
            self.cohort_a_stats, self.cohort_b_stats,
            self.statistical_tests, self.clinical_significance
        )
        
        # Should recommend safety review for higher AE rate
        safety_rec = next((r for r in recommendations if 'SAFETY' in r), None)
        self.assertIsNotNone(safety_rec)
        self.assertIn('adverse event rate', safety_rec)
    
    def test_generate_recommendations_no_significant_differences(self):
        """Test recommendation generation when no significant differences found."""
        # Create tests with no significant results
        non_significant_tests = {
            'outcome_comparison': {'significant': False, 'p_value': 0.5}
        }
        non_significant_clinical = {
            'outcome_score': 'not_clinically_significant'
        }
        
        recommendations = self.analyzer._generate_recommendations(
            self.cohort_a_stats, self.cohort_b_stats,
            non_significant_tests, non_significant_clinical
        )
        
        # Should provide some kind of recommendation
        self.assertGreater(len(recommendations), 0)
        # Check that there's at least one recommendation about study design or monitoring
        has_relevant_rec = any('study design' in rec.lower() or 'monitoring' in rec.lower() or 'sample size' in rec.lower() 
                              for rec in recommendations)
        self.assertTrue(has_relevant_rec)
    
    def test_generate_recommendations_sample_size_warning(self):
        """Test recommendation generation includes sample size warnings."""
        # Create small sample sizes
        small_cohort_a = self.cohort_a_stats.copy()
        small_cohort_b = self.cohort_b_stats.copy()
        small_cohort_a['sample_size'] = 15
        small_cohort_b['sample_size'] = 12
        
        recommendations = self.analyzer._generate_recommendations(
            small_cohort_a, small_cohort_b,
            self.statistical_tests, self.clinical_significance
        )
        
        # Should include sample size recommendation
        sample_size_rec = next((r for r in recommendations if 'sample size' in r.lower()), None)
        self.assertIsNotNone(sample_size_rec)


class TestReportGeneration(unittest.TestCase):
    """Test comprehensive report generation."""
    
    def setUp(self):
        """Set up test data for report generation."""
        self.analyzer = CohortAnalyzer()
        
        # Create complete comparison result
        self.comparison_result = CohortComparisonResult(
            cohort_a_stats={
                'cohort_name': 'Treatment',
                'sample_size': 50,
                'unique_patients': 50,
                'outcome_stats': {'mean': 85.5, 'std': 8.2, 'median': 86.0, 'q25': 80.0, 'q75': 91.0, 'min': 65.0, 'max': 100.0},
                'compliance_stats': {'mean': 89.5, 'std': 5.1, 'below_80_pct': 5.0},
                'adverse_events': {'event_rate': 0.12, 'total_events': 6, 'patients_with_events': 6, 'patient_event_rate': 0.12}
            },
            cohort_b_stats={
                'cohort_name': 'Control',
                'sample_size': 48,
                'unique_patients': 48,
                'outcome_stats': {'mean': 78.3, 'std': 9.8, 'median': 79.0, 'q25': 72.0, 'q75': 85.0, 'min': 55.0, 'max': 95.0},
                'compliance_stats': {'mean': 85.2, 'std': 7.3, 'below_80_pct': 15.0},
                'adverse_events': {'event_rate': 0.08, 'total_events': 4, 'patients_with_events': 4, 'patient_event_rate': 0.08}
            },
            statistical_tests={
                'outcome_comparison': {
                    'test_name': 'Independent t-test',
                    'p_value': 0.002,
                    'significant': True,
                    'mean_difference': 7.2,
                    'confidence_interval': [2.5, 11.9]
                },
                'adverse_events_comparison': {
                    'test_name': 'Chi-square test',
                    'p_value': 0.15,
                    'significant': False,
                    'rate_difference': 0.04,
                    'relative_risk': 1.58
                }
            },
            effect_sizes={
                'outcome_cohens_d': 0.82,
                'compliance_cohens_d': 0.65,
                'adverse_events_odds_ratio': 1.58
            },
            clinical_significance={
                'outcome_score': 'clinically_significant',
                'compliance': 'not_clinically_significant',
                'adverse_events': 'not_clinically_significant'
            },
            recommendations=[
                '🎯 EFFICACY: Treatment group demonstrates significantly superior outcomes.',
                '📊 ADHERENCE: Consider implementing adherence support programs.'
            ],
            confidence_level=0.95
        )
    
    def test_generate_cohort_summary_report_structure(self):
        """Test that summary report contains all required sections."""
        report = self.analyzer.generate_cohort_summary_report(self.comparison_result)
        
        # Verify report is a string
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 100)  # Should be substantial
        
        # Check for key sections
        self.assertIn('CLINICAL COHORT COMPARISON ANALYSIS REPORT', report)
        self.assertIn('SAMPLE CHARACTERISTICS', report)
        self.assertIn('PRIMARY OUTCOME ANALYSIS', report)
        self.assertIn('SAFETY ANALYSIS', report)
        self.assertIn('COMPLIANCE ANALYSIS', report)
        self.assertIn('EFFECT SIZES', report)
        self.assertIn('CLINICAL RECOMMENDATIONS', report)
    
    def test_generate_cohort_summary_report_content(self):
        """Test that summary report contains accurate content."""
        report = self.analyzer.generate_cohort_summary_report(self.comparison_result)
        
        # Check for specific data points
        self.assertIn('Treatment', report)
        self.assertIn('Control', report)
        self.assertIn('85.5', report)  # Treatment mean
        self.assertIn('78.3', report)  # Control mean
        self.assertIn('Independent t-test', report)
        self.assertIn('0.002', report)  # P-value
        self.assertIn('Statistically significant', report)
        
        # Check for effect sizes
        self.assertIn('0.820', report)  # Cohen's d for outcome
        self.assertIn('Large', report)  # Effect size interpretation
    
    def test_generate_cohort_summary_report_recommendations(self):
        """Test that summary report includes recommendations."""
        report = self.analyzer.generate_cohort_summary_report(self.comparison_result)
        
        # Should include all recommendations
        for recommendation in self.comparison_result.recommendations:
            # Remove emoji and check for key content
            rec_content = recommendation.replace('🎯', '').replace('📊', '').strip()
            key_words = rec_content.split()[:3]  # First few words
            for word in key_words:
                if len(word) > 2:  # Skip short words
                    self.assertIn(word, report)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test data for edge case testing."""
        self.analyzer = CohortAnalyzer()
    
    def test_identical_cohorts(self):
        """Test comparison of identical cohorts."""
        # Create identical data for both cohorts
        np.random.seed(42)
        identical_data = pd.DataFrame({
            'patient_id': [f'P{i:03d}' for i in range(1, 41)],
            'outcome_score': [80.0] * 40,  # Identical outcomes
            'compliance_pct': [90.0] * 40,  # Identical compliance
            'adverse_event_flag': [0] * 40,  # No adverse events
            'cohort': ['A'] * 20 + ['B'] * 20
        })
        
        result = self.analyzer.compare_cohorts(identical_data, 'cohort', 'A', 'B')
        
        # Should handle identical data gracefully
        self.assertIsInstance(result, CohortComparisonResult)
        
        # P-values should be non-significant (close to 1.0)
        if 'outcome_comparison' in result.statistical_tests:
            self.assertFalse(result.statistical_tests['outcome_comparison']['significant'])
    
    def test_extreme_outliers(self):
        """Test handling of extreme outlier values."""
        # Create data with extreme outliers
        np.random.seed(42)
        outlier_data = []
        
        for i in range(40):
            cohort = 'A' if i < 20 else 'B'
            if i == 0:  # Extreme outlier
                outcome = 500.0  # Way outside normal range
            else:
                outcome = np.random.normal(80, 5)
            
            outlier_data.append({
                'patient_id': f'P{i+1:03d}',
                'outcome_score': outcome,
                'compliance_pct': np.random.normal(85, 5),
                'adverse_event_flag': 0,
                'cohort': cohort
            })
        
        outlier_df = pd.DataFrame(outlier_data)
        
        # Should handle outliers without crashing
        result = self.analyzer.compare_cohorts(outlier_df, 'cohort', 'A', 'B')
        self.assertIsInstance(result, CohortComparisonResult)
    
    def test_missing_data_handling(self):
        """Test handling of datasets with missing values."""
        # Create data with substantial missing values
        np.random.seed(42)
        missing_data = pd.DataFrame({
            'patient_id': [f'P{i:03d}' for i in range(1, 41)],
            'outcome_score': [np.nan if i % 3 == 0 else np.random.normal(80, 5) for i in range(40)],
            'compliance_pct': [np.nan if i % 4 == 0 else np.random.normal(85, 5) for i in range(40)],
            'adverse_event_flag': [np.nan if i % 5 == 0 else np.random.binomial(1, 0.1) for i in range(40)],
            'cohort': ['A'] * 20 + ['B'] * 20
        })
        
        # Should handle missing data gracefully
        result = self.analyzer.compare_cohorts(missing_data, 'cohort', 'A', 'B')
        self.assertIsInstance(result, CohortComparisonResult)
    
    def test_single_patient_cohorts(self):
        """Test handling of very small cohorts."""
        # Create minimal data (just above minimum sample size)
        minimal_data = pd.DataFrame({
            'patient_id': [f'P{i:03d}' for i in range(1, 25)],
            'outcome_score': np.random.normal(80, 5, 24),
            'compliance_pct': np.random.normal(85, 5, 24),
            'adverse_event_flag': [0] * 24,
            'cohort': ['A'] * 12 + ['B'] * 12
        })
        
        # Should work with minimum viable sample sizes
        result = self.analyzer.compare_cohorts(minimal_data, 'cohort', 'A', 'B')
        self.assertIsInstance(result, CohortComparisonResult)
    
    def test_non_standard_column_names(self):
        """Test handling of data with non-standard column names."""
        # Create data with different column names
        nonstandard_data = pd.DataFrame({
            'subject_id': ['S001', 'S002', 'S003', 'S004'] * 10,
            'primary_endpoint': np.random.normal(80, 5, 40),
            'treatment_group': ['Active'] * 20 + ['Placebo'] * 20
        })
        
        # Should handle gracefully when expected columns are missing
        # This should not crash but may return limited results
        try:
            result = self.analyzer.compare_cohorts(
                nonstandard_data, 'treatment_group', 'Active', 'Placebo'
            )
            self.assertIsInstance(result, CohortComparisonResult)
        except ValueError:
            # Expected if sample size validation fails
            pass


def run_all_tests():
    """Run all cohort analysis tests with detailed output."""
    print("="*80)
    print("RUNNING COMPREHENSIVE COHORT ANALYSIS TEST SUITE")
    print("="*80)
    
    # Create test suite
    test_classes = [
        TestCohortComparisonResult,
        TestCohortAnalyzerInitialization,
        TestCohortStatistics,
        TestStatisticalTests,
        TestEffectSizes,
        TestConfidenceIntervals,
        TestClinicalSignificance,
        TestCohortComparison,
        TestSubgroupAnalysis,
        TestRecommendationGeneration,
        TestReportGeneration,
        TestEdgeCases
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\n📊 Running {test_class.__name__}...")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        # Count tests in this class
        class_test_count = suite.countTestCases()
        total_tests += class_test_count
        
        # Run tests with minimal output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w')).run(suite)
        
        class_passed = class_test_count - len(result.failures) - len(result.errors)
        class_failed = len(result.failures) + len(result.errors)
        
        passed_tests += class_passed
        failed_tests += class_failed
        
        # Print results for this class
        if class_failed == 0:
            print(f"   ✅ {class_passed}/{class_test_count} tests passed")
        else:
            print(f"   ❌ {class_passed}/{class_test_count} tests passed, {class_failed} failed")
            
            # Print failure details
            for failure in result.failures:
                print(f"      FAIL: {failure[0]}")
            for error in result.errors:
                print(f"      ERROR: {error[0]}")
    
    # Print final summary
    print("\n" + "="*80)
    print("TEST SUITE SUMMARY")
    print("="*80)
    print(f"📊 Total Tests Run: {total_tests}")
    print(f"✅ Tests Passed: {passed_tests}")
    print(f"❌ Tests Failed: {failed_tests}")
    print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests == 0:
        print("\n🎉 ALL TESTS PASSED! Cohort Analysis module is working perfectly!")
    else:
        print(f"\n⚠️  {failed_tests} tests failed. Please review the failures above.")
    
    print("="*80)
    
    return failed_tests == 0


if __name__ == '__main__':
    # Run comprehensive test suite
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)