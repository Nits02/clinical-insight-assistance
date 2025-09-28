"""
Cohort Analysis Module for Clinical Insights Assistant

This module provides statistical analysis and comparison tools for different patient cohorts
in clinical trials, including efficacy comparisons, safety analysis, and demographic analysis.

Comparative Analysis Between Cohorts
This module enables the statistical comparison of outcomes, safety profiles, and other
metrics between different patient cohorts (e.g., treatment groups, demographic subgroups).
It's essential for understanding treatment efficacy and safety differences.

Purpose:
• Perform statistical tests to compare two or more cohorts.
• Calculate effect sizes and confidence intervals.
• Generate summary statistics for each cohort.
• Provide recommendations based on comparison results.

Step-by-Step Implementation:
1. Define the CohortAnalyzer Class:
   The class will manage the analysis configuration and methods.
2. Implement _get_default_config:
   Define default parameters for statistical tests.
3. Implement _get_cohort_stats:
   A helper method to extract statistics for a given cohort.
4. Implement compare_cohorts:
   The main method for performing cohort comparison.
5. Add main function for testing:
   This module provides robust statistical comparison capabilities, which
   are fundamental for evaluating the effectiveness and safety of different treatment
   approaches in clinical trials. The use of scipy.stats and statsmodels ensures reliable
   statistical inference.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

# Configure logging for monitoring cohort analysis operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CohortComparisonResult:
    """
    Data class to hold comprehensive cohort comparison results.
    
    This structured approach ensures consistent reporting of cohort analysis results
    across all comparison methods, providing standardized information for
    clinical decision-making and regulatory reporting.
    
    Attributes:
        cohort_a_stats (Dict): Comprehensive statistics for cohort A
        cohort_b_stats (Dict): Comprehensive statistics for cohort B
        statistical_tests (Dict): Results of statistical hypothesis tests
        effect_sizes (Dict): Effect size calculations (Cohen's d, odds ratios)
        clinical_significance (Dict): Clinical significance assessments
        recommendations (List): Clinical recommendations based on analysis
        confidence_level (float): Confidence level used in analysis
    """
    cohort_a_stats: Dict[str, Any]
    cohort_b_stats: Dict[str, Any]
    statistical_tests: Dict[str, Any]
    effect_sizes: Dict[str, float]
    clinical_significance: Dict[str, str]
    recommendations: List[str]
    confidence_level: float


class CohortAnalyzer:
    """
    Class for analyzing and comparing different patient cohorts in clinical trials.
    
    The CohortAnalyzer class provides comprehensive statistical analysis capabilities
    for comparing treatment groups, demographic subgroups, and other patient cohorts.
    It implements robust statistical methods including:
    - Descriptive statistics calculation
    - Hypothesis testing (t-tests, Mann-Whitney U, Chi-square)
    - Effect size calculations (Cohen's d, odds ratios)
    - Clinical significance assessment
    - Subgroup analysis capabilities
    
    This class is fundamental for evaluating the effectiveness and safety of different
    treatment approaches in clinical trials, ensuring reliable statistical inference
    through scipy.stats and proper clinical interpretation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Cohort Analyzer with statistical and clinical parameters.
        
        The initialization sets up analysis thresholds and parameters that define
        statistical significance, clinical relevance, and analysis methodology.
        Custom configuration allows for study-specific adjustments while maintaining
        robust default values based on clinical research standards.
        
        Args:
            config (Dict, optional): Configuration dictionary for analysis parameters.
                                   If not provided, uses clinically-validated default values.
        """
        # Load configuration for statistical analysis parameters
        self.config = config or self._get_default_config()
        
        # Initialize storage for analysis results across multiple comparisons
        self.analysis_results = {}
        
        logger.info("Cohort Analyzer initialized with statistical analysis configuration")
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration for cohort analysis with clinically-validated parameters.
        
        This method defines the foundational statistical and clinical parameters used
        across all analysis methods. These values are based on clinical research
        best practices, regulatory guidelines, and statistical standards for
        clinical trial analysis.
        
        The configuration is organized into logical groups:
        - Statistical config: Hypothesis testing parameters and thresholds
        - Clinical thresholds: Minimum clinically meaningful differences
        - Analysis config: General analysis parameters and quality controls
        
        Returns:
            Dict: Comprehensive configuration parameters for all analysis methods.
        """
        return {
            # Statistical test parameters - Core hypothesis testing configuration
            # These parameters control the statistical rigor of cohort comparisons
            'statistical_config': {
                'alpha': 0.05,              # Significance level (Type I error rate)
                'power': 0.8,               # Statistical power (1 - Type II error rate)
                'effect_size_thresholds': {
                    'small': 0.2,           # Small effect size threshold (Cohen's conventions)
                    'medium': 0.5,          # Medium effect size threshold
                    'large': 0.8            # Large effect size threshold
                },
                'min_sample_size': 10       # Minimum sample size for reliable analysis
            },
            
            # Clinical significance thresholds - Minimum meaningful differences
            # These define when statistical differences become clinically relevant
            'clinical_thresholds': {
                'outcome_score_meaningful_diff': 5.0,    # Minimum clinically meaningful outcome difference
                'compliance_meaningful_diff': 10.0,      # Minimum meaningful compliance difference (%)
                'adverse_event_rate_threshold': 0.05     # 5% difference threshold for AE rates
            },
            
            # Analysis parameters - General configuration for analysis methodology
            # These control the technical aspects of the statistical analysis
            'analysis_config': {
                'confidence_level': 0.95,        # Confidence level for interval estimation
                'bootstrap_iterations': 1000,    # Number of bootstrap samples for robust estimation
                'outlier_threshold': 3.0,        # Z-score threshold for outlier detection
                'missing_data_threshold': 0.2    # Maximum allowed missing data proportion
            }
        }
    
    def compare_cohorts(self, data: pd.DataFrame, cohort_column: str = 'cohort', 
                       cohort_a: str = 'A', cohort_b: str = 'B') -> CohortComparisonResult:
        """
        Perform comprehensive statistical comparison between two patient cohorts.
        
        This is the main analysis method that orchestrates a complete cohort comparison,
        including descriptive statistics, hypothesis testing, effect size calculation,
        and clinical significance assessment. The method follows a systematic approach:
        
        1. Data validation and cohort extraction
        2. Descriptive statistics calculation for both cohorts
        3. Statistical hypothesis testing (parametric and non-parametric)
        4. Effect size calculation (Cohen's d, odds ratios)
        5. Clinical significance assessment
        6. Clinical recommendation generation
        
        Args:
            data (pd.DataFrame): Clinical trial data containing patient records.
            cohort_column (str): Column name containing cohort identifiers.
            cohort_a (str): Identifier for first cohort to compare.
            cohort_b (str): Identifier for second cohort to compare.
            
        Returns:
            CohortComparisonResult: Comprehensive comparison results with statistical
                                  tests, effect sizes, and clinical recommendations.
        """
        logger.info(f"Starting comprehensive cohort comparison: {cohort_a} vs {cohort_b}")
        
        # Extract and validate cohort data
        # This ensures we have sufficient data for reliable statistical analysis
        cohort_a_data = data[data[cohort_column] == cohort_a].copy()
        cohort_b_data = data[data[cohort_column] == cohort_b].copy()
        
        # Validate sample sizes for statistical power
        min_sample_size = self.config['statistical_config']['min_sample_size']
        if len(cohort_a_data) < min_sample_size:
            raise ValueError(f"Cohort {cohort_a} has insufficient sample size: {len(cohort_a_data)} < {min_sample_size}")
        if len(cohort_b_data) < min_sample_size:
            raise ValueError(f"Cohort {cohort_b} has insufficient sample size: {len(cohort_b_data)} < {min_sample_size}")
        
        # Calculate comprehensive descriptive statistics for both cohorts
        # This provides the foundation for all subsequent statistical comparisons
        cohort_a_stats = self._calculate_cohort_statistics(cohort_a_data, cohort_a)
        cohort_b_stats = self._calculate_cohort_statistics(cohort_b_data, cohort_b)
        
        # Perform statistical hypothesis tests
        # This includes parametric and non-parametric tests as appropriate
        statistical_tests = self._perform_statistical_tests(cohort_a_data, cohort_b_data)
        
        # Calculate effect sizes for practical significance assessment
        # Effect sizes provide information about the magnitude of differences
        effect_sizes = self._calculate_effect_sizes(cohort_a_data, cohort_b_data)
        
        # Assess clinical significance of observed differences
        # This translates statistical findings into clinical relevance
        clinical_significance = self._assess_clinical_significance(
            cohort_a_stats, cohort_b_stats, statistical_tests
        )
        
        # Generate evidence-based clinical recommendations
        # These provide actionable insights for clinical decision-making
        recommendations = self._generate_recommendations(
            cohort_a_stats, cohort_b_stats, statistical_tests, clinical_significance
        )
        
        # Create comprehensive result object
        result = CohortComparisonResult(
            cohort_a_stats=cohort_a_stats,
            cohort_b_stats=cohort_b_stats,
            statistical_tests=statistical_tests,
            effect_sizes=effect_sizes,
            clinical_significance=clinical_significance,
            recommendations=recommendations,
            confidence_level=self.config['analysis_config']['confidence_level']
        )
        
        # Store results for future reference and meta-analysis
        self.analysis_results[f"{cohort_a}_vs_{cohort_b}"] = result
        
        logger.info(f"Cohort comparison completed successfully: {cohort_a} vs {cohort_b}")
        return result
    
    def _calculate_cohort_statistics(self, cohort_data: pd.DataFrame, cohort_name: str) -> Dict[str, Any]:
        """
        Calculate comprehensive descriptive statistics for a patient cohort.
        
        This method computes detailed descriptive statistics that provide a complete
        picture of the cohort's characteristics, including:
        - Sample size and patient counts
        - Outcome score distributions
        - Compliance patterns
        - Adverse event profiles
        - Dosage distributions
        - Temporal characteristics
        
        Args:
            cohort_data (pd.DataFrame): Data for the specific cohort.
            cohort_name (str): Name identifier for the cohort.
            
        Returns:
            Dict[str, Any]: Comprehensive descriptive statistics dictionary.
        """
        # Initialize statistics dictionary with basic cohort information
        stats_dict = {
            'cohort_name': cohort_name,
            'sample_size': len(cohort_data),
            'unique_patients': cohort_data['patient_id'].nunique() if 'patient_id' in cohort_data.columns else len(cohort_data)
        }
        
        # Outcome score statistics - Treatment effectiveness measures
        # These statistics are crucial for evaluating treatment efficacy
        if 'outcome_score' in cohort_data.columns:
            outcome_scores = cohort_data['outcome_score'].dropna()
            stats_dict['outcome_stats'] = {
                'mean': float(outcome_scores.mean()),
                'median': float(outcome_scores.median()),
                'std': float(outcome_scores.std()),
                'min': float(outcome_scores.min()),
                'max': float(outcome_scores.max()),
                'q25': float(outcome_scores.quantile(0.25)),
                'q75': float(outcome_scores.quantile(0.75)),
                'count': len(outcome_scores),
                'cv': float(outcome_scores.std() / outcome_scores.mean()) if outcome_scores.mean() != 0 else 0
            }
        
        # Compliance statistics - Patient adherence analysis
        # Critical for understanding treatment adherence patterns
        if 'compliance_pct' in cohort_data.columns:
            compliance = cohort_data['compliance_pct'].dropna()
            stats_dict['compliance_stats'] = {
                'mean': float(compliance.mean()),
                'median': float(compliance.median()),
                'std': float(compliance.std()),
                'min': float(compliance.min()),
                'max': float(compliance.max()),
                'below_80_pct': float((compliance < 80).mean() * 100),  # Poor compliance threshold
                'below_50_pct': float((compliance < 50).mean() * 100),  # Critical compliance threshold
                'count': len(compliance)
            }
        
        # Adverse events statistics - Safety profile analysis
        # Essential for safety monitoring and risk-benefit assessment
        if 'adverse_event_flag' in cohort_data.columns:
            ae_data = cohort_data['adverse_event_flag']
            total_events = ae_data.sum()
            total_records = len(ae_data)
            
            stats_dict['adverse_events'] = {
                'total_events': int(total_events),
                'total_records': total_records,
                'event_rate': float(total_events / total_records),
                'patients_with_events': int(cohort_data.groupby('patient_id')['adverse_event_flag'].max().sum()) if 'patient_id' in cohort_data.columns else int(total_events),
                'patient_event_rate': float(cohort_data.groupby('patient_id')['adverse_event_flag'].max().mean()) if 'patient_id' in cohort_data.columns else float(total_events / total_records)
            }
        
        # Dosage statistics - Treatment intensity analysis
        # Important for dose-response relationship assessment
        if 'dosage_mg' in cohort_data.columns:
            dosage = cohort_data['dosage_mg'].dropna()
            stats_dict['dosage_stats'] = {
                'mean': float(dosage.mean()),
                'median': float(dosage.median()),
                'std': float(dosage.std()),
                'unique_dosages': sorted(dosage.unique().tolist()),
                'dosage_distribution': dosage.value_counts().to_dict()
            }
        
        # Temporal analysis - Study timeline characteristics
        # Provides context for time-based analysis and trends
        if 'visit_date' in cohort_data.columns:
            cohort_data['visit_date'] = pd.to_datetime(cohort_data['visit_date'])
            stats_dict['temporal_stats'] = {
                'date_range': {
                    'start': cohort_data['visit_date'].min().isoformat(),
                    'end': cohort_data['visit_date'].max().isoformat()
                },
                'duration_days': (cohort_data['visit_date'].max() - cohort_data['visit_date'].min()).days,
                'visits_per_patient': float(len(cohort_data) / cohort_data['patient_id'].nunique()) if 'patient_id' in cohort_data.columns else 1
            }
        
        return stats_dict
    
    def _perform_statistical_tests(self, cohort_a_data: pd.DataFrame, 
                                 cohort_b_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive statistical hypothesis tests to compare cohorts.
        
        This method implements multiple statistical tests appropriate for different
        data types and distributions:
        - Parametric tests (t-test) for normally distributed continuous data
        - Non-parametric tests (Mann-Whitney U) for non-normal continuous data
        - Chi-square tests for categorical data (adverse events)
        
        The method automatically selects appropriate tests based on data characteristics
        and provides comprehensive test results including effect sizes and confidence intervals.
        
        Args:
            cohort_a_data (pd.DataFrame): Data for cohort A.
            cohort_b_data (pd.DataFrame): Data for cohort B.
            
        Returns:
            Dict[str, Any]: Comprehensive statistical test results.
        """
        test_results = {}
        alpha = self.config['statistical_config']['alpha']
        
        # Outcome score comparison - Primary efficacy endpoint analysis
        # Uses appropriate test based on data distribution normality
        if 'outcome_score' in cohort_a_data.columns and 'outcome_score' in cohort_b_data.columns:
            outcome_a = cohort_a_data['outcome_score'].dropna()
            outcome_b = cohort_b_data['outcome_score'].dropna()
            
            # Test for normality using Shapiro-Wilk test (sample size limited for computational efficiency)
            sample_size_a = min(5000, len(outcome_a))
            sample_size_b = min(5000, len(outcome_b))
            _, p_norm_a = stats.shapiro(outcome_a.sample(sample_size_a)) if len(outcome_a) > 3 else (0, 0.001)
            _, p_norm_b = stats.shapiro(outcome_b.sample(sample_size_b)) if len(outcome_b) > 3 else (0, 0.001)
            
            # Select appropriate test based on normality assumption
            if p_norm_a > 0.05 and p_norm_b > 0.05:
                # Use parametric t-test for normally distributed data
                t_stat, p_value = stats.ttest_ind(outcome_a, outcome_b)
                test_name = "Independent t-test"
            else:
                # Use non-parametric Mann-Whitney U test for non-normal data
                t_stat, p_value = stats.mannwhitneyu(outcome_a, outcome_b, alternative='two-sided')
                test_name = "Mann-Whitney U test"
            
            test_results['outcome_comparison'] = {
                'test_name': test_name,
                'statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < alpha),
                'mean_difference': float(outcome_a.mean() - outcome_b.mean()),
                'confidence_interval': self._calculate_confidence_interval(outcome_a, outcome_b),
                'normality_test_p_values': {'cohort_a': float(p_norm_a), 'cohort_b': float(p_norm_b)}
            }
        
        # Compliance comparison - Treatment adherence analysis
        # Generally uses parametric test as compliance percentages are often normally distributed
        if 'compliance_pct' in cohort_a_data.columns and 'compliance_pct' in cohort_b_data.columns:
            compliance_a = cohort_a_data['compliance_pct'].dropna()
            compliance_b = cohort_b_data['compliance_pct'].dropna()
            
            t_stat, p_value = stats.ttest_ind(compliance_a, compliance_b)
            test_results['compliance_comparison'] = {
                'test_name': 'Independent t-test',
                'statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < alpha),
                'mean_difference': float(compliance_a.mean() - compliance_b.mean()),
                'confidence_interval': self._calculate_confidence_interval(compliance_a, compliance_b)
            }
        
        # Adverse events comparison - Safety profile analysis using Chi-square test
        # Appropriate for comparing proportions between groups
        if 'adverse_event_flag' in cohort_a_data.columns and 'adverse_event_flag' in cohort_b_data.columns:
            ae_a = cohort_a_data['adverse_event_flag'].sum()
            ae_b = cohort_b_data['adverse_event_flag'].sum()
            n_a = len(cohort_a_data)
            n_b = len(cohort_b_data)
            
            # Create 2x2 contingency table for Chi-square test
            contingency_table = np.array([[ae_a, n_a - ae_a], [ae_b, n_b - ae_b]])
            
            # Calculate additional safety metrics
            rate_a = ae_a / n_a
            rate_b = ae_b / n_b
            relative_risk = rate_a / rate_b if rate_b > 0 else float('inf')
            
            # Choose appropriate statistical test based on data
            try:
                # Check if we can use Chi-square test (expected frequencies >= 5)
                if ae_a >= 5 and ae_b >= 5 and (n_a - ae_a) >= 5 and (n_b - ae_b) >= 5:
                    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                    test_name = 'Chi-square test'
                    statistic = float(chi2_stat)
                    expected_freq = expected.tolist()
                else:
                    # Use Fisher's exact test for small samples or zero cells
                    from scipy.stats import fisher_exact
                    odds_ratio, p_value = fisher_exact(contingency_table)
                    test_name = "Fisher's exact test"
                    statistic = float(odds_ratio)
                    expected_freq = None
                
                test_results['adverse_events_comparison'] = {
                    'test_name': test_name,
                    'statistic': statistic,
                    'p_value': float(p_value),
                    'significant': bool(p_value < alpha),
                    'rate_difference': float(rate_a - rate_b),
                    'relative_risk': float(relative_risk),
                    'contingency_table': contingency_table.tolist(),
                    'expected_frequencies': expected_freq
                }
            except (ValueError, ZeroDivisionError) as e:
                # Handle edge cases where statistical test cannot be performed
                logger.warning(f"Could not perform adverse events statistical test: {e}")
                test_results['adverse_events_comparison'] = {
                    'test_name': 'Descriptive comparison only',
                    'statistic': None,
                    'p_value': None,
                    'significant': None,
                    'rate_difference': float(rate_a - rate_b),
                    'relative_risk': float(relative_risk),
                    'contingency_table': contingency_table.tolist(),
                    'expected_frequencies': None
                }
        
        return test_results
    
    def _calculate_effect_sizes(self, cohort_a_data: pd.DataFrame, 
                              cohort_b_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate effect sizes to assess practical significance of differences.
        
        Effect sizes provide information about the magnitude of differences between
        cohorts, independent of sample size. This method calculates:
        - Cohen's d for continuous variables (outcome scores, compliance)
        - Odds ratios for binary variables (adverse events)
        
        Effect sizes are crucial for clinical interpretation as they indicate
        whether statistically significant differences are also practically meaningful.
        
        Args:
            cohort_a_data (pd.DataFrame): Data for cohort A.
            cohort_b_data (pd.DataFrame): Data for cohort B.
            
        Returns:
            Dict[str, float]: Effect size calculations.
        """
        effect_sizes = {}
        
        # Cohen's d for outcome scores - Primary efficacy effect size
        # Cohen's d represents the standardized mean difference between groups
        if 'outcome_score' in cohort_a_data.columns and 'outcome_score' in cohort_b_data.columns:
            outcome_a = cohort_a_data['outcome_score'].dropna()
            outcome_b = cohort_b_data['outcome_score'].dropna()
            
            # Calculate pooled standard deviation
            pooled_std = np.sqrt(((len(outcome_a) - 1) * outcome_a.var() + 
                                (len(outcome_b) - 1) * outcome_b.var()) / 
                               (len(outcome_a) + len(outcome_b) - 2))
            
            # Calculate Cohen's d
            if pooled_std > 0:
                cohens_d = (outcome_a.mean() - outcome_b.mean()) / pooled_std
                effect_sizes['outcome_cohens_d'] = float(cohens_d)
        
        # Cohen's d for compliance - Treatment adherence effect size
        if 'compliance_pct' in cohort_a_data.columns and 'compliance_pct' in cohort_b_data.columns:
            compliance_a = cohort_a_data['compliance_pct'].dropna()
            compliance_b = cohort_b_data['compliance_pct'].dropna()
            
            # Calculate pooled standard deviation
            pooled_std = np.sqrt(((len(compliance_a) - 1) * compliance_a.var() + 
                                (len(compliance_b) - 1) * compliance_b.var()) / 
                               (len(compliance_a) + len(compliance_b) - 2))
            
            # Calculate Cohen's d
            if pooled_std > 0:
                cohens_d = (compliance_a.mean() - compliance_b.mean()) / pooled_std
                effect_sizes['compliance_cohens_d'] = float(cohens_d)
        
        # Odds ratio for adverse events - Safety profile effect size
        # Odds ratio represents the odds of adverse events in one group vs another
        if 'adverse_event_flag' in cohort_a_data.columns and 'adverse_event_flag' in cohort_b_data.columns:
            ae_a = cohort_a_data['adverse_event_flag'].sum()
            ae_b = cohort_b_data['adverse_event_flag'].sum()
            n_a = len(cohort_a_data)
            n_b = len(cohort_b_data)
            
            # Calculate odds for each group
            odds_a = ae_a / (n_a - ae_a) if (n_a - ae_a) > 0 else float('inf')
            odds_b = ae_b / (n_b - ae_b) if (n_b - ae_b) > 0 else float('inf')
            
            # Calculate odds ratio
            odds_ratio = odds_a / odds_b if odds_b > 0 and odds_b != float('inf') else float('inf')
            effect_sizes['adverse_events_odds_ratio'] = float(odds_ratio)
        
        return effect_sizes
    
    def _calculate_confidence_interval(self, sample_a: pd.Series, sample_b: pd.Series, 
                                     confidence_level: float = 0.95) -> List[float]:
        """
        Calculate confidence interval for the difference in means using Welch's method.
        
        This method calculates confidence intervals for the difference between two
        sample means, accounting for potentially unequal variances (Welch's t-test approach).
        The confidence interval provides a range of plausible values for the true
        population difference.
        
        Args:
            sample_a (pd.Series): Sample data from cohort A.
            sample_b (pd.Series): Sample data from cohort B.
            confidence_level (float): Confidence level for interval calculation.
            
        Returns:
            List[float]: [lower_bound, upper_bound] of confidence interval.
        """
        # Calculate difference in sample means
        mean_diff = sample_a.mean() - sample_b.mean()
        
        # Handle identical samples or zero variance cases
        var_a = sample_a.var()
        var_b = sample_b.var()
        
        if var_a == 0 and var_b == 0:
            # Both samples are identical, CI around difference is exactly zero
            return [0.0, 0.0]
        
        # Calculate standard error using Welch's formula (unequal variances)
        se = np.sqrt(var_a/len(sample_a) + var_b/len(sample_b))
        
        # Handle case where standard error is zero
        if se == 0:
            return [float(mean_diff), float(mean_diff)]
        
        # Calculate degrees of freedom using Welch-Satterthwaite equation
        try:
            df = ((var_a/len(sample_a) + var_b/len(sample_b))**2) / \
                 ((var_a/len(sample_a))**2/(len(sample_a)-1) + 
                  (var_b/len(sample_b))**2/(len(sample_b)-1))
        except ZeroDivisionError:
            # If division by zero, use simple pooled estimate
            df = len(sample_a) + len(sample_b) - 2
        
        # Ensure df is not NaN or infinite
        if not np.isfinite(df) or df <= 0:
            df = len(sample_a) + len(sample_b) - 2
        
        # Calculate critical t-value
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # Calculate margin of error and confidence interval
        margin_of_error = t_critical * se
        lower_bound = mean_diff - margin_of_error
        upper_bound = mean_diff + margin_of_error
        
        return [float(lower_bound), float(upper_bound)]
    
    def _assess_clinical_significance(self, cohort_a_stats: Dict, cohort_b_stats: Dict, 
                                    statistical_tests: Dict) -> Dict[str, str]:
        """
        Assess clinical significance of observed statistical differences.
        
        This method translates statistical findings into clinical relevance by
        comparing observed differences to predefined minimum clinically meaningful
        differences. Clinical significance assessment is crucial because:
        - Statistical significance doesn't always imply clinical importance
        - Small differences may be statistically significant with large samples
        - Clinical decision-making requires meaningful effect sizes
        
        Args:
            cohort_a_stats (Dict): Statistics for cohort A.
            cohort_b_stats (Dict): Statistics for cohort B.
            statistical_tests (Dict): Statistical test results.
            
        Returns:
            Dict[str, str]: Clinical significance assessments.
        """
        clinical_sig = {}
        thresholds = self.config['clinical_thresholds']
        
        # Outcome score clinical significance assessment
        # Based on minimum clinically important difference (MCID)
        if 'outcome_stats' in cohort_a_stats and 'outcome_stats' in cohort_b_stats:
            mean_diff = abs(cohort_a_stats['outcome_stats']['mean'] - 
                           cohort_b_stats['outcome_stats']['mean'])
            
            if mean_diff >= thresholds['outcome_score_meaningful_diff']:
                clinical_sig['outcome_score'] = 'clinically_significant'
            else:
                clinical_sig['outcome_score'] = 'not_clinically_significant'
        
        # Compliance clinical significance assessment
        # Based on adherence thresholds that impact treatment effectiveness
        if 'compliance_stats' in cohort_a_stats and 'compliance_stats' in cohort_b_stats:
            compliance_diff = abs(cohort_a_stats['compliance_stats']['mean'] - 
                                cohort_b_stats['compliance_stats']['mean'])
            
            if compliance_diff >= thresholds['compliance_meaningful_diff']:
                clinical_sig['compliance'] = 'clinically_significant'
            else:
                clinical_sig['compliance'] = 'not_clinically_significant'
        
        # Adverse events clinical significance assessment
        # Based on safety thresholds that impact risk-benefit analysis
        if 'adverse_events' in cohort_a_stats and 'adverse_events' in cohort_b_stats:
            ae_rate_diff = abs(cohort_a_stats['adverse_events']['event_rate'] - 
                             cohort_b_stats['adverse_events']['event_rate'])
            
            if ae_rate_diff >= thresholds['adverse_event_rate_threshold']:
                clinical_sig['adverse_events'] = 'clinically_significant'
            else:
                clinical_sig['adverse_events'] = 'not_clinically_significant'
        
        return clinical_sig
    
    def _generate_recommendations(self, cohort_a_stats: Dict, cohort_b_stats: Dict, 
                                statistical_tests: Dict, clinical_significance: Dict) -> List[str]:
        """
        Generate evidence-based clinical recommendations from cohort comparison analysis.
        
        This method synthesizes statistical test results, effect sizes, and clinical
        significance assessments to provide actionable recommendations for:
        - Treatment protocol optimization
        - Safety monitoring enhancements
        - Patient adherence improvement strategies
        - Study design modifications
        
        Recommendations are prioritized based on clinical impact and statistical evidence.
        
        Args:
            cohort_a_stats (Dict): Statistics for cohort A.
            cohort_b_stats (Dict): Statistics for cohort B.
            statistical_tests (Dict): Statistical test results.
            clinical_significance (Dict): Clinical significance assessments.
            
        Returns:
            List[str]: Prioritized list of clinical recommendations.
        """
        recommendations = []
        
        # Outcome-based recommendations - Primary efficacy considerations
        if 'outcome_comparison' in statistical_tests:
            outcome_test = statistical_tests['outcome_comparison']
            if outcome_test['significant'] and clinical_significance.get('outcome_score') == 'clinically_significant':
                if outcome_test['mean_difference'] > 0:
                    recommendations.append(
                        f"🎯 EFFICACY: Cohort {cohort_a_stats['cohort_name']} demonstrates significantly "
                        f"superior outcomes (mean difference: {outcome_test['mean_difference']:.2f}, "
                        f"p={outcome_test['p_value']:.4f}). Consider adopting the treatment protocol "
                        f"used in Cohort {cohort_a_stats['cohort_name']} as the standard approach."
                    )
                else:
                    recommendations.append(
                        f"🎯 EFFICACY: Cohort {cohort_b_stats['cohort_name']} demonstrates significantly "
                        f"superior outcomes (mean difference: {abs(outcome_test['mean_difference']):.2f}, "
                        f"p={outcome_test['p_value']:.4f}). Consider adopting the treatment protocol "
                        f"used in Cohort {cohort_b_stats['cohort_name']} as the standard approach."
                    )
        
        # Safety-based recommendations - Adverse event considerations
        if 'adverse_events_comparison' in statistical_tests:
            ae_test = statistical_tests['adverse_events_comparison']
            if ae_test['significant'] and clinical_significance.get('adverse_events') == 'clinically_significant':
                if ae_test['rate_difference'] > 0:
                    recommendations.append(
                        f"⚠️ SAFETY: Cohort {cohort_a_stats['cohort_name']} shows significantly higher "
                        f"adverse event rate ({ae_test['rate_difference']:.1%} difference, "
                        f"p={ae_test['p_value']:.4f}). Implement enhanced safety monitoring and "
                        f"investigate potential protocol modifications to reduce adverse events."
                    )
                else:
                    recommendations.append(
                        f"⚠️ SAFETY: Cohort {cohort_b_stats['cohort_name']} shows significantly higher "
                        f"adverse event rate ({abs(ae_test['rate_difference']):.1%} difference, "
                        f"p={ae_test['p_value']:.4f}). Implement enhanced safety monitoring and "
                        f"investigate potential protocol modifications to reduce adverse events."
                    )
        
        # Compliance-based recommendations - Patient adherence optimization
        if 'compliance_comparison' in statistical_tests:
            compliance_test = statistical_tests['compliance_comparison']
            if compliance_test['significant'] and clinical_significance.get('compliance') == 'clinically_significant':
                if compliance_test['mean_difference'] > 0:
                    recommendations.append(
                        f"📊 ADHERENCE: Cohort {cohort_a_stats['cohort_name']} demonstrates significantly "
                        f"better compliance ({compliance_test['mean_difference']:.1f}% difference, "
                        f"p={compliance_test['p_value']:.4f}). Analyze adherence-promoting factors "
                        f"and implement similar support strategies for Cohort {cohort_b_stats['cohort_name']}."
                    )
                else:
                    recommendations.append(
                        f"📊 ADHERENCE: Cohort {cohort_b_stats['cohort_name']} demonstrates significantly "
                        f"better compliance ({abs(compliance_test['mean_difference']):.1f}% difference, "
                        f"p={compliance_test['p_value']:.4f}). Analyze adherence-promoting factors "
                        f"and implement similar support strategies for Cohort {cohort_a_stats['cohort_name']}."
                    )
        
        # Sample size and statistical power recommendations
        min_sample_size = self.config['statistical_config']['min_sample_size']
        recommended_sample_size = min_sample_size * 5  # Conservative recommendation
        
        if (cohort_a_stats['sample_size'] < recommended_sample_size or 
            cohort_b_stats['sample_size'] < recommended_sample_size):
            recommendations.append(
                f"📈 STUDY DESIGN: Consider increasing sample sizes (current: A={cohort_a_stats['sample_size']}, "
                f"B={cohort_b_stats['sample_size']}) to ≥{recommended_sample_size} per cohort for enhanced "
                f"statistical power and more robust clinical conclusions."
            )
        
        # Effect size interpretation recommendations
        if hasattr(self, '_last_effect_sizes'):  # This would be set during analysis
            for metric, effect_size in self._last_effect_sizes.items():
                if 'cohens_d' in metric:
                    if abs(effect_size) >= 0.8:
                        recommendations.append(
                            f"💪 EFFECT SIZE: Large effect size detected for {metric.replace('_cohens_d', '')} "
                            f"(Cohen's d = {effect_size:.2f}). This represents a clinically meaningful "
                            f"difference with high practical significance."
                        )
        
        # Default recommendation when no significant differences found
        if not recommendations:
            recommendations.append(
                f"🔍 MONITORING: No statistically or clinically significant differences detected "
                f"between cohorts. Continue current monitoring protocols and consider extended "
                f"follow-up period or larger sample sizes for more definitive conclusions. "
                f"Both treatment approaches appear equivalent based on current evidence."
            )
        
        return recommendations
    
    def perform_subgroup_analysis(self, data: pd.DataFrame, subgroup_column: str, 
                                outcome_column: str = 'outcome_score') -> Dict[str, Any]:
        """
        Perform comprehensive subgroup analysis within and across cohorts.
        
        This method enables analysis of treatment effects within patient subgroups
        (e.g., age groups, gender, comorbidities) to identify differential treatment
        responses and guide personalized medicine approaches.
        
        The analysis includes:
        - Descriptive statistics for each subgroup
        - ANOVA testing for multiple group comparisons
        - Subgroup-specific effect sizes
        - Clinical interpretation of subgroup differences
        
        Args:
            data (pd.DataFrame): Clinical trial data.
            subgroup_column (str): Column defining subgroups for analysis.
            outcome_column (str): Primary outcome variable for comparison.
            
        Returns:
            Dict[str, Any]: Comprehensive subgroup analysis results.
        """
        logger.info(f"Performing subgroup analysis by {subgroup_column} for outcome {outcome_column}")
        
        subgroup_results = {}
        min_sample_size = self.config['statistical_config']['min_sample_size']
        
        # Get unique subgroups and validate sample sizes
        subgroups = data[subgroup_column].unique()
        valid_subgroups = []
        
        # Calculate statistics for each subgroup
        for subgroup in subgroups:
            subgroup_data = data[data[subgroup_column] == subgroup]
            
            # Skip subgroups with insufficient sample size
            if len(subgroup_data) < min_sample_size:
                logger.warning(f"Subgroup {subgroup} has insufficient sample size: {len(subgroup_data)}")
                continue
            
            valid_subgroups.append(subgroup)
            
            # Calculate comprehensive subgroup statistics
            subgroup_stats = {
                'subgroup': subgroup,
                'sample_size': len(subgroup_data),
                'outcome_mean': float(subgroup_data[outcome_column].mean()),
                'outcome_std': float(subgroup_data[outcome_column].std()),
                'outcome_median': float(subgroup_data[outcome_column].median()),
                'outcome_iqr': float(subgroup_data[outcome_column].quantile(0.75) - 
                                   subgroup_data[outcome_column].quantile(0.25))
            }
            
            # Add adverse event statistics if available
            if 'adverse_event_flag' in subgroup_data.columns:
                subgroup_stats['adverse_event_rate'] = float(subgroup_data['adverse_event_flag'].mean())
                subgroup_stats['adverse_event_count'] = int(subgroup_data['adverse_event_flag'].sum())
            
            # Add compliance statistics if available
            if 'compliance_pct' in subgroup_data.columns:
                subgroup_stats['compliance_mean'] = float(subgroup_data['compliance_pct'].mean())
                subgroup_stats['compliance_std'] = float(subgroup_data['compliance_pct'].std())
            
            subgroup_results[str(subgroup)] = subgroup_stats
        
        # Perform ANOVA test if multiple valid subgroups exist (2 or more)
        if len(valid_subgroups) >= 2:
            subgroup_outcomes = []
            subgroup_names = []
            
            for subgroup in valid_subgroups:
                outcomes = data[data[subgroup_column] == subgroup][outcome_column].dropna()
                if len(outcomes) > 0:
                    subgroup_outcomes.append(outcomes)
                    subgroup_names.append(str(subgroup))
            
            if len(subgroup_outcomes) > 1:
                # Perform one-way ANOVA
                f_stat, p_value = stats.f_oneway(*subgroup_outcomes)
                
                subgroup_results['anova_test'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < self.config['statistical_config']['alpha'],
                    'subgroups_tested': subgroup_names,
                    'test_name': 'One-way ANOVA'
                }
                
                # If ANOVA is significant, perform post-hoc pairwise comparisons
                if p_value < self.config['statistical_config']['alpha']:
                    pairwise_results = {}
                    for i, subgroup1 in enumerate(subgroup_names):
                        for j, subgroup2 in enumerate(subgroup_names[i+1:], i+1):
                            data1 = subgroup_outcomes[i]
                            data2 = subgroup_outcomes[j]
                            
                            # Perform t-test with Bonferroni correction
                            t_stat, p_val = stats.ttest_ind(data1, data2)
                            corrected_alpha = self.config['statistical_config']['alpha'] / len(subgroup_names)
                            
                            pairwise_results[f"{subgroup1}_vs_{subgroup2}"] = {
                                't_statistic': float(t_stat),
                                'p_value': float(p_val),
                                'p_value_corrected': float(p_val * len(subgroup_names)),
                                'significant_corrected': (p_val * len(subgroup_names)) < self.config['statistical_config']['alpha'],
                                'mean_difference': float(data1.mean() - data2.mean())
                            }
                    
                    subgroup_results['pairwise_comparisons'] = pairwise_results
        
        logger.info(f"Subgroup analysis completed for {len(valid_subgroups)} subgroups")
        return subgroup_results
    
    def generate_cohort_summary_report(self, comparison_result: CohortComparisonResult) -> str:
        """
        Generate a comprehensive text summary report of the cohort comparison analysis.
        
        This method creates a formatted report suitable for clinical review, regulatory
        submission, or publication. The report includes:
        - Executive summary of key findings
        - Detailed statistical results
        - Clinical significance assessment
        - Evidence-based recommendations
        
        Args:
            comparison_result (CohortComparisonResult): Comprehensive comparison results.
            
        Returns:
            str: Formatted summary report for clinical review.
        """
        report = []
        
        # Header section with cohort identification
        cohort_a_name = comparison_result.cohort_a_stats['cohort_name']
        cohort_b_name = comparison_result.cohort_b_stats['cohort_name']
        report.append(f"CLINICAL COHORT COMPARISON ANALYSIS REPORT")
        report.append(f"Cohort {cohort_a_name} vs Cohort {cohort_b_name}")
        report.append("=" * 70)
        report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Confidence Level: {comparison_result.confidence_level:.0%}")
        
        # Sample size and demographics section
        report.append(f"\n📊 SAMPLE CHARACTERISTICS:")
        report.append("-" * 40)
        report.append(f"Cohort {cohort_a_name}:")
        report.append(f"  • Total records: {comparison_result.cohort_a_stats['sample_size']:,}")
        report.append(f"  • Unique patients: {comparison_result.cohort_a_stats['unique_patients']:,}")
        
        report.append(f"Cohort {cohort_b_name}:")
        report.append(f"  • Total records: {comparison_result.cohort_b_stats['sample_size']:,}")
        report.append(f"  • Unique patients: {comparison_result.cohort_b_stats['unique_patients']:,}")
        
        # Primary outcome analysis section
        if 'outcome_stats' in comparison_result.cohort_a_stats and 'outcome_stats' in comparison_result.cohort_b_stats:
            report.append(f"\n🎯 PRIMARY OUTCOME ANALYSIS:")
            report.append("-" * 40)
            
            outcome_a = comparison_result.cohort_a_stats['outcome_stats']
            outcome_b = comparison_result.cohort_b_stats['outcome_stats']
            
            report.append(f"Cohort {cohort_a_name} Outcomes:")
            report.append(f"  • Mean ± SD: {outcome_a['mean']:.2f} ± {outcome_a['std']:.2f}")
            report.append(f"  • Median (IQR): {outcome_a['median']:.2f} ({outcome_a['q25']:.2f}-{outcome_a['q75']:.2f})")
            report.append(f"  • Range: {outcome_a['min']:.2f} - {outcome_a['max']:.2f}")
            
            report.append(f"Cohort {cohort_b_name} Outcomes:")
            report.append(f"  • Mean ± SD: {outcome_b['mean']:.2f} ± {outcome_b['std']:.2f}")
            report.append(f"  • Median (IQR): {outcome_b['median']:.2f} ({outcome_b['q25']:.2f}-{outcome_b['q75']:.2f})")
            report.append(f"  • Range: {outcome_b['min']:.2f} - {outcome_b['max']:.2f}")
            
            # Statistical test results
            if 'outcome_comparison' in comparison_result.statistical_tests:
                test = comparison_result.statistical_tests['outcome_comparison']
                report.append(f"\nStatistical Analysis:")
                report.append(f"  • Test: {test['test_name']}")
                report.append(f"  • Mean difference: {test['mean_difference']:.2f}")
                report.append(f"  • 95% CI: [{test['confidence_interval'][0]:.2f}, {test['confidence_interval'][1]:.2f}]")
                report.append(f"  • P-value: {test['p_value']:.4f}")
                report.append(f"  • Result: {'Statistically significant' if test['significant'] else 'Not statistically significant'}")
                
                # Clinical significance
                if 'outcome_score' in comparison_result.clinical_significance:
                    clinical_sig = comparison_result.clinical_significance['outcome_score']
                    report.append(f"  • Clinical significance: {clinical_sig.replace('_', ' ').title()}")
        
        # Safety analysis section
        if 'adverse_events' in comparison_result.cohort_a_stats and 'adverse_events' in comparison_result.cohort_b_stats:
            report.append(f"\n⚠️ SAFETY ANALYSIS:")
            report.append("-" * 40)
            
            ae_a = comparison_result.cohort_a_stats['adverse_events']
            ae_b = comparison_result.cohort_b_stats['adverse_events']
            
            report.append(f"Cohort {cohort_a_name} Safety Profile:")
            report.append(f"  • Event rate: {ae_a['event_rate']:.1%} ({ae_a['total_events']} events)")
            report.append(f"  • Patients with events: {ae_a['patients_with_events']} ({ae_a['patient_event_rate']:.1%})")
            
            report.append(f"Cohort {cohort_b_name} Safety Profile:")
            report.append(f"  • Event rate: {ae_b['event_rate']:.1%} ({ae_b['total_events']} events)")
            report.append(f"  • Patients with events: {ae_b['patients_with_events']} ({ae_b['patient_event_rate']:.1%})")
            
            # Statistical comparison
            if 'adverse_events_comparison' in comparison_result.statistical_tests:
                test = comparison_result.statistical_tests['adverse_events_comparison']
                report.append(f"\nStatistical Analysis:")
                report.append(f"  • Test: {test['test_name']}")
                if 'rate_difference' in test:
                    report.append(f"  • Rate difference: {test['rate_difference']:.1%}")
                if 'relative_risk' in test and test['relative_risk'] is not None:
                    report.append(f"  • Relative risk: {test['relative_risk']:.2f}")
                if 'p_value' in test and test['p_value'] is not None:
                    report.append(f"  • P-value: {test['p_value']:.4f}")
                    if 'significant' in test and test['significant'] is not None:
                        report.append(f"  • Result: {'Statistically significant' if test['significant'] else 'Not statistically significant'}")
                    else:
                        report.append(f"  • Result: Statistical significance not determined")
                else:
                    report.append(f"  • Result: Descriptive analysis only")
        
        # Compliance analysis section
        if 'compliance_stats' in comparison_result.cohort_a_stats and 'compliance_stats' in comparison_result.cohort_b_stats:
            report.append(f"\n📊 COMPLIANCE ANALYSIS:")
            report.append("-" * 40)
            
            comp_a = comparison_result.cohort_a_stats['compliance_stats']
            comp_b = comparison_result.cohort_b_stats['compliance_stats']
            
            report.append(f"Cohort {cohort_a_name}: {comp_a['mean']:.1f}% ± {comp_a['std']:.1f}%")
            report.append(f"  • Poor compliance (<80%): {comp_a['below_80_pct']:.1f}% of patients")
            
            report.append(f"Cohort {cohort_b_name}: {comp_b['mean']:.1f}% ± {comp_b['std']:.1f}%")
            report.append(f"  • Poor compliance (<80%): {comp_b['below_80_pct']:.1f}% of patients")
        
        # Effect sizes section
        if comparison_result.effect_sizes:
            report.append(f"\n💪 EFFECT SIZES:")
            report.append("-" * 40)
            for metric, effect_size in comparison_result.effect_sizes.items():
                if 'cohens_d' in metric:
                    magnitude = 'Large' if abs(effect_size) >= 0.8 else 'Medium' if abs(effect_size) >= 0.5 else 'Small'
                    report.append(f"  • {metric.replace('_cohens_d', '').title()}: {effect_size:.3f} ({magnitude})")
                elif 'odds_ratio' in metric:
                    report.append(f"  • {metric.replace('_odds_ratio', '').title()}: {effect_size:.3f}")
        
        # Clinical recommendations section
        if comparison_result.recommendations:
            report.append(f"\n🎯 CLINICAL RECOMMENDATIONS:")
            report.append("-" * 40)
            for i, recommendation in enumerate(comparison_result.recommendations, 1):
                # Format multi-line recommendations
                lines = recommendation.split('. ')
                report.append(f"{i}. {lines[0]}.")
                for line in lines[1:]:
                    if line.strip():
                        report.append(f"   {line.strip()}.")
                report.append("")
        
        # Footer with analysis metadata
        report.append("=" * 70)
        report.append("Report generated by Clinical Insights Assistant - Cohort Analysis Module")
        report.append(f"Statistical significance threshold: α = {self.config['statistical_config']['alpha']}")
        
        return "\n".join(report)


def main():
    """
    Main function for testing and demonstrating the Cohort Analyzer capabilities.
    
    This function generates comprehensive synthetic clinical trial data with intentional
    differences between cohorts to demonstrate the full range of analysis capabilities:
    - Cohort A: Better outcomes but higher adverse event rate
    - Cohort B: Lower outcomes but better safety profile
    - Age-based subgroups for subgroup analysis
    
    The demonstration includes complete cohort comparison and subgroup analysis workflows.
    """
    # Set random seed for reproducible synthetic data generation
    np.random.seed(42)
    
    logger.info("Generating comprehensive synthetic clinical trial data for cohort analysis testing...")
    
    # Generate realistic synthetic clinical trial data with cohort differences
    data = []
    for i in range(200):  # 200 total records for robust statistical analysis
        patient_id = f"P{i+1:03d}"
        cohort = 'A' if i < 100 else 'B'  # Equal cohort sizes
        
        # Create differential treatment effects between cohorts
        # Cohort A: Superior efficacy but increased adverse events (risk-benefit trade-off)
        if cohort == 'A':
            outcome = np.random.normal(85, 8)      # Higher mean outcome score
            compliance = np.random.normal(90, 5)   # Good compliance
            adverse_event = 1 if np.random.random() < 0.15 else 0  # 15% AE rate
        else:  # Cohort B: Lower efficacy but better safety profile
            outcome = np.random.normal(78, 10)     # Lower mean outcome score
            compliance = np.random.normal(85, 8)   # Slightly lower compliance
            adverse_event = 1 if np.random.random() < 0.08 else 0  # 8% AE rate
        
        # Create comprehensive patient record
        data.append({
            'patient_id': patient_id,
            'trial_day': np.random.randint(1, 31),  # 30-day trial period
            'dosage_mg': 50,  # Standard dosage for comparison
            'compliance_pct': np.clip(compliance, 0, 100),  # Ensure valid percentage
            'adverse_event_flag': adverse_event,
            'outcome_score': np.clip(outcome, 0, 100),  # Ensure valid score range
            'cohort': cohort,
            'visit_date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 30)),
            'age_group': 'young' if np.random.random() < 0.6 else 'elderly',  # 60% young, 40% elderly
            'gender': 'Female' if np.random.random() < 0.5 else 'Male'  # Equal gender distribution
        })
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(data)
    
    logger.info(f"Generated synthetic dataset: {len(df)} records, {df['patient_id'].nunique()} patients, {df['cohort'].nunique()} cohorts")
    
    # Initialize Cohort Analyzer with default configuration
    analyzer = CohortAnalyzer()
    
    # Perform comprehensive cohort comparison
    logger.info("Performing comprehensive cohort comparison analysis...")
    comparison_result = analyzer.compare_cohorts(df, 'cohort', 'A', 'B')
    
    # Generate and display comprehensive summary report
    summary_report = analyzer.generate_cohort_summary_report(comparison_result)
    print("\n" + "="*80)
    print("COHORT ANALYSIS DEMONSTRATION")
    print("="*80)
    print(summary_report)
    
    # Perform subgroup analysis by age group
    logger.info("Performing subgroup analysis by age group...")
    subgroup_results = analyzer.perform_subgroup_analysis(df, 'age_group', 'outcome_score')
    
    print(f"\n🔍 SUBGROUP ANALYSIS RESULTS (by Age Group):")
    print("-" * 50)
    for subgroup, stats in subgroup_results.items():
        if subgroup not in ['anova_test', 'pairwise_comparisons']:
            print(f"{subgroup.upper()} GROUP:")
            print(f"  • Sample size: {stats['sample_size']}")
            print(f"  • Mean outcome: {stats['outcome_mean']:.2f} ± {stats['outcome_std']:.2f}")
            if 'adverse_event_rate' in stats:
                print(f"  • Adverse event rate: {stats['adverse_event_rate']:.1%}")
            if 'compliance_mean' in stats:
                print(f"  • Mean compliance: {stats['compliance_mean']:.1f}%")
            print()
    
    # Display ANOVA results if available
    if 'anova_test' in subgroup_results:
        anova = subgroup_results['anova_test']
        print(f"ANOVA Test Results:")
        print(f"  • F-statistic: {anova['f_statistic']:.3f}")
        print(f"  • P-value: {anova['p_value']:.4f}")
        print(f"  • Result: {'Significant differences between subgroups' if anova['significant'] else 'No significant differences'}")
    
    # Display effect size interpretation
    if comparison_result.effect_sizes:
        print(f"\n💪 EFFECT SIZE INTERPRETATION:")
        print("-" * 35)
        for metric, effect_size in comparison_result.effect_sizes.items():
            if 'cohens_d' in metric:
                if abs(effect_size) >= 0.8:
                    interpretation = "Large effect - clinically meaningful difference"
                elif abs(effect_size) >= 0.5:
                    interpretation = "Medium effect - moderate clinical importance"
                elif abs(effect_size) >= 0.2:
                    interpretation = "Small effect - limited clinical significance"
                else:
                    interpretation = "Negligible effect - minimal clinical relevance"
                
                print(f"{metric.replace('_cohens_d', '').title()}: {effect_size:.3f} - {interpretation}")
    
    print("\n" + "="*80)
    print("✅ Cohort analysis demonstration completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()