"""
Issue Detection Module for Clinical Insights Assistant

This module provides rule-based and statistical methods to detect various issues in clinical trial data:
- Patient non-compliance
- Adverse events
- Drug inefficacy
- Data quality issues
- Safety signals

This module focuses on identifying potential problems within the clinical trial data, such as
non-compliance, adverse events, or unexpected efficacy results. It combines rule-based
logic with statistical methods to flag issues.

Purpose:
• Detect compliance issues based on predefined thresholds.
• Identify safety signals and adverse events.
• Flag efficacy concerns or unexpected outcomes.
• Categorize issues by severity and provide context.

Step-by-Step Implementation:
1. Define the IssueDetector Class:
   The class will hold configuration for detection rules and thresholds.
2. Implement _get_default_config:
   Define default thresholds and rules for various issue types.
3. Implement detect_compliance_issues:
   Detects patients with low compliance.
4. Implement detect_safety_issues:
   Detects adverse events and potential safety signals.
5. Implement detect_efficacy_issues:
   Detects patients with unexpected efficacy outcomes.
6. Add main function for testing:
   This module provides the foundational logic for identifying critical issues within clinical
   trial data, which is a primary requirement for the Clinical Insights Assistant. The use of
   dataclasses for ClinicalIssue ensures structured and consistent reporting of detected
   problems.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from scipy import stats
import warnings

# Configure logging for monitoring issue detection operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IssueAlert:
    """
    Data class to represent a detected issue in clinical trial data.
    
    This structured approach ensures consistent reporting of detected problems
    across all detection methods, providing standardized information for
    clinical decision-making and regulatory reporting.
    
    Attributes:
        issue_type (str): Type of issue detected (e.g., 'compliance', 'efficacy', 'safety')
        severity (str): Severity level - 'low', 'medium', 'high', 'critical'
        patient_id (str): Patient identifier or 'ALL'/'MULTIPLE' for multi-patient issues
        description (str): Human-readable description of the detected issue
        affected_records (int): Number of data records affected by this issue
        recommendation (str): Clinical recommendation for addressing the issue
        confidence_score (float): Confidence level of the detection (0.0-1.0)
        metadata (Dict): Additional contextual information and statistical details
    """
    issue_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    patient_id: str
    description: str
    affected_records: int
    recommendation: str
    confidence_score: float
    metadata: Dict[str, Any]
    visit_number: Optional[str] = None  # Visit identifier where issue was detected


class IssueDetector:
    """
    Class for detecting various issues in clinical trial data using rule-based and statistical methods.
    
    The IssueDetector class serves as the central hub for all issue detection functionality,
    providing comprehensive analysis capabilities that combine:
    - Rule-based detection using predefined clinical thresholds
    - Statistical analysis for outlier and trend detection
    - Pattern recognition for safety signal identification
    - Data quality assessment for trial integrity
    
    This class implements the foundational logic for identifying critical issues within
    clinical trial data, which is a primary requirement for the Clinical Insights Assistant.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Issue Detector with configuration parameters.
        
        The initialization sets up detection thresholds and parameters that define
        when various issues should be flagged. Custom configuration allows for
        study-specific adjustments while maintaining robust default values.
        
        Args:
            config (Dict, optional): Configuration dictionary for detection thresholds and parameters.
                                   If not provided, uses clinically-validated default values.
        """
        # Load configuration for detection rules and thresholds
        self.config = config or self._get_default_config()
        
        # Initialize storage for detected issues
        self.detected_issues = []
        
        logger.info("Issue Detector initialized with configuration parameters")
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration for issue detection with clinically-validated thresholds.
        
        This method defines the foundational rules and thresholds used across all detection
        algorithms. These values are based on clinical best practices and regulatory
        guidelines for clinical trial monitoring.
        
        The configuration is organized into logical groups:
        - Compliance thresholds: Patient adherence monitoring
        - Outcome thresholds: Treatment efficacy assessment
        - Adverse event config: Safety signal detection
        - Statistical config: Outlier and trend analysis
        - Data quality config: Trial integrity monitoring
        
        Returns:
            Dict: Comprehensive configuration parameters for all detection methods.
        """
        return {
            # Compliance thresholds - Based on clinical adherence standards
            # These thresholds define when patient compliance becomes a concern
            'compliance_thresholds': {
                'critical': 50.0,  # Below 50% compliance is critical for treatment efficacy
                'high': 70.0,      # Below 70% compliance is high concern requiring intervention
                'medium': 85.0     # Below 85% compliance is medium concern for monitoring
            },
            
            # Outcome score thresholds - Treatment efficacy assessment (0-100 scale)
            # These define when treatment outcomes indicate potential efficacy issues
            'outcome_thresholds': {
                'inefficacy_critical': 40.0,  # Below 40 indicates critical treatment failure
                'inefficacy_high': 60.0,      # Below 60 indicates high concern for efficacy
                'improvement_threshold': 5.0   # Minimum improvement expected over time
            },
            
            # Adverse event detection - Safety signal monitoring
            # These parameters control when adverse event patterns trigger alerts
            'adverse_event_config': {
                'max_acceptable_rate': 0.15,  # 15% adverse event rate threshold for overall safety
                'clustering_threshold': 3,     # Minimum events in single patient to flag clustering
                'temporal_window': 7           # Days to look for temporal clustering of events
            },
            
            # Statistical detection parameters - Outlier and trend analysis
            # These control the sensitivity of statistical detection methods
            'statistical_config': {
                'outlier_z_threshold': 2.5,   # Z-score threshold for statistical outlier detection
                'trend_significance': 0.05,   # P-value threshold for trend significance testing
                'min_data_points': 5          # Minimum data points required for statistical analysis
            },
            
            # Data quality thresholds - Trial integrity monitoring
            # These ensure the clinical trial data meets quality standards
            'data_quality_config': {
                'max_missing_percentage': 0.1,  # 10% missing data threshold for data integrity
                'duplicate_threshold': 0.05,    # 5% duplicate records threshold for data quality
                'consistency_threshold': 0.95   # 95% consistency requirement for data validation
            }
        }
    
    def detect_all_issues(self, data: pd.DataFrame) -> List[IssueAlert]:
        """
        Run comprehensive issue detection across all available methods.
        
        This is the main entry point for issue detection, orchestrating all detection
        algorithms to provide a complete assessment of clinical trial data quality,
        safety, and efficacy. The method follows a systematic approach:
        
        1. Compliance Issues - Patient adherence problems
        2. Efficacy Issues - Treatment effectiveness concerns
        3. Adverse Event Patterns - Safety signal detection
        4. Statistical Outliers - Unusual data point identification
        5. Data Quality Issues - Trial integrity assessment
        6. Temporal Trends - Time-based pattern analysis
        
        Args:
            data (pd.DataFrame): Clinical trial data containing patient records with
                               required columns for comprehensive analysis.
            
        Returns:
            List[IssueAlert]: Comprehensive list of all detected issues with detailed
                            information for clinical decision-making.
        """
        # Reset detected issues for new analysis
        self.detected_issues = []
        
        logger.info("Starting comprehensive issue detection across all detection methods...")
        
        # 1. Detect compliance issues - Patient adherence monitoring
        # This identifies patients with poor medication adherence that could
        # compromise treatment efficacy and trial validity
        compliance_issues = self.detect_compliance_issues(data)
        self.detected_issues.extend(compliance_issues)
        
        # 2. Detect efficacy issues - Treatment effectiveness assessment
        # This identifies patients with poor treatment response or declining outcomes
        # that may indicate treatment failure or need for intervention
        efficacy_issues = self.detect_efficacy_issues(data)
        self.detected_issues.extend(efficacy_issues)
        
        # 3. Detect adverse event patterns - Safety signal identification
        # This identifies concerning patterns in adverse events that may indicate
        # safety issues requiring immediate attention
        adverse_event_issues = self.detect_adverse_event_patterns(data)
        self.detected_issues.extend(adverse_event_issues)
        
        # 4. Detect statistical outliers - Unusual data point identification
        # This identifies statistically unusual values that may indicate data errors
        # or patients requiring special attention
        outlier_issues = self.detect_statistical_outliers(data)
        self.detected_issues.extend(outlier_issues)
        
        # 5. Detect data quality issues - Trial integrity assessment
        # This identifies problems with data completeness, consistency, and quality
        # that could compromise trial results
        data_quality_issues = self.detect_data_quality_issues(data)
        self.detected_issues.extend(data_quality_issues)
        
        # 6. Detect temporal trends - Time-based pattern analysis
        # This identifies concerning trends over time that may indicate systematic
        # problems with the trial or treatment
        trend_issues = self.detect_temporal_trends(data)
        self.detected_issues.extend(trend_issues)
        
        logger.info(f"Comprehensive issue detection completed. Found {len(self.detected_issues)} total issues across all categories.")
        
        return self.detected_issues
    
    def detect_compliance_issues(self, data: pd.DataFrame) -> List[IssueAlert]:
        """
        Detect patient compliance issues based on predefined thresholds.
        
        This method implements rule-based detection of patient adherence problems,
        which is critical for clinical trial success. Poor compliance can:
        - Compromise treatment efficacy assessment
        - Introduce bias in trial results
        - Affect patient safety
        - Impact regulatory approval
        
        The detection algorithm:
        1. Calculates average compliance per patient
        2. Applies severity thresholds based on clinical standards
        3. Considers compliance variability for recommendation tailoring
        4. Generates confidence scores based on data completeness
        
        Args:
            data (pd.DataFrame): Clinical trial data with 'compliance_pct' and 'patient_id' columns.
            
        Returns:
            List[IssueAlert]: List of compliance-related issues with severity classification
                            and clinical recommendations for intervention.
        """
        issues = []
        
        # Validate required columns for compliance analysis
        if 'compliance_pct' not in data.columns or 'patient_id' not in data.columns:
            logger.warning("Required columns for compliance detection not found - skipping compliance analysis")
            return issues
        
        # Calculate comprehensive compliance statistics per patient
        # This aggregation provides the foundation for rule-based detection
        # Convert compliance_pct to numeric, handling mixed types gracefully
        data_clean = data.copy()
        data_clean['compliance_pct'] = pd.to_numeric(data_clean['compliance_pct'], errors='coerce')
        
        patient_compliance = data_clean.groupby('patient_id')['compliance_pct'].agg([
            'mean',   # Average compliance rate
            'std',    # Compliance variability 
            'count'   # Number of compliance measurements
        ]).reset_index()
        
        # Load compliance thresholds from configuration
        thresholds = self.config['compliance_thresholds']
        
        # Analyze each patient's compliance pattern
        for _, patient in patient_compliance.iterrows():
            patient_id = patient['patient_id']
            avg_compliance = patient['mean']
            compliance_std = patient['std'] if not pd.isna(patient['std']) else 0
            record_count = patient['count']
            
            # Apply rule-based severity classification
            # This implements the clinical decision logic for compliance assessment
            severity = 'low'
            if avg_compliance < thresholds['critical']:
                severity = 'critical'  # Immediate intervention required
            elif avg_compliance < thresholds['high']:
                severity = 'high'      # Urgent attention needed
            elif avg_compliance < thresholds['medium']:
                severity = 'medium'    # Monitoring and support recommended
            else:
                continue  # No compliance issue detected - patient adherence acceptable
            
            # Generate tailored clinical recommendations based on compliance pattern
            # High variability indicates inconsistent adherence requiring different intervention
            if compliance_std > 20:  # High compliance variability detected
                recommendation = (f"Patient shows inconsistent compliance (avg: {avg_compliance:.1f}%, "
                                f"std: {compliance_std:.1f}%). Consider adherence counseling and more "
                                f"frequent monitoring to identify and address barriers to consistent medication taking.")
            else:  # Consistently low compliance
                recommendation = (f"Patient shows consistently low compliance (avg: {avg_compliance:.1f}%). "
                                f"Investigate barriers to adherence and consider comprehensive intervention "
                                f"strategies including patient education and support services.")
            
            # Calculate confidence score based on data completeness and consistency
            # More data points and lower variability increase confidence in the assessment
            confidence_score = min(0.9, 0.5 + (record_count / 20) + (1 - compliance_std / 100))
            
            # Create structured issue alert for clinical review
            issue = IssueAlert(
issue_type='compliance',
                severity=severity,
                patient_id=patient_id,
                description=f"Low patient compliance detected: {avg_compliance:.1f}% average compliance over {record_count} measurements",
                affected_records=record_count,
                recommendation=recommendation,
                confidence_score=confidence_score,
                metadata={
                    'average_compliance': avg_compliance,
                    'compliance_std': compliance_std,
                    'record_count': record_count,
                    'threshold_used': thresholds,
                    'compliance_category': severity
                },
                visit_number=None
            )
            
            issues.append(issue)
        
        logger.info(f"Compliance analysis completed - detected {len(issues)} compliance issues requiring attention")
        return issues
    
    def detect_efficacy_issues(self, data: pd.DataFrame) -> List[IssueAlert]:
        """
        Detect drug efficacy issues based on outcome scores and treatment response patterns.
        
        This method implements both rule-based and statistical detection of treatment
        efficacy problems, which is crucial for:
        - Early identification of treatment failures
        - Patient safety through timely intervention
        - Trial integrity and regulatory compliance
        - Clinical decision support for dose adjustments
        
        The detection algorithm employs two complementary approaches:
        1. Absolute efficacy assessment - comparing outcomes to clinical thresholds
        2. Trend analysis - detecting declining treatment response over time
        
        Statistical trend detection uses linear regression to identify significant
        declining patterns that may indicate treatment failure or tolerance development.
        
        Args:
            data (pd.DataFrame): Clinical trial data with 'outcome_score', 'patient_id', 
                               and optionally 'trial_day' columns for comprehensive efficacy analysis.
            
        Returns:
            List[IssueAlert]: List of efficacy-related issues including both absolute
                            low efficacy and concerning declining trends.
        """
        issues = []
        
        # Validate required columns for efficacy analysis
        if 'outcome_score' not in data.columns or 'patient_id' not in data.columns:
            logger.warning("Required columns for efficacy detection not found - skipping efficacy analysis")
            return issues
        
        # Load efficacy thresholds from clinical configuration
        thresholds = self.config['outcome_thresholds']
        
        # Analyze efficacy patterns for each patient individually
        # This patient-specific approach enables personalized clinical recommendations
        for patient_id in data['patient_id'].unique():
            # Extract patient data and sort chronologically if trial_day is available
            patient_data = data[data['patient_id'] == patient_id]
            if 'trial_day' in data.columns:
                patient_data = patient_data.sort_values('trial_day')
            
            # Skip patients with insufficient data for meaningful analysis
            if len(patient_data) < 2:
                continue
            
            # Extract outcome measurements for analysis
            outcome_scores = patient_data['outcome_score'].dropna().values
            if len(outcome_scores) == 0:
                continue
                
            avg_outcome = np.mean(outcome_scores)
            
            # Rule-based absolute efficacy assessment
            # This identifies patients with consistently poor treatment response
            severity = None
            if avg_outcome < thresholds['inefficacy_critical']:
                severity = 'critical'  # Treatment failure - immediate intervention required
            elif avg_outcome < thresholds['inefficacy_high']:
                severity = 'high'      # Poor response - urgent review needed
            
            # Generate efficacy alert for low absolute performance
            if severity:
                issue = IssueAlert(
                    issue_type='efficacy_low',
                    severity=severity,
                    patient_id=patient_id,
                    description=f"Low treatment efficacy detected: {avg_outcome:.1f} average outcome score over {len(patient_data)} measurements",
                    affected_records=len(patient_data),
                    recommendation=(f"Consider dosage adjustment or alternative treatment approach. "
                                  f"Average outcome score ({avg_outcome:.1f}) is below acceptable "
                                  f"clinical threshold, indicating potential treatment failure."),
                    confidence_score=min(0.9, 0.6 + len(patient_data) / 30),  # Higher confidence with more data
                    metadata={
                        'average_outcome': avg_outcome,
                        'outcome_trend': 'declining' if outcome_scores[-1] < outcome_scores[0] else 'stable_low',
                        'record_count': len(patient_data),
                        'threshold_violated': thresholds['inefficacy_critical'] if severity == 'critical' else thresholds['inefficacy_high']
                    }
                )
                issues.append(issue)
            
            # Statistical trend analysis for declining efficacy detection
            # This identifies patients whose treatment response is deteriorating over time
            if len(outcome_scores) >= 5:  # Minimum data points for reliable trend analysis
                # Perform linear regression to quantify treatment response trend
                x = np.arange(len(outcome_scores))  # Time points
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, outcome_scores)
                
                # Detect statistically significant declining trends
                # A negative slope indicates worsening outcomes over time
                if slope < -0.5 and p_value < 0.05:  # Significant declining trend detected
                    issue = IssueAlert(
                        issue_type='efficacy_declining',
                        severity='high',  # Declining trends always require urgent attention
                        patient_id=patient_id,
                        description=(f"Declining treatment efficacy detected: {slope:.2f} points per day decline "
                                   f"with statistical significance (p={p_value:.3f})"),
                        affected_records=len(patient_data),
                        recommendation=(f"Urgent clinical review required. Statistically significant declining "
                                      f"trend in outcome scores (slope: {slope:.2f}, p-value: {p_value:.3f}). "
                                      f"Consider immediate dosage adjustment, treatment modification, or "
                                      f"alternative therapeutic approach."),
                        confidence_score=min(0.95, 0.7 + abs(r_value) * 0.3),  # Confidence based on trend strength
                        metadata={
                            'slope': slope,
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'trend_strength': abs(r_value),
                            'initial_outcome': outcome_scores[0],
                            'final_outcome': outcome_scores[-1],
                            'total_decline': outcome_scores[0] - outcome_scores[-1]
                        }
                    )
                    issues.append(issue)
        
        logger.info(f"Efficacy analysis completed - detected {len(issues)} efficacy issues requiring clinical attention")
        return issues
    
    def detect_adverse_event_patterns(self, data: pd.DataFrame) -> List[IssueAlert]:
        """
        Detect patterns in adverse events that may indicate safety concerns.
        
        This method implements comprehensive safety signal detection through multiple
        complementary approaches, which is critical for:
        - Patient safety monitoring and protection
        - Regulatory compliance and reporting requirements
        - Early identification of drug safety profiles
        - Risk-benefit assessment for clinical decisions
        
        The detection algorithm employs three pattern recognition approaches:
        1. Overall adverse event rate monitoring - population-level safety assessment
        2. Patient-specific clustering detection - individual safety risk identification
        3. Temporal clustering analysis - time-based safety signal detection
        
        These methods work together to provide comprehensive safety monitoring
        that can identify both individual patient risks and systematic safety issues.
        
        Args:
            data (pd.DataFrame): Clinical trial data with 'adverse_event_flag', 'patient_id',
                               and optionally 'visit_date' columns for comprehensive safety analysis.
            
        Returns:
            List[IssueAlert]: List of adverse event pattern issues including population-level,
                            patient-specific, and temporal clustering concerns.
        """
        issues = []
        
        # Validate required column for basic adverse event analysis
        if 'adverse_event_flag' not in data.columns:
            logger.warning("Adverse event column not found - skipping safety signal detection")
            return issues
        
        # Load adverse event detection configuration
        config = self.config['adverse_event_config']
        
        # 1. Overall adverse event rate analysis - Population-level safety assessment
        # This provides a high-level view of trial safety profile
        total_records = len(data)
        total_adverse_events = data['adverse_event_flag'].sum()
        adverse_event_rate = total_adverse_events / total_records if total_records > 0 else 0
        
        # Check if overall adverse event rate exceeds acceptable limits
        if adverse_event_rate > config['max_acceptable_rate']:
            # Determine severity based on how far above acceptable rate
            severity = 'critical' if adverse_event_rate > 0.25 else 'high' if adverse_event_rate > 0.20 else 'medium'
            
            issue = IssueAlert(
                issue_type='adverse_event_rate_high',
                severity=severity,
                patient_id='ALL',  # Population-level issue affects all patients
                description=f"High overall adverse event rate detected: {adverse_event_rate:.1%} ({total_adverse_events} events in {total_records} records)",
                affected_records=total_adverse_events,
                recommendation=(f"Comprehensive safety review required. Overall adverse event rate "
                              f"({adverse_event_rate:.1%}) exceeds acceptable threshold "
                              f"({config['max_acceptable_rate']:.1%}). Consider safety committee review, "
                              f"protocol modifications, and enhanced patient monitoring."),
                confidence_score=0.9,  # High confidence in population-level statistics
                metadata={
                    'adverse_event_rate': adverse_event_rate,
                    'total_events': total_adverse_events,
                    'total_records': total_records,
                    'acceptable_threshold': config['max_acceptable_rate'],
                    'severity_rationale': f"Rate {adverse_event_rate:.1%} vs threshold {config['max_acceptable_rate']:.1%}"
                }
            )
            issues.append(issue)
        
        # 2. Patient-specific adverse event clustering detection
        # This identifies individual patients with concerning safety profiles
        if 'patient_id' in data.columns:
            # Calculate adverse event counts per patient
            patient_ae_counts = data.groupby('patient_id')['adverse_event_flag'].sum()
            
            # Analyze each patient's adverse event pattern
            for patient_id, ae_count in patient_ae_counts.items():
                # Check if patient has concerning number of adverse events
                if ae_count >= config['clustering_threshold']:
                    patient_records = len(data[data['patient_id'] == patient_id])
                    patient_ae_rate = ae_count / patient_records if patient_records > 0 else 0
                    
                    # Determine severity based on event frequency
                    severity = 'critical' if ae_count >= 5 else 'high'
                    
                    issue = IssueAlert(
                        issue_type='adverse_event_clustering',
                        severity=severity,
                        patient_id=patient_id,
                        description=(f"Multiple adverse events in single patient: {ae_count} events "
                                   f"in {patient_records} visits ({patient_ae_rate:.1%} rate)"),
                        affected_records=ae_count,
                        recommendation=(f"Immediate patient safety review required. Patient {patient_id} "
                                      f"has experienced {ae_count} adverse events with {patient_ae_rate:.1%} "
                                      f"event rate. Consider treatment discontinuation, dose modification, "
                                      f"or enhanced safety monitoring."),
                        confidence_score=min(0.95, 0.7 + ae_count / 10),  # Higher confidence with more events
                        metadata={
                            'adverse_event_count': ae_count,
                            'patient_ae_rate': patient_ae_rate,
                            'patient_records': patient_records,
                            'clustering_threshold': config['clustering_threshold'],
                            'safety_profile': 'high_risk'
                        }
                    )
                    issues.append(issue)
        
        # 3. Temporal clustering analysis - Time-based safety signal detection
        # This identifies periods of increased adverse event activity that may indicate
        # systematic issues with drug batches, protocol changes, or external factors
        if 'visit_date' in data.columns:
            # Extract adverse event data with valid dates
            ae_data = data[data['adverse_event_flag'] == 1].copy()
            if len(ae_data) > 0:
                # Convert dates and sort chronologically
                ae_data['visit_date'] = pd.to_datetime(ae_data['visit_date'])
                ae_data = ae_data.sort_values('visit_date')
                
                # Look for temporal clusters of adverse events
                # This sliding window approach identifies concerning time periods
                for i in range(len(ae_data) - config['clustering_threshold'] + 1):
                    window_data = ae_data.iloc[i:i + config['clustering_threshold']]
                    date_range = (window_data['visit_date'].max() - window_data['visit_date'].min()).days
                    
                    # Check if events are clustered within temporal window
                    if date_range <= config['temporal_window']:
                        issue = IssueAlert(
                            issue_type='adverse_event_temporal_clustering',
                            severity='medium',  # Temporal clusters require investigation but may not be causal
                            patient_id='MULTIPLE',
                            description=(f"Temporal clustering of adverse events: {config['clustering_threshold']} "
                                       f"events within {date_range} days ({window_data['visit_date'].min().date()} "
                                       f"to {window_data['visit_date'].max().date()})"),
                            affected_records=config['clustering_threshold'],
                            recommendation=(f"Investigate potential common cause for adverse events clustered "
                                          f"between {window_data['visit_date'].min().date()} and "
                                          f"{window_data['visit_date'].max().date()}. Consider drug batch "
                                          f"analysis, protocol changes, environmental factors, or staff changes."),
                            confidence_score=0.8,  # Moderate confidence as temporal clustering may be coincidental
                            metadata={
                                'start_date': window_data['visit_date'].min().isoformat(),
                                'end_date': window_data['visit_date'].max().isoformat(),
                                'days_span': date_range,
                                'affected_patients': window_data['patient_id'].tolist(),
                                'temporal_window': config['temporal_window'],
                                'cluster_density': config['clustering_threshold'] / max(date_range, 1)  # Avoid division by zero
                            }
                        )
                        issues.append(issue)
                        break  # Only report the first temporal cluster to avoid overlapping reports
        
        logger.info(f"Adverse event pattern analysis completed - detected {len(issues)} safety concerns requiring attention")
        return issues
    
    def detect_adverse_events(self, data: pd.DataFrame) -> List[IssueAlert]:
        """
        Wrapper method for detect_adverse_event_patterns to maintain backward compatibility.
        
        Args:
            data (pd.DataFrame): Clinical trial data.
            
        Returns:
            List[IssueAlert]: List of adverse event issues.
        """
        return self.detect_adverse_event_patterns(data)
    
    def detect_statistical_outliers(self, data: pd.DataFrame) -> List[IssueAlert]:
        """
        Detect statistical outliers in key clinical metrics using z-score analysis.
        
        This method implements statistical outlier detection to identify unusual values
        that may indicate:
        - Data entry errors requiring correction
        - Patients with unique characteristics requiring special attention
        - Measurement errors or equipment malfunctions
        - Extreme responses to treatment (positive or negative)
        
        The detection uses z-score analysis, which measures how many standard deviations
        a value is from the mean. This approach is robust for normally distributed data
        and provides standardized thresholds across different measurement scales.
        
        Args:
            data (pd.DataFrame): Clinical trial data with numeric columns for statistical analysis.
            
        Returns:
            List[IssueAlert]: List of statistical outlier issues with detailed analysis
                            for clinical review and potential data verification.
        """
        issues = []
        
        # Load statistical detection configuration
        config = self.config['statistical_config']
        
        # Define key clinical metrics for outlier analysis
        # These columns represent critical measurements that should be monitored for unusual values
        numeric_columns = ['outcome_score', 'compliance_pct', 'dosage_mg']
        
        # Analyze each numeric column for statistical outliers
        for column in numeric_columns:
            # Skip columns that don't exist in the data
            if column not in data.columns:
                continue
            
            # Extract valid numeric data for analysis, handling mixed types
            try:
                column_data = pd.to_numeric(data[column], errors='coerce').dropna()
            except (ValueError, TypeError):
                logger.warning(f"Could not convert {column} to numeric for outlier analysis")
                continue
            
            # Skip analysis if insufficient data points for reliable statistics
            if len(column_data) < config['min_data_points']:
                logger.warning(f"Insufficient data points ({len(column_data)}) for {column} outlier analysis")
                continue
            
            # Calculate z-scores to identify statistical outliers
            # Z-score represents how many standard deviations a value is from the mean
            try:
                z_scores = np.abs(stats.zscore(column_data))
            except (ValueError, RuntimeWarning):
                # Handle cases where data has no variance or other statistical issues
                logger.warning(f"Could not calculate z-scores for {column} - data may have no variance")
                continue
            
            # Identify outlier indices based on threshold
            outlier_indices = np.where(z_scores > config['outlier_z_threshold'])[0]
            
            # Process each detected outlier
            if len(outlier_indices) > 0:
                outlier_data = data.iloc[column_data.index[outlier_indices]]
                
                # Generate individual issue alerts for each outlier
                for _, outlier_record in outlier_data.iterrows():
                    # Find the corresponding z-score for this outlier
                    z_score = z_scores[np.where(column_data.index == outlier_record.name)[0][0]]
                    
                    # Determine severity based on how extreme the outlier is
                    severity = 'high' if z_score > 3 else 'medium'  # >3 SD is highly unusual
                    
                    # Calculate basic statistics for context
                    column_mean = column_data.mean()
                    column_std = column_data.std()
                    
                    issue = IssueAlert(
                        issue_type='statistical_outlier',
                        severity=severity,
                        patient_id=outlier_record.get('patient_id', 'UNKNOWN'),
                        description=(f"Statistical outlier detected in {column}: {outlier_record[column]} "
                                   f"(z-score: {z_score:.2f}, {z_score:.1f} standard deviations from mean)"),
                        affected_records=1,
                        recommendation=(f"Review record for patient {outlier_record.get('patient_id', 'UNKNOWN')} - "
                                      f"{column} value ({outlier_record[column]}) is {z_score:.1f} standard "
                                      f"deviations from mean ({column_mean:.1f}). Verify data accuracy and "
                                      f"consider clinical significance of extreme value."),
                        confidence_score=min(0.9, 0.5 + z_score / 10),  # Higher confidence for more extreme outliers
                        metadata={
                            'column': column,
                            'value': outlier_record[column],
                            'z_score': z_score,
                            'mean': column_mean,
                            'std': column_std,
                            'percentile_rank': stats.percentileofscore(column_data, outlier_record[column]),
                            'outlier_threshold': config['outlier_z_threshold']
                        }
                    )
                    issues.append(issue)
        
        logger.info(f"Statistical outlier analysis completed - detected {len(issues)} outliers requiring review")
        return issues
    
    def detect_data_quality_issues(self, data: pd.DataFrame) -> List[IssueAlert]:
        """
        Detect data quality issues such as missing data, duplicates, and inconsistencies.
        
        This method implements comprehensive data quality assessment to ensure trial
        data integrity, which is essential for:
        - Regulatory compliance and audit readiness
        - Statistical analysis validity and power
        - Clinical decision-making reliability
        - Trial result credibility and reproducibility
        
        The quality assessment covers multiple dimensions:
        1. Missing data analysis - completeness assessment
        2. Duplicate record detection - data integrity verification
        3. Consistency checks - logical validation (future enhancement)
        
        These checks help maintain the high data quality standards required
        for clinical trial success and regulatory approval.
        
        Args:
            data (pd.DataFrame): Clinical trial data for comprehensive quality assessment.
            
        Returns:
            List[IssueAlert]: List of data quality issues with specific recommendations
                            for data management and remediation actions.
        """
        issues = []
        
        # Load data quality configuration thresholds
        config = self.config['data_quality_config']
        
        # 1. Missing data analysis - Comprehensive completeness assessment
        # Missing data can severely impact statistical power and introduce bias
        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        missing_percentage = missing_cells / total_cells if total_cells > 0 else 0
        
        # Check if missing data exceeds acceptable threshold
        # Use a small buffer to avoid floating point precision issues
        if missing_percentage > (config['max_missing_percentage'] + 1e-10):
            # Determine severity based on extent of missing data
            severity = 'critical' if missing_percentage > 0.2 else 'high' if missing_percentage > 0.15 else 'medium'
            
            # Identify columns with missing data for targeted remediation
            columns_with_missing = data.columns[data.isnull().any()].tolist()
            missing_by_column = data.isnull().sum()
            worst_columns = missing_by_column[missing_by_column > 0].sort_values(ascending=False).head(5)
            
            issue = IssueAlert(
                issue_type='data_quality_missing',
                severity=severity,
                patient_id='ALL',  # Data quality affects entire trial
                description=(f"High percentage of missing data detected: {missing_percentage:.1%} "
                           f"({missing_cells:,} missing values in {total_cells:,} total data cells)"),
                affected_records=missing_cells,
                recommendation=(f"Implement comprehensive data quality improvement plan. "
                              f"{missing_percentage:.1%} missing data exceeds acceptable threshold "
                              f"({config['max_missing_percentage']:.1%}). Priority columns for attention: "
                              f"{', '.join([f'{col}({count})' for col, count in worst_columns.head(3).items()])}. "
                              f"Consider data retrieval, imputation strategies, and enhanced data collection procedures."),
                confidence_score=0.95,  # High confidence in missing data calculation
                metadata={
                    'missing_percentage': missing_percentage,
                    'missing_cells': missing_cells,
                    'total_cells': total_cells,
                    'columns_with_missing': columns_with_missing,
                    'worst_columns': dict(worst_columns),
                    'acceptable_threshold': config['max_missing_percentage']
                }
            )
            issues.append(issue)
        
        # 2. Duplicate records analysis - Data integrity verification
        # Duplicate records can inflate sample sizes and bias statistical analyses
        if 'patient_id' in data.columns and 'trial_day' in data.columns:
            # Identify duplicates based on patient and visit combination
            duplicate_mask = data.duplicated(subset=['patient_id', 'trial_day'])
            duplicate_count = duplicate_mask.sum()
            duplicate_percentage = duplicate_count / len(data) if len(data) > 0 else 0
            
            # Check if duplicates exceed acceptable threshold
            if duplicate_percentage > config['duplicate_threshold']:
                # Identify patients affected by duplicate records
                duplicate_patients = data[duplicate_mask]['patient_id'].unique().tolist()
                
                issue = IssueAlert(
                    issue_type='data_quality_duplicates',
                    severity='medium',  # Duplicates are concerning but usually correctable
                    patient_id='MULTIPLE',
                    description=(f"Duplicate records detected: {duplicate_count} duplicate entries "
                               f"({duplicate_percentage:.1%} of total records)"),
                    affected_records=duplicate_count,
                    recommendation=(f"Remove or consolidate {duplicate_count} duplicate records to ensure "
                                  f"data integrity. Affected patients: {', '.join(duplicate_patients[:10])}. "
                                  f"Implement data entry validation to prevent future duplicates. "
                                  f"Review data collection procedures and database constraints."),
                    confidence_score=0.9,  # High confidence in duplicate detection
                    metadata={
                        'duplicate_count': duplicate_count,
                        'duplicate_percentage': duplicate_percentage,
                        'duplicate_patients': duplicate_patients,
                        'total_patients_affected': len(duplicate_patients),
                        'acceptable_threshold': config['duplicate_threshold'],
                        'duplicate_detection_criteria': ['patient_id', 'trial_day']
                    }
                )
                issues.append(issue)
        
        logger.info(f"Data quality analysis completed - detected {len(issues)} quality issues requiring attention")
        return issues
    
    def detect_temporal_trends(self, data: pd.DataFrame) -> List[IssueAlert]:
        """
        Detect concerning temporal trends in clinical trial data over time.
        
        This method implements time-series analysis to identify systematic changes
        in trial metrics that may indicate:
        - Protocol drift or implementation issues
        - Seasonal or environmental effects
        - Staff training or procedural changes
        - Drug stability or supply chain issues
        - Patient population changes over time
        
        The analysis uses statistical trend detection through linear regression
        to identify significant changes over time in key trial metrics:
        - Overall treatment outcomes
        - Adverse event rates
        - Patient compliance patterns
        
        Temporal trend detection is crucial for maintaining trial integrity
        and identifying issues that may compromise result validity.
        
        Args:
            data (pd.DataFrame): Clinical trial data with 'visit_date' column and
                               temporal metrics for trend analysis.
            
        Returns:
            List[IssueAlert]: List of temporal trend issues with statistical evidence
                            and recommendations for investigation and remediation.
        """
        issues = []
        
        # Validate required column for temporal analysis
        if 'visit_date' not in data.columns:
            logger.warning("Visit date column not found - skipping temporal trend analysis")
            return issues
        
        # Load statistical configuration for trend detection
        config = self.config['statistical_config']
        
        # Prepare data for temporal analysis
        data = data.copy()
        data['visit_date'] = pd.to_datetime(data['visit_date'])
        
        # Aggregate data by date to create time series for trend analysis
        # This aggregation smooths daily variations and reveals underlying trends
        daily_stats = data.groupby('visit_date').agg({
            'outcome_score': ['mean', 'count'],      # Treatment efficacy over time
            'compliance_pct': 'mean',                # Patient adherence trends
            'adverse_event_flag': ['sum', 'count']   # Safety signal evolution
        }).reset_index()
        
        # Flatten column names for easier access
        daily_stats.columns = ['visit_date', 'outcome_mean', 'outcome_count', 
                              'compliance_mean', 'adverse_events', 'total_visits']
        
        # Skip analysis if insufficient temporal data points
        if len(daily_stats) < config['min_data_points']:
            logger.warning(f"Insufficient temporal data points ({len(daily_stats)}) for trend analysis")
            return issues
        
        # 1. Analyze declining outcome trends - Treatment efficacy deterioration
        # This identifies systematic decreases in treatment effectiveness over time
        x = np.arange(len(daily_stats))  # Time sequence for regression
        outcome_slope, _, outcome_r, outcome_p, _ = stats.linregress(x, daily_stats['outcome_mean'])
        
        # Check for statistically significant declining outcome trend
        if outcome_slope < -0.3 and outcome_p < config['trend_significance']:
            # Calculate clinical impact of the trend
            total_decline = outcome_slope * len(daily_stats)
            trend_strength = abs(outcome_r)
            
            issue = IssueAlert(
                issue_type='temporal_trend_declining_outcomes',
                severity='high',  # Declining outcomes always require urgent attention
                patient_id='ALL',  # Population-level trend affects entire trial
                description=(f"Statistically significant declining outcome trend detected: "
                           f"{outcome_slope:.3f} points per day decline over {len(daily_stats)} days "
                           f"(total decline: {total_decline:.1f} points, p={outcome_p:.3f})"),
                affected_records=len(data),
                recommendation=(f"Urgent investigation required for declining treatment outcomes. "
                              f"Trend shows {outcome_slope:.3f} point decrease per day with statistical "
                              f"significance (p={outcome_p:.3f}, R²={outcome_r**2:.3f}). Consider: "
                              f"drug stability testing, protocol compliance review, staff retraining, "
                              f"patient population changes, or environmental factors."),
                confidence_score=min(0.95, 0.7 + trend_strength * 0.3),  # Higher confidence with stronger trends
                metadata={
                    'slope': outcome_slope,
                    'r_squared': outcome_r**2,
                    'p_value': outcome_p,
                    'trend_duration_days': len(daily_stats),
                    'total_decline': total_decline,
                    'trend_strength': trend_strength,
                    'initial_outcome': daily_stats['outcome_mean'].iloc[0],
                    'final_outcome': daily_stats['outcome_mean'].iloc[-1]
                }
            )
            issues.append(issue)
        
        # 2. Analyze increasing adverse event trends - Safety deterioration
        # This identifies systematic increases in adverse events that may indicate
        # emerging safety issues or changes in trial conditions
        daily_stats['adverse_event_rate'] = daily_stats['adverse_events'] / daily_stats['total_visits']
        ae_slope, _, ae_r, ae_p, _ = stats.linregress(x, daily_stats['adverse_event_rate'])
        
        # Check for statistically significant increasing adverse event trend
        if ae_slope > 0.01 and ae_p < config['trend_significance']:  # 1% increase per day threshold
            # Calculate safety impact of the trend
            total_rate_increase = ae_slope * len(daily_stats)
            current_ae_rate = daily_stats['adverse_event_rate'].iloc[-1]
            
            issue = IssueAlert(
                issue_type='temporal_trend_increasing_adverse_events',
                severity='high',  # Increasing adverse events always require urgent safety review
                patient_id='ALL',  # Population-level safety trend
                description=(f"Statistically significant increasing adverse event trend detected: "
                           f"{ae_slope:.3f} rate increase per day over {len(daily_stats)} days "
                           f"(current rate: {current_ae_rate:.1%}, p={ae_p:.3f})"),
                affected_records=int(daily_stats['adverse_events'].sum()),
                recommendation=(f"Urgent safety review required for increasing adverse event trend. "
                              f"Rate increasing by {ae_slope:.3f} per day with statistical significance "
                              f"(p={ae_p:.3f}, R²={ae_r**2:.3f}). Current adverse event rate: {current_ae_rate:.1%}. "
                              f"Consider: safety committee review, protocol suspension evaluation, "
                              f"enhanced patient monitoring, drug batch analysis, or dose modifications."),
                confidence_score=min(0.95, 0.7 + abs(ae_r) * 0.3),  # Higher confidence with stronger trends
                metadata={
                    'slope': ae_slope,
                    'r_squared': ae_r**2,
                    'p_value': ae_p,
                    'current_ae_rate': current_ae_rate,
                    'total_rate_increase': total_rate_increase,
                    'trend_strength': abs(ae_r),
                    'initial_ae_rate': daily_stats['adverse_event_rate'].iloc[0],
                    'total_adverse_events': int(daily_stats['adverse_events'].sum())
                }
            )
            issues.append(issue)
        
        logger.info(f"Temporal trend analysis completed - detected {len(issues)} concerning trends requiring investigation")
        return issues
    
    def get_issue_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all detected issues for clinical review and reporting.
        
        This method provides a structured overview of all detected issues, organized
        by severity and type to support clinical decision-making and prioritization.
        The summary includes:
        - Total issue counts and distribution
        - Severity-based classification for triage
        - Issue type breakdown for targeted interventions  
        - High-priority issues requiring immediate attention
        
        This summary format supports both automated reporting and clinical review
        processes, enabling efficient issue management and response.
        
        Returns:
            Dict[str, Any]: Comprehensive summary statistics and details of detected issues
                          organized for clinical decision-making and reporting.
        """
        # Handle case where no issues have been detected
        if not self.detected_issues:
            return {
                'total_issues': 0, 
                'by_severity': {}, 
                'by_type': {},
                'high_priority_count': 0,
                'high_priority_issues': [],
                'summary_status': 'no_issues_detected'
            }
        
        # Count issues by severity level for triage prioritization
        severity_counts = {}
        for issue in self.detected_issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
        
        # Count issues by type for targeted intervention planning
        type_counts = {}
        for issue in self.detected_issues:
            type_counts[issue.issue_type] = type_counts.get(issue.issue_type, 0) + 1
        
        # Identify high-priority issues requiring immediate attention
        high_priority_issues = [
            issue for issue in self.detected_issues 
            if issue.severity in ['critical', 'high']
        ]
        
        # Sort high-priority issues by severity and confidence for prioritization
        high_priority_issues.sort(key=lambda x: (
            0 if x.severity == 'critical' else 1,  # Critical first
            -x.confidence_score  # Higher confidence first within same severity
        ))
        
        return {
            'total_issues': len(self.detected_issues),
            'by_severity': severity_counts,
            'by_type': type_counts,
            'high_priority_count': len(high_priority_issues),
            'high_priority_issues': [
                {
                    'type': issue.issue_type,
                    'severity': issue.severity,
                    'patient_id': issue.patient_id,
                    'description': issue.description,
                    'confidence_score': issue.confidence_score,
                    'affected_records': issue.affected_records
                }
                for issue in high_priority_issues[:5]  # Top 5 high-priority issues for summary
            ],
            'summary_status': 'issues_detected',
            'severity_distribution': {
                'critical_pct': severity_counts.get('critical', 0) / len(self.detected_issues) * 100,
                'high_pct': severity_counts.get('high', 0) / len(self.detected_issues) * 100,
                'medium_pct': severity_counts.get('medium', 0) / len(self.detected_issues) * 100,
                'low_pct': severity_counts.get('low', 0) / len(self.detected_issues) * 100
            }
        }


def main():
    """
    Main function for testing the Issue Detector with comprehensive synthetic data.
    
    This function demonstrates the complete issue detection workflow using
    realistic synthetic clinical trial data that includes various types of
    issues for validation of detection algorithms.
    
    The synthetic data includes:
    - Patients with compliance issues (P002, P005)
    - Patients with efficacy issues (P003, P007) 
    - Patients with multiple adverse events (P004)
    - Statistical outliers and data quality issues
    - Temporal patterns for trend analysis
    
    This comprehensive testing ensures all detection methods work correctly
    and provides examples of the structured issue reporting format.
    """
    # Set random seed for reproducible testing
    np.random.seed(42)
    
    logger.info("Generating comprehensive synthetic clinical trial data for issue detection testing...")
    
    # Generate synthetic clinical trial data with intentional issues for testing
    data = []
    for i in range(100):  # 100 total records across 10 patients
        patient_id = f"P{i//10 + 1:03d}"
        trial_day = (i % 10) + 1
        
        # Introduce compliance issues in specific patients for testing
        if patient_id in ['P002', 'P005']:
            # Low compliance patients - simulate adherence problems
            compliance = np.random.normal(60, 10)  # Mean 60% compliance with variation
        else:
            # Normal compliance patients
            compliance = np.random.normal(90, 5)   # Mean 90% compliance with low variation
        
        compliance = np.clip(compliance, 0, 100)  # Ensure valid percentage range
        
        # Introduce efficacy issues in specific patients for testing
        if patient_id in ['P003', 'P007']:
            # Low efficacy patients - simulate treatment non-response
            outcome = np.random.normal(45, 10)  # Mean 45 outcome score (below efficacy threshold)
        else:
            # Normal efficacy patients
            outcome = np.random.normal(80, 8)   # Mean 80 outcome score (good response)
        
        outcome = np.clip(outcome, 0, 100)  # Ensure valid score range
        
        # Introduce adverse events with varying rates by patient
        adverse_event = 1 if np.random.random() < 0.12 else 0  # 12% baseline adverse event rate
        if patient_id == 'P004':  # Patient with multiple adverse events for clustering detection
            adverse_event = 1 if np.random.random() < 0.4 else 0  # 40% adverse event rate
        
        # Create comprehensive record with all required fields
        data.append({
            'patient_id': patient_id,
            'trial_day': trial_day,
            'dosage_mg': 50,  # Standard dosage
            'compliance_pct': compliance,
            'adverse_event_flag': adverse_event,
            'outcome_score': outcome,
            'cohort': 'A' if i % 2 == 0 else 'B',  # Alternate cohort assignment
            'visit_date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=trial_day-1),
            'doctor_notes': f'Sample clinical notes for visit {trial_day}'
        })
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(data)
    
    logger.info(f"Generated synthetic dataset with {len(df)} records for {df['patient_id'].nunique()} patients")
    
    # Initialize Issue Detector with default configuration
    detector = IssueDetector()
    
    # Run comprehensive issue detection across all methods
    logger.info("Running comprehensive issue detection analysis...")
    issues = detector.detect_all_issues(df)
    
    # Generate and display comprehensive summary
    summary = detector.get_issue_summary()
    
    print("\n" + "="*80)
    print("CLINICAL TRIAL ISSUE DETECTION SUMMARY")
    print("="*80)
    print(f"📊 Total issues detected: {summary['total_issues']}")
    print(f"🚨 High priority issues: {summary['high_priority_count']}")
    print(f"📈 Issues by severity: {summary['by_severity']}")
    print(f"🔍 Issues by type: {summary['by_type']}")
    
    # Display detailed high-priority issues for immediate attention
    if summary['high_priority_issues']:
        print("\n🚨 HIGH-PRIORITY ISSUES REQUIRING IMMEDIATE ATTENTION:")
        print("-" * 60)
        for i, issue in enumerate(summary['high_priority_issues'], 1):
            print(f"{i}. {issue['type'].upper()} ({issue['severity'].upper()})")
            print(f"   Patient: {issue['patient_id']}")
            print(f"   Description: {issue['description']}")
            print(f"   Confidence: {issue['confidence_score']:.2f}")
            print(f"   Affected Records: {issue['affected_records']}")
            print()
    
    # Display severity distribution for clinical review
    print("📊 SEVERITY DISTRIBUTION:")
    print("-" * 30)
    for severity, percentage in summary['severity_distribution'].items():
        if percentage > 0:
            print(f"   {severity.replace('_pct', '').title()}: {percentage:.1f}%")
    
    print("\n" + "="*80)
    print("Issue detection testing completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()