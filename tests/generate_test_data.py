#!/usr/bin/env python3
"""
Generate Comprehensive Test Data for Clinical Insights Assistant

This script creates a realistic clinical trial dataset with various patterns,
edge cases, and scenarios to test all functionality of the platform.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

def generate_comprehensive_clinical_data():
    """Generate comprehensive clinical trial dataset for testing all features."""
    
    # Configuration
    num_patients = 200
    num_cohorts = 4
    visits_per_patient = 6
    start_date = datetime(2024, 1, 1)
    
    # Cohort configurations
    cohorts = {
        'Treatment_A': {'dosage': 50, 'patients': 60, 'efficacy_boost': 0.15},
        'Treatment_B': {'dosage': 25, 'patients': 60, 'efficacy_boost': 0.08}, 
        'Treatment_C': {'dosage': 75, 'patients': 50, 'efficacy_boost': 0.22},
        'Control': {'dosage': 0, 'patients': 30, 'efficacy_boost': 0.0}
    }
    
    # Generate patient demographics
    age_groups = ['18-30', '31-45', '46-60', '60+']
    genders = ['M', 'F', 'Other']
    ethnicities = ['Caucasian', 'Hispanic', 'African American', 'Asian', 'Other']
    
    data = []
    patient_counter = 1
    
    for cohort_name, cohort_config in cohorts.items():
        for patient_idx in range(cohort_config['patients']):
            patient_id = f"PATIENT_{patient_counter:03d}"
            
            # Patient demographics (consistent per patient)
            age = random.randint(18, 80)
            age_group = '18-30' if age <= 30 else '31-45' if age <= 45 else '46-60' if age <= 60 else '60+'
            gender = random.choice(genders)
            ethnicity = random.choice(ethnicities)
            weight = random.normalvariate(70, 15) if gender == 'F' else random.normalvariate(85, 18)
            weight = max(45, min(150, weight))  # Realistic bounds
            
            # Baseline characteristics (affect outcomes)
            baseline_severity = random.normalvariate(50, 15)
            baseline_severity = max(10, min(90, baseline_severity))
            
            # Patient-specific factors
            compliance_tendency = random.normalvariate(90, 10)  # Individual compliance pattern
            response_rate = random.normalvariate(1.0, 0.3)  # Individual response to treatment
            adverse_event_risk = random.uniform(0.05, 0.25)  # Individual AE risk
            
            # Generate visits for this patient
            for visit_num in range(1, visits_per_patient + 1):
                visit_date = start_date + timedelta(days=(visit_num-1) * 30 + random.randint(-7, 7))
                
                # Compliance decreases over time with individual variation
                time_factor = 1 - (visit_num - 1) * 0.02  # 2% decrease per visit
                compliance = compliance_tendency * time_factor + random.normalvariate(0, 5)
                compliance = max(30, min(100, compliance))
                
                # Outcome score progression
                base_outcome = 60 + (baseline_severity - 50) * 0.3  # Baseline affects outcomes
                
                # Treatment effect (cumulative over time)
                treatment_effect = 0
                if cohort_name != 'Control':
                    treatment_effect = cohort_config['efficacy_boost'] * visit_num * 5 * response_rate
                
                # Compliance effect
                compliance_effect = (compliance - 80) * 0.1
                
                # Time trend (natural progression)
                time_effect = visit_num * 2
                
                # Random variation
                random_effect = random.normalvariate(0, 8)
                
                outcome_score = base_outcome + treatment_effect + compliance_effect + time_effect + random_effect
                outcome_score = max(20, min(100, outcome_score))
                
                # Adverse events (higher risk with higher dosage and lower compliance)
                ae_risk = adverse_event_risk
                if cohort_config['dosage'] > 0:
                    ae_risk += cohort_config['dosage'] * 0.002  # Dose-related risk
                if compliance < 80:
                    ae_risk += 0.05  # Poor compliance increases risk
                
                adverse_event = 1 if random.random() < ae_risk else 0
                
                # Biomarkers (correlated with treatment and outcomes)
                biomarker_a = 10 + treatment_effect * 0.05 + random.normalvariate(0, 2)
                biomarker_b = 8 + (outcome_score - 60) * 0.03 + random.normalvariate(0, 1.5)
                
                # Additional biomarkers for more complex analysis
                biomarker_c = baseline_severity * 0.2 + random.normalvariate(0, 3)
                inflammatory_marker = max(0, 5 - treatment_effect * 0.08 + random.normalvariate(0, 2))
                
                # Vital signs
                systolic_bp = 120 + age * 0.3 + random.normalvariate(0, 15)
                diastolic_bp = 80 + age * 0.2 + random.normalvariate(0, 10)
                heart_rate = 70 + random.normalvariate(0, 12)
                temperature = 36.5 + random.normalvariate(0, 0.5)
                
                # Laboratory values
                hemoglobin = 14 + (1 if gender == 'M' else -1) + random.normalvariate(0, 1.5)
                white_blood_cells = 7000 + random.normalvariate(0, 2000)
                platelet_count = 250000 + random.normalvariate(0, 50000)
                
                # Quality of life scores
                qol_physical = outcome_score * 0.8 + random.normalvariate(0, 10)
                qol_mental = outcome_score * 0.7 + random.normalvariate(0, 12)
                qol_social = outcome_score * 0.6 + random.normalvariate(0, 15)
                
                # Doctor notes categories
                notes_categories = ['Stable', 'Improved', 'Declined', 'Adverse Event', 'Dose Adjustment']
                note_weights = [0.4, 0.3, 0.1, 0.1 if adverse_event else 0.05, 0.1]
                doctor_notes_category = random.choices(notes_categories, weights=note_weights)[0]
                
                # Concomitant medications (more common in older patients)
                concomitant_meds = random.randint(0, min(8, age // 10))
                
                data.append({
                    'patient_id': patient_id,
                    'cohort': cohort_name,
                    'visit_number': visit_num,
                    'visit_date': visit_date.strftime('%Y-%m-%d'),
                    'dosage_mg': cohort_config['dosage'],
                    'compliance_pct': round(compliance, 1),
                    'adverse_event_flag': adverse_event,
                    'outcome_score': round(outcome_score, 1),
                    'biomarker_a': round(biomarker_a, 2),
                    'biomarker_b': round(biomarker_b, 2),
                    'biomarker_c': round(biomarker_c, 2),
                    'inflammatory_marker': round(inflammatory_marker, 2),
                    'age': age,
                    'age_group': age_group,
                    'gender': gender,
                    'ethnicity': ethnicity,
                    'weight_kg': round(weight, 1),
                    'baseline_severity': round(baseline_severity, 1),
                    'systolic_bp': round(systolic_bp, 0),
                    'diastolic_bp': round(diastolic_bp, 0),
                    'heart_rate': round(heart_rate, 0),
                    'temperature_c': round(temperature, 1),
                    'hemoglobin': round(hemoglobin, 1),
                    'wbc_count': round(white_blood_cells, 0),
                    'platelet_count': round(platelet_count, 0),
                    'qol_physical': round(qol_physical, 1),
                    'qol_mental': round(qol_mental, 1),
                    'qol_social': round(qol_social, 1),
                    'doctor_notes_category': doctor_notes_category,
                    'concomitant_medications': concomitant_meds
                })
            
            patient_counter += 1
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing data patterns (realistic)
    missing_indices = random.sample(range(len(df)), k=int(len(df) * 0.02))
    for idx in missing_indices:
        # Randomly make some biomarkers missing
        missing_cols = random.sample(['biomarker_c', 'inflammatory_marker', 'qol_mental', 'qol_social'], 
                                   k=random.randint(1, 2))
        for col in missing_cols:
            df.at[idx, col] = np.nan
    
    # Add some protocol deviations (higher in later visits)
    protocol_deviation_indices = []
    for idx, row in df.iterrows():
        if row['visit_number'] > 3 and random.random() < 0.05:  # 5% chance for later visits
            protocol_deviation_indices.append(idx)
    
    df['protocol_deviation'] = 0
    df.loc[protocol_deviation_indices, 'protocol_deviation'] = 1
    
    # Add study sites for multi-site analysis
    sites = ['Site_001_NYC', 'Site_002_LA', 'Site_003_Chicago', 'Site_004_Miami', 'Site_005_Boston']
    df['study_site'] = [random.choice(sites) for _ in range(len(df))]
    
    # Add discontinuation flags for dropout analysis
    df['discontinued'] = 0
    # Some patients discontinue (more likely with adverse events or poor outcomes)
    for patient in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == patient].sort_values('visit_number')
        if len(patient_data) >= 3:  # Only consider if they have at least 3 visits
            last_outcome = patient_data.iloc[-1]['outcome_score']
            last_compliance = patient_data.iloc[-1]['compliance_pct']
            ae_count = patient_data['adverse_event_flag'].sum()
            
            # Calculate discontinuation probability
            disc_prob = 0.02  # Base probability
            if last_outcome < 50:
                disc_prob += 0.1
            if last_compliance < 70:
                disc_prob += 0.08
            if ae_count >= 2:
                disc_prob += 0.15
            
            if random.random() < disc_prob:
                # Mark as discontinued from last visit onwards
                last_visit = patient_data.iloc[-1]['visit_number']
                df.loc[(df['patient_id'] == patient) & (df['visit_number'] == last_visit), 'discontinued'] = 1
    
    return df

def main():
    """Generate and save comprehensive test data."""
    print("🧪 Generating comprehensive clinical trial test data...")
    
    # Generate the data
    df = generate_comprehensive_clinical_data()
    
    # Create output directory (use data/ folder instead of sample_data/)
    output_dir = Path('../data')
    output_dir.mkdir(exist_ok=True)
    
    # Save comprehensive dataset
    comprehensive_file = output_dir / 'comprehensive_clinical_trial_data.csv'
    df.to_csv(comprehensive_file, index=False)
    
    # Create smaller focused datasets for specific testing
    
    # 1. High adverse event dataset
    ae_df = df[df['adverse_event_flag'] == 1].head(50)
    ae_file = output_dir / 'high_adverse_events_data.csv'
    ae_df.to_csv(ae_file, index=False)
    
    # 2. Poor compliance dataset  
    poor_compliance_df = df[df['compliance_pct'] < 70].head(50)
    compliance_file = output_dir / 'poor_compliance_data.csv'
    poor_compliance_df.to_csv(compliance_file, index=False)
    
    # 3. Longitudinal dataset (all visits for subset of patients)
    selected_patients = df['patient_id'].unique()[:20]
    longitudinal_df = df[df['patient_id'].isin(selected_patients)]
    longitudinal_file = output_dir / 'longitudinal_analysis_data.csv'
    longitudinal_df.to_csv(longitudinal_file, index=False)
    
    # Print summary
    print(f"✅ Generated comprehensive dataset: {len(df)} records, {df['patient_id'].nunique()} patients")
    print(f"📊 Cohort distribution:")
    print(df['cohort'].value_counts())
    print(f"⚠️  Adverse events: {df['adverse_event_flag'].sum()} ({df['adverse_event_flag'].mean()*100:.1f}%)")
    print(f"📈 Average outcome score: {df['outcome_score'].mean():.1f}")
    print(f"💊 Average compliance: {df['compliance_pct'].mean():.1f}%")
    print(f"🏥 Study sites: {df['study_site'].nunique()}")
    print(f"❌ Discontinuations: {df['discontinued'].sum()}")
    
    print(f"\n📁 Files created in data/ folder:")
    print(f"   • {comprehensive_file}")
    print(f"   • {ae_file}")
    print(f"   • {compliance_file}")
    print(f"   • {longitudinal_file}")
    
    print(f"\n🎯 Dataset features for testing:")
    print(f"   • Multiple cohorts with different dosages")
    print(f"   • Realistic patient demographics and characteristics")
    print(f"   • Time-based progression and compliance patterns")
    print(f"   • Correlated biomarkers and outcome measures")
    print(f"   • Protocol deviations and study discontinuations")
    print(f"   • Multi-site data for site analysis")
    print(f"   • Missing data patterns for robustness testing")
    print(f"   • Quality of life measures")
    print(f"   • Vital signs and laboratory values")

if __name__ == "__main__":
    main()