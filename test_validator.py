#!/usr/bin/env python3
"""Quick test script to verify the DataValidator implementation."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from virtual_ward_omop_generator.validation.validator import DataValidator

def create_test_data():
    """Create minimal test data for validation."""
    
    # Create person data
    person_data = pd.DataFrame({
        'person_id': [1, 2, 3],
        'gender_concept_id': [8507, 8532, 8507],  # Male, Female, Male
        'year_of_birth': [1980, 1975, 1990],
        'month_of_birth': [5, 8, 12],
        'day_of_birth': [15, 22, 3],
        'race_concept_id': [8527, 8527, 8527],  # White
        'ethnicity_concept_id': [38003564, 38003564, 38003564]  # Not Hispanic
    })
    
    # Create observation periods
    obs_period_data = pd.DataFrame({
        'observation_period_id': [1, 2, 3],
        'person_id': [1, 2, 3],
        'observation_period_start_date': ['2024-01-01', '2024-01-01', '2024-01-01'],
        'observation_period_end_date': ['2024-12-31', '2024-12-31', '2024-12-31'],
        'period_type_concept_id': [2100000014, 2100000014, 2100000014]
    })
    
    # Create episodes
    episode_data = pd.DataFrame({
        'episode_id': [1, 2, 3],
        'person_id': [1, 2, 3],
        'episode_concept_id': [2100000001, 2100000001, 2100000001],
        'episode_start_datetime': ['2024-06-01 08:00:00', '2024-07-01 09:00:00', '2024-08-01 10:00:00'],
        'episode_end_datetime': ['2024-06-15 18:00:00', '2024-07-20 17:00:00', '2024-08-10 16:00:00'],
        'episode_parent_id': [None, None, None],
        'episode_number': [1, 1, 1],
        'episode_object_concept_id': [2100000001, 2100000001, 2100000001],
        'episode_type_concept_id': [2100000001, 2100000001, 2100000001],
        'episode_source_value': ['VW_EP_001', 'VW_EP_002', 'VW_EP_003'],
        'episode_source_concept_id': [2100000001, 2100000001, 2100000001]
    })
    
    # Create measurements with some valid and invalid values
    measurement_data = pd.DataFrame({
        'measurement_id': [1, 2, 3, 4, 5],
        'person_id': [1, 1, 2, 2, 3],
        'measurement_concept_id': [3027018, 3004249, 3027018, 3016502, 3025315],  # HR, SBP, HR, O2Sat, Weight
        'measurement_date': ['2024-06-02', '2024-06-02', '2024-07-02', '2024-07-02', '2024-08-02'],
        'measurement_datetime': ['2024-06-02 10:00:00', '2024-06-02 10:05:00', '2024-07-02 11:00:00', '2024-07-02 11:05:00', '2024-08-02 12:00:00'],
        'measurement_type_concept_id': [2100000012, 2100000012, 2100000012, 2100000012, 2100000012],
        'value_as_number': [72.0, 120.0, 85.0, 98.0, 70.0],  # Normal values
        'value_as_concept_id': [None, None, None, None, None],
        'unit_concept_id': [2100000031, 2100000032, 2100000031, 2100000030, 8739],  # BPM, mmHg, BPM, %, kg
        'visit_occurrence_id': [1, 1, 2, 2, 3]
    })
    
    # Create visit occurrences
    visit_data = pd.DataFrame({
        'visit_occurrence_id': [1, 2, 3],
        'person_id': [1, 2, 3],
        'visit_concept_id': [2100000002, 2100000002, 2100000002],  # Telehealth visit
        'visit_start_date': ['2024-06-02', '2024-07-02', '2024-08-02'],
        'visit_start_datetime': ['2024-06-02 10:00:00', '2024-07-02 11:00:00', '2024-08-02 12:00:00'],
        'visit_end_date': ['2024-06-02', '2024-07-02', '2024-08-02'],
        'visit_end_datetime': ['2024-06-02 10:30:00', '2024-07-02 11:30:00', '2024-08-02 12:30:00'],
        'visit_type_concept_id': [2100000002, 2100000002, 2100000002]
    })
    
    return {
        'person': person_data,
        'observation_period': obs_period_data,
        'episode': episode_data,
        'measurement': measurement_data,
        'visit_occurrence': visit_data
    }

def create_concept_dictionary():
    """Create a minimal concept dictionary for testing."""
    return pd.DataFrame({
        'concept_id': [8507, 8532, 8527, 38003564, 2100000001, 2100000002, 2100000012, 2100000014,
                      3027018, 3004249, 3016502, 3025315, 2100000031, 2100000032, 2100000030, 8739],
        'concept_name': ['Male', 'Female', 'White', 'Not Hispanic or Latino', 'Virtual Ward Episode',
                        'Telehealth Visit', 'Patient Device Measurement', 'Observation Period—Virtual Ward',
                        'Heart rate', 'Systolic blood pressure', 'Oxygen saturation', 'Body weight',
                        'beats per minute', 'mmHg', 'percent', 'kilogram'],
        'concept_code': ['M', 'F', 'W', 'NH', 'VW_EP', 'TH_VISIT', 'PDM', 'OP_VW',
                        'HR', 'SBP', 'O2SAT', 'WEIGHT', 'BPM', 'MMHG', 'PCT', 'KG']
    })

def main():
    """Run validation test."""
    print("Testing DataValidator implementation...")
    
    # Create test configuration
    config = {
        'validation': {
            'strict_mode': True,
            'clinical_bounds_check': True
        }
    }
    
    # Create test data
    test_data = create_test_data()
    concept_dict = create_concept_dictionary()
    
    # Initialize validator
    validator = DataValidator(config, concept_dict)
    
    # Run validation
    print("\nRunning validation...")
    is_valid = validator.validate(test_data)
    
    # Get validation report
    report = validator.get_validation_report()
    
    # Print results
    print(f"\nValidation Result: {'PASSED' if is_valid else 'FAILED'}")
    print(f"Errors: {report['summary']['total_errors']}")
    print(f"Warnings: {report['summary']['total_warnings']}")
    
    if report['errors']:
        print("\nErrors:")
        for error in report['errors']:
            print(f"  - {error}")
    
    if report['warnings']:
        print("\nWarnings:")
        for warning in report['warnings']:
            print(f"  - {warning}")
    
    if 'coverage_report' in report['details']:
        coverage = report['details']['coverage_report']
        print(f"\nCoverage Report:")
        if 'data_completeness' in coverage:
            print("Data Completeness:")
            for table, completeness in coverage['data_completeness'].items():
                print(f"  {table}: {completeness:.1%}")
    
    print("\nValidation test completed!")

if __name__ == "__main__":
    main()