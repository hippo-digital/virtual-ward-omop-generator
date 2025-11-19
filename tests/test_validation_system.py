#!/usr/bin/env python3
"""Unit tests for the DataValidator validation system."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from virtual_ward_omop_generator.validation.validator import DataValidator
from virtual_ward_omop_generator.models.omop import OMOPTables


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'validation': {
                'strict_mode': True,
                'clinical_bounds_check': True
            }
        }
        
        # Create concept dictionary
        self.concept_dict = pd.DataFrame({
            'concept_id': [8507, 8532, 8527, 38003564, 2100000001, 2100000002, 
                          2100000012, 2100000014, 3027018, 3004249, 3016502, 
                          3025315, 2100000031, 2100000032, 2100000030, 8739],
            'concept_name': ['Male', 'Female', 'White', 'Not Hispanic or Latino', 
                           'Virtual Ward Episode', 'Telehealth Visit', 'Patient Device Measurement',
                           'Observation Period—Virtual Ward', 'Heart rate', 'Systolic blood pressure',
                           'Oxygen saturation', 'Body weight', 'beats per minute', 'mmHg', 
                           'percent', 'kilogram'],
            'concept_code': ['M', 'F', 'W', 'NH', 'VW_EP', 'TH_VISIT', 'PDM', 'OP_VW',
                           'HR', 'SBP', 'O2SAT', 'WEIGHT', 'BPM', 'MMHG', 'PCT', 'KG']
        })
        
        self.validator = DataValidator(self.config, self.concept_dict)
    
    def create_valid_test_data(self):
        """Create valid test data for validation."""
        person_data = pd.DataFrame({
            'person_id': [1, 2, 3],
            'gender_concept_id': [8507, 8532, 8507],
            'year_of_birth': [1980, 1975, 1990],
            'month_of_birth': [5, 8, 12],
            'day_of_birth': [15, 22, 3],
            'race_concept_id': [8527, 8527, 8527],
            'ethnicity_concept_id': [38003564, 38003564, 38003564]
        })
        
        obs_period_data = pd.DataFrame({
            'observation_period_id': [1, 2, 3],
            'person_id': [1, 2, 3],
            'observation_period_start_date': ['2024-01-01', '2024-01-01', '2024-01-01'],
            'observation_period_end_date': ['2024-12-31', '2024-12-31', '2024-12-31'],
            'period_type_concept_id': [2100000014, 2100000014, 2100000014]
        })
        
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
        
        measurement_data = pd.DataFrame({
            'measurement_id': [1, 2, 3, 4, 5],
            'person_id': [1, 1, 2, 2, 3],
            'measurement_concept_id': [3027018, 3004249, 3027018, 3016502, 3025315],
            'measurement_date': ['2024-06-02', '2024-06-02', '2024-07-02', '2024-07-02', '2024-08-02'],
            'measurement_datetime': ['2024-06-02 10:00:00', '2024-06-02 10:05:00', 
                                   '2024-07-02 11:00:00', '2024-07-02 11:05:00', '2024-08-02 12:00:00'],
            'measurement_type_concept_id': [2100000012, 2100000012, 2100000012, 2100000012, 2100000012],
            'value_as_number': [72.0, 120.0, 85.0, 98.0, 70.0],
            'value_as_concept_id': [None, None, None, None, None],
            'unit_concept_id': [2100000031, 2100000032, 2100000031, 2100000030, 8739]
        })
        
        return {
            'person': person_data,
            'observation_period': obs_period_data,
            'episode': episode_data,
            'measurement': measurement_data
        }


class TestForeignKeyValidation(TestDataValidator):
    """Test foreign key validation functionality."""
    
    def test_valid_foreign_keys(self):
        """Test validation passes with valid foreign keys."""
        data = self.create_valid_test_data()
        
        # Clear any previous results
        self.validator.clear_results()
        
        # Run FK validation
        self.validator._validate_foreign_keys(data)
        
        # Should have no errors for valid data
        self.assertFalse(self.validator.has_errors)
        self.assertEqual(len(self.validator.validation_errors), 0)
    
    def test_orphaned_records_detection(self):
        """Test detection of orphaned records in child tables."""
        data = self.create_valid_test_data()
        
        # Add orphaned record in episode table
        orphaned_episode = pd.DataFrame({
            'episode_id': [4],
            'person_id': [999],  # Non-existent person_id
            'episode_concept_id': [2100000001],
            'episode_start_datetime': ['2024-09-01 08:00:00'],
            'episode_end_datetime': ['2024-09-15 18:00:00'],
            'episode_parent_id': [None],
            'episode_number': [1],
            'episode_object_concept_id': [2100000001],
            'episode_type_concept_id': [2100000001],
            'episode_source_value': ['VW_EP_004'],
            'episode_source_concept_id': [2100000001]
        })
        
        data['episode'] = pd.concat([data['episode'], orphaned_episode], ignore_index=True)
        
        # Clear any previous results
        self.validator.clear_results()
        
        # Run FK validation
        self.validator._validate_foreign_keys(data)
        
        # Should detect orphaned record
        self.assertTrue(self.validator.has_errors)
        self.assertGreater(len(self.validator.validation_errors), 0)
        
        # Check error message contains orphaned information
        error_messages = ' '.join(self.validator.validation_errors)
        self.assertIn('orphaned', error_messages.lower())
    
    def test_missing_referenced_table(self):
        """Test handling of missing referenced tables."""
        data = self.create_valid_test_data()
        
        # Remove person table to create missing reference
        del data['person']
        
        # Clear any previous results
        self.validator.clear_results()
        
        # Run FK validation
        self.validator._validate_foreign_keys(data)
        
        # Should detect missing referenced table
        self.assertTrue(self.validator.has_errors)
        error_messages = ' '.join(self.validator.validation_errors)
        self.assertIn('missing', error_messages.lower())
    
    def test_invalid_concept_references(self):
        """Test detection of invalid concept ID references."""
        data = self.create_valid_test_data()
        
        # Add invalid concept ID
        data['person'].loc[0, 'gender_concept_id'] = 999999  # Invalid concept ID
        
        # Clear any previous results
        self.validator.clear_results()
        
        # Run FK validation
        self.validator._validate_foreign_keys(data)
        
        # Should detect invalid concept reference
        self.assertTrue(self.validator.has_errors)
        error_messages = ' '.join(self.validator.validation_errors)
        self.assertIn('invalid concept', error_messages.lower())
    
    def test_find_orphaned_records_method(self):
        """Test the _find_orphaned_records helper method."""
        # Create parent and child dataframes
        parent_df = pd.DataFrame({'parent_id': [1, 2, 3]})
        child_df = pd.DataFrame({'child_id': [1, 2, 3, 4], 'parent_id': [1, 2, 3, 999]})
        
        # Find orphaned records
        orphaned = self.validator._find_orphaned_records(
            child_df, 'parent_id', parent_df, 'parent_id'
        )
        
        # Should find one orphaned record
        self.assertEqual(len(orphaned), 1)
        self.assertEqual(orphaned.iloc[0], 999)
    
    def test_null_values_in_foreign_keys(self):
        """Test handling of null values in foreign key validation."""
        data = self.create_valid_test_data()
        
        # Add null foreign key value
        data['episode'].loc[0, 'person_id'] = None
        
        # Clear any previous results
        self.validator.clear_results()
        
        # Run FK validation
        self.validator._validate_foreign_keys(data)
        
        # Should not report null values as orphaned
        self.assertFalse(self.validator.has_errors)


class TestTemporalIntegrityValidation(TestDataValidator):
    """Test temporal integrity validation functionality."""
    
    def test_valid_temporal_integrity(self):
        """Test validation passes with valid temporal relationships."""
        data = self.create_valid_test_data()
        
        # Clear any previous results
        self.validator.clear_results()
        
        # Run temporal validation
        self.validator._validate_temporal_integrity(data)
        
        # Should have no errors for valid temporal data
        self.assertFalse(self.validator.has_errors)
    
    def test_invalid_datetime_ranges(self):
        """Test detection of invalid start > end datetime ranges."""
        data = self.create_valid_test_data()
        
        # Create invalid datetime range (start > end)
        data['episode'].loc[0, 'episode_start_datetime'] = '2024-06-20 08:00:00'
        data['episode'].loc[0, 'episode_end_datetime'] = '2024-06-15 18:00:00'
        
        # Clear any previous results
        self.validator.clear_results()
        
        # Run temporal validation
        self.validator._validate_temporal_integrity(data)
        
        # Should detect invalid datetime range
        self.assertTrue(self.validator.has_errors)
        error_messages = ' '.join(self.validator.validation_errors)
        self.assertIn('start > end', error_messages.lower())
    
    def test_observation_period_coverage(self):
        """Test validation of observation period coverage."""
        data = self.create_valid_test_data()
        
        # Create episode outside observation period
        data['episode'].loc[0, 'episode_start_datetime'] = '2023-12-01 08:00:00'
        data['episode'].loc[0, 'episode_end_datetime'] = '2023-12-15 18:00:00'
        
        # Clear any previous results
        self.validator.clear_results()
        
        # Run temporal validation
        self.validator._validate_temporal_integrity(data)
        
        # Should detect episode outside observation period
        self.assertTrue(self.validator.has_errors)
        error_messages = ' '.join(self.validator.validation_errors)
        self.assertIn('outside observation periods', error_messages.lower())
    
    def test_events_within_episode_boundaries(self):
        """Test validation that events fall within episode boundaries."""
        data = self.create_valid_test_data()
        
        # Add measurement outside episode boundary
        outside_measurement = pd.DataFrame({
            'measurement_id': [6],
            'person_id': [1],
            'measurement_concept_id': [3027018],
            'measurement_date': ['2024-05-01'],  # Before episode starts
            'measurement_datetime': ['2024-05-01 10:00:00'],
            'measurement_type_concept_id': [2100000012],
            'value_as_number': [75.0],
            'value_as_concept_id': [None],
            'unit_concept_id': [2100000031]
        })
        
        data['measurement'] = pd.concat([data['measurement'], outside_measurement], ignore_index=True)
        
        # Clear any previous results
        self.validator.clear_results()
        
        # Run temporal validation
        self.validator._validate_temporal_integrity(data)
        
        # Should detect event outside episode boundaries
        self.assertTrue(self.validator.has_warnings)  # This creates warnings, not errors
        warning_messages = ' '.join(self.validator.validation_warnings)
        self.assertIn('outside episode boundaries', warning_messages.lower())
    
    def test_missing_observation_periods(self):
        """Test handling of persons with episodes but no observation periods."""
        data = self.create_valid_test_data()
        
        # Remove observation period for person 1
        data['observation_period'] = data['observation_period'][
            data['observation_period']['person_id'] != 1
        ]
        
        # Clear any previous results
        self.validator.clear_results()
        
        # Run temporal validation
        self.validator._validate_temporal_integrity(data)
        
        # Should detect missing observation period
        self.assertTrue(self.validator.has_errors)
        error_messages = ' '.join(self.validator.validation_errors)
        self.assertIn('no observation periods', error_messages.lower())


class TestClinicalBoundsValidation(TestDataValidator):
    """Test clinical bounds and measurement validation functionality."""
    
    def test_valid_clinical_bounds(self):
        """Test validation passes with values within clinical bounds."""
        data = self.create_valid_test_data()
        
        # Clear any previous results
        self.validator.clear_results()
        
        # Run clinical bounds validation
        self.validator._validate_clinical_bounds(data)
        
        # Should have no warnings for valid clinical values
        self.assertFalse(self.validator.has_warnings)
    
    def test_values_outside_clinical_bounds(self):
        """Test detection of values outside clinical bounds."""
        data = self.create_valid_test_data()
        
        # Add measurement with value outside clinical bounds
        data['measurement'].loc[0, 'value_as_number'] = 300.0  # Extremely high heart rate
        
        # Clear any previous results
        self.validator.clear_results()
        
        # Run clinical bounds validation
        self.validator._validate_clinical_bounds(data)
        
        # Should detect value outside clinical bounds
        self.assertTrue(self.validator.has_warnings)
        warning_messages = ' '.join(self.validator.validation_warnings)
        self.assertIn('outside normal range', warning_messages.lower())
    
    def test_unit_consistency_validation(self):
        """Test validation of unit consistency per measurement concept."""
        data = self.create_valid_test_data()
        
        # Create inconsistent units for same measurement concept
        data['measurement'].loc[0, 'unit_concept_id'] = 2100000032  # mmHg instead of BPM for heart rate
        
        # Clear any previous results
        self.validator.clear_results()
        
        # Run clinical bounds validation
        self.validator._validate_clinical_bounds(data)
        
        # Should detect inconsistent units
        self.assertTrue(self.validator.has_warnings)
        warning_messages = ' '.join(self.validator.validation_warnings)
        self.assertIn('inconsistent units', warning_messages.lower())
    
    def test_value_usage_validation(self):
        """Test validation of proper value_as_number vs value_as_concept_id usage."""
        data = self.create_valid_test_data()
        
        # Create measurement with both numeric and concept values
        data['measurement'].loc[0, 'value_as_concept_id'] = 8507  # Should not have both
        
        # Clear any previous results
        self.validator.clear_results()
        
        # Run clinical bounds validation
        self.validator._validate_clinical_bounds(data)
        
        # Should detect improper value usage
        self.assertTrue(self.validator.has_warnings)
        warning_messages = ' '.join(self.validator.validation_warnings)
        self.assertIn('both numeric and concept values', warning_messages.lower())
    
    def test_missing_values_validation(self):
        """Test detection of measurements with no values."""
        data = self.create_valid_test_data()
        
        # Create measurement with no values
        data['measurement'].loc[0, 'value_as_number'] = None
        data['measurement'].loc[0, 'value_as_concept_id'] = None
        
        # Clear any previous results
        self.validator.clear_results()
        
        # Run clinical bounds validation
        self.validator._validate_clinical_bounds(data)
        
        # Should detect missing values
        self.assertTrue(self.validator.has_errors)
        error_messages = ' '.join(self.validator.validation_errors)
        self.assertIn('no value specified', error_messages.lower())
    
    def test_get_measurement_type_mapping(self):
        """Test the _get_measurement_type helper method."""
        # Test known concept ID mapping
        measurement_type = self.validator._get_measurement_type(3027018)
        self.assertEqual(measurement_type, 'heart_rate')
        
        # Test unknown concept ID
        measurement_type = self.validator._get_measurement_type(999999)
        self.assertEqual(measurement_type, 'unknown')


class TestCoverageReporting(TestDataValidator):
    """Test coverage and quality reporting functionality."""
    
    def test_survey_completion_calculation(self):
        """Test calculation of survey completion rates."""
        data = self.create_valid_test_data()
        
        # Add survey conduct data
        survey_data = pd.DataFrame({
            'survey_conduct_id': [1, 2, 3],
            'person_id': [1, 1, 2],
            'survey_concept_id': [2100000001, 2100000001, 2100000001],
            'survey_start_datetime': ['2024-06-02 09:00:00', '2024-06-03 09:00:00', '2024-07-02 09:00:00'],
            'survey_end_datetime': ['2024-06-02 09:30:00', '2024-06-03 09:30:00', '2024-07-02 09:30:00']
        })
        data['survey_conduct'] = survey_data
        
        # Calculate survey completion rates
        completion_rates = self.validator._calculate_survey_completion(
            data['episode'], data['survey_conduct']
        )
        
        # Should return completion rates for each episode
        self.assertIsInstance(completion_rates, dict)
        self.assertIn(1, completion_rates)  # Episode 1
        self.assertIn(2, completion_rates)  # Episode 2
        self.assertIn(3, completion_rates)  # Episode 3
        
        # Rates should be between 0 and 1
        for rate in completion_rates.values():
            self.assertGreaterEqual(rate, 0.0)
            self.assertLessEqual(rate, 1.0)
    
    def test_intervention_frequency_calculation(self):
        """Test calculation of intervention frequencies."""
        data = self.create_valid_test_data()
        
        # Add intervention data
        drug_data = pd.DataFrame({
            'drug_exposure_id': [1, 2],
            'person_id': [1, 2],
            'drug_concept_id': [2100000001, 2100000001],
            'drug_exposure_start_datetime': ['2024-06-02 10:00:00', '2024-07-02 11:00:00'],
            'drug_exposure_end_datetime': ['2024-06-07 10:00:00', '2024-07-07 11:00:00'],
            'drug_type_concept_id': [2100000001, 2100000001]
        })
        data['drug_exposure'] = drug_data
        
        # Calculate intervention frequencies
        frequencies = self.validator._calculate_intervention_frequencies(data)
        
        # Should return frequencies for intervention tables
        self.assertIsInstance(frequencies, dict)
        self.assertIn('drug_exposure', frequencies)
        self.assertEqual(frequencies['drug_exposure'], 2)
    
    def test_measurement_outlier_detection(self):
        """Test detection of measurement outliers using IQR method."""
        # Create measurement data with outliers
        measurement_data = pd.DataFrame({
            'measurement_id': list(range(1, 21)),
            'person_id': [1] * 20,
            'measurement_concept_id': [3027018] * 20,  # Heart rate
            'measurement_datetime': [f'2024-06-{i:02d} 10:00:00' for i in range(1, 21)],
            'measurement_type_concept_id': [2100000012] * 20,
            'value_as_number': [70, 72, 74, 76, 78, 80, 82, 84, 86, 88,  # Normal values
                               90, 92, 94, 96, 98, 100, 102, 104, 200, 250],  # Last two are outliers
            'value_as_concept_id': [None] * 20,
            'unit_concept_id': [2100000031] * 20
        })
        
        # Detect outliers
        outliers = self.validator._detect_measurement_outliers(measurement_data)
        
        # Should detect outliers
        self.assertIsInstance(outliers, dict)
        self.assertGreater(len(outliers), 0)
        
        # Check that extreme values are detected
        for outlier_list in outliers.values():
            self.assertIn(200.0, outlier_list)
            self.assertIn(250.0, outlier_list)
    
    def test_data_completeness_calculation(self):
        """Test calculation of data completeness."""
        data = self.create_valid_test_data()
        
        # Add some null values
        data['person'].loc[0, 'month_of_birth'] = None
        data['measurement'].loc[0, 'value_as_number'] = None
        
        # Calculate data completeness
        completeness = self.validator._calculate_data_completeness(data)
        
        # Should return completeness for each table
        self.assertIsInstance(completeness, dict)
        for table_name in data.keys():
            self.assertIn(table_name, completeness)
            self.assertGreaterEqual(completeness[table_name], 0.0)
            self.assertLessEqual(completeness[table_name], 1.0)
    
    def test_coverage_report_generation(self):
        """Test generation of complete coverage report."""
        data = self.create_valid_test_data()
        
        # Add survey conduct data for coverage calculation
        survey_data = pd.DataFrame({
            'survey_conduct_id': [1, 2],
            'person_id': [1, 2],
            'survey_concept_id': [2100000001, 2100000001],
            'survey_start_datetime': ['2024-06-02 09:00:00', '2024-07-02 09:00:00'],
            'survey_end_datetime': ['2024-06-02 09:30:00', '2024-07-02 09:30:00']
        })
        data['survey_conduct'] = survey_data
        
        # Clear any previous results
        self.validator.clear_results()
        
        # Generate coverage report
        self.validator._generate_coverage_report(data)
        
        # Check that report was generated
        self.assertIn('coverage_report', self.validator.validation_report)
        report = self.validator.validation_report['coverage_report']
        
        # Should contain expected sections
        self.assertIn('survey_completion', report)
        self.assertIn('data_completeness', report)


class TestValidationIntegration(TestDataValidator):
    """Test complete validation workflow integration."""
    
    def test_complete_validation_workflow(self):
        """Test the complete validation workflow."""
        data = self.create_valid_test_data()
        
        # Run complete validation
        is_valid = self.validator.validate(data)
        
        # Should pass validation for valid data
        self.assertTrue(is_valid)
        self.assertFalse(self.validator.has_errors)
    
    def test_validation_with_errors(self):
        """Test validation workflow with errors."""
        data = self.create_valid_test_data()
        
        # Introduce errors
        data['episode'].loc[0, 'person_id'] = 999  # Orphaned record
        data['episode'].loc[1, 'episode_start_datetime'] = '2024-07-25 08:00:00'  # Start > end
        
        # Run complete validation
        is_valid = self.validator.validate(data)
        
        # Should fail validation
        self.assertFalse(is_valid)
        self.assertTrue(self.validator.has_errors)
    
    def test_validation_report_generation(self):
        """Test generation of complete validation report."""
        data = self.create_valid_test_data()
        
        # Run validation
        self.validator.validate(data)
        
        # Get validation report
        report = self.validator.get_validation_report()
        
        # Should contain expected sections
        self.assertIn('errors', report)
        self.assertIn('warnings', report)
        self.assertIn('details', report)
        self.assertIn('summary', report)
        
        # Summary should contain counts
        summary = report['summary']
        self.assertIn('total_errors', summary)
        self.assertIn('total_warnings', summary)
        self.assertIn('validation_passed', summary)
    
    def test_validation_without_concept_dictionary(self):
        """Test validation without concept dictionary."""
        # Create validator without concept dictionary
        validator_no_concepts = DataValidator(self.config, None)
        data = self.create_valid_test_data()
        
        # Should still run validation (but skip concept validation)
        is_valid = validator_no_concepts.validate(data)
        self.assertTrue(is_valid)
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        empty_data = {
            'person': pd.DataFrame(),
            'episode': pd.DataFrame(),
            'measurement': pd.DataFrame()
        }
        
        # Should handle empty data gracefully
        is_valid = self.validator.validate(empty_data)
        self.assertTrue(is_valid)  # Empty data is technically valid


if __name__ == '__main__':
    # Create tests directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    # Run the tests
    unittest.main(verbosity=2)