"""Comprehensive data validation system for OMOP CDM data integrity."""

from typing import Dict, Any, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..models.base import BaseValidator
from ..models.omop import OMOPTables


class DataValidator(BaseValidator):
    """Validates generated OMOP data for integrity and consistency."""
    
    def __init__(self, config: Dict[str, Any], concept_dict: Optional[pd.DataFrame] = None):
        super().__init__(config)
        self.concept_dict = concept_dict
        self.omop_tables = OMOPTables(config)
        self.validation_report = {}
        
        # Clinical bounds for measurements
        self.clinical_bounds = {
            # Vital signs
            'heart_rate': (30, 200),  # bpm
            'systolic_bp': (70, 250),  # mmHg
            'diastolic_bp': (40, 150),  # mmHg
            'temperature': (32.0, 42.0),  # Celsius
            'oxygen_saturation': (70, 100),  # percent
            'respiratory_rate': (8, 40),  # breaths per minute
            'weight': (30, 300),  # kg
            'height': (100, 250),  # cm
            'bmi': (10, 60),  # kg/m²
            
            # Lab values (common ranges)
            'glucose': (50, 600),  # mg/dL
            'hemoglobin': (5, 20),  # g/dL
            'creatinine': (0.3, 15),  # mg/dL
            'sodium': (120, 160),  # mEq/L
            'potassium': (2.0, 8.0),  # mEq/L
        }
    
    def validate(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Validate OMOP data for integrity and consistency.
        
        Args:
            data: Dictionary of table name to DataFrame
            
        Returns:
            True if validation passes (no errors)
        """
        self.clear_results()
        self.validation_report = {}
        
        self.logger.info("Starting comprehensive data validation")
        
        # Run all validation checks
        self._validate_foreign_keys(data)
        self._validate_temporal_integrity(data)
        self._validate_clinical_bounds(data)
        self._generate_coverage_report(data)
        
        # Log summary
        total_errors = len(self.validation_errors)
        total_warnings = len(self.validation_warnings)
        
        self.logger.info(f"Validation complete: {total_errors} errors, {total_warnings} warnings")
        
        if total_errors > 0:
            self.logger.error("Data validation failed - see errors above")
            return False
        
        self.logger.info("Data validation passed")
        return True
    
    def _validate_foreign_keys(self, data: Dict[str, pd.DataFrame]) -> None:
        """Validate foreign key relationships across all tables.
        
        Args:
            data: Dictionary of table name to DataFrame
        """
        self.logger.info("Validating foreign key relationships")
        
        for table_name, df in data.items():
            if df.empty:
                continue
                
            schema = self.omop_tables.get_schema(table_name)
            if not schema or not schema.foreign_keys:
                continue
                
            for fk_column, reference in schema.foreign_keys.items():
                if fk_column not in df.columns:
                    continue
                    
                ref_table, ref_column = reference.split('.')
                if ref_table not in data or data[ref_table].empty:
                    self.add_error(f"Referenced table {ref_table} is missing or empty for FK {table_name}.{fk_column}")
                    continue
                
                # Check for orphaned records
                orphaned = self._find_orphaned_records(df, fk_column, data[ref_table], ref_column)
                if len(orphaned) > 0:
                    self.add_error(f"Found {len(orphaned)} orphaned records in {table_name}.{fk_column}")
                    self.validation_report[f"{table_name}_orphaned_{fk_column}"] = orphaned.tolist()
        
        # Validate concept ID references if concept dictionary is available
        if self.concept_dict is not None:
            self._validate_concept_references(data)
    
    def _find_orphaned_records(self, child_df: pd.DataFrame, child_col: str, 
                              parent_df: pd.DataFrame, parent_col: str) -> pd.Series:
        """Find records in child table that don't have corresponding parent records.
        
        Args:
            child_df: Child table DataFrame
            child_col: Foreign key column in child table
            parent_df: Parent table DataFrame
            parent_col: Primary key column in parent table
            
        Returns:
            Series of orphaned values
        """
        # Remove null values from comparison
        child_values = child_df[child_col].dropna()
        parent_values = parent_df[parent_col].dropna()
        
        # Find values in child that don't exist in parent
        orphaned = child_values[~child_values.isin(parent_values)]
        return orphaned
    
    def _validate_concept_references(self, data: Dict[str, pd.DataFrame]) -> None:
        """Validate concept ID references against concept dictionary.
        
        Args:
            data: Dictionary of table name to DataFrame
        """
        self.logger.info("Validating concept ID references")
        
        valid_concept_ids = set(self.concept_dict['concept_id'].values)
        
        # Define concept columns to check
        concept_columns = {
            'person': ['gender_concept_id', 'race_concept_id', 'ethnicity_concept_id'],
            'observation_period': ['period_type_concept_id'],
            'episode': ['episode_concept_id', 'episode_object_concept_id', 'episode_type_concept_id', 'episode_source_concept_id'],
            'visit_occurrence': ['visit_concept_id', 'visit_type_concept_id', 'visit_source_concept_id', 
                                'admitted_from_concept_id', 'discharged_to_concept_id'],
            'survey_conduct': ['survey_concept_id', 'assisted_concept_id', 'respondent_type_concept_id',
                              'timing_concept_id', 'collection_method_concept_id', 'survey_source_concept_id',
                              'validated_survey_concept_id'],
            'observation': ['observation_concept_id', 'observation_type_concept_id', 'value_as_concept_id',
                           'qualifier_concept_id', 'unit_concept_id', 'observation_source_concept_id',
                           'obs_event_field_concept_id'],
            'measurement': ['measurement_concept_id', 'measurement_type_concept_id', 'operator_concept_id',
                           'value_as_concept_id', 'unit_concept_id', 'measurement_source_concept_id',
                           'unit_source_concept_id', 'meas_event_field_concept_id'],
            'condition_occurrence': ['condition_concept_id', 'condition_type_concept_id', 'condition_status_concept_id',
                                   'condition_source_concept_id'],
            'drug_exposure': ['drug_concept_id', 'drug_type_concept_id', 'route_concept_id', 'drug_source_concept_id'],
            'procedure_occurrence': ['procedure_concept_id', 'procedure_type_concept_id', 'modifier_concept_id',
                                   'procedure_source_concept_id'],
            'device_exposure': ['device_concept_id', 'device_type_concept_id', 'unit_concept_id',
                               'device_source_concept_id', 'unit_source_concept_id'],
            'death': ['death_type_concept_id', 'cause_concept_id', 'cause_source_concept_id']
        }
        
        for table_name, columns in concept_columns.items():
            if table_name not in data or data[table_name].empty:
                continue
                
            df = data[table_name]
            for column in columns:
                if column not in df.columns:
                    continue
                    
                # Check for invalid concept IDs (excluding nulls)
                concept_values = df[column].dropna()
                invalid_concepts = concept_values[~concept_values.isin(valid_concept_ids)]
                
                if len(invalid_concepts) > 0:
                    self.add_error(f"Found {len(invalid_concepts)} invalid concept IDs in {table_name}.{column}")
                    self.validation_report[f"{table_name}_invalid_{column}"] = invalid_concepts.unique().tolist()
    
    def _validate_temporal_integrity(self, data: Dict[str, pd.DataFrame]) -> None:
        """Validate temporal consistency and episode boundaries.
        
        Args:
            data: Dictionary of table name to DataFrame
        """
        self.logger.info("Validating temporal integrity")
        
        # Check observation periods encompass all episodes
        if 'observation_period' in data and 'episode' in data:
            self._validate_observation_period_coverage(data['observation_period'], data['episode'])
        
        # Check start <= end for datetime ranges
        datetime_range_tables = {
            'episode': ('episode_start_datetime', 'episode_end_datetime'),
            'visit_occurrence': ('visit_start_datetime', 'visit_end_datetime'),
            'survey_conduct': ('survey_start_datetime', 'survey_end_datetime'),
            'condition_occurrence': ('condition_start_datetime', 'condition_end_datetime'),
            'drug_exposure': ('drug_exposure_start_datetime', 'drug_exposure_end_datetime'),
            'procedure_occurrence': ('procedure_datetime', 'procedure_end_datetime'),
            'device_exposure': ('device_exposure_start_datetime', 'device_exposure_end_datetime')
        }
        
        for table_name, (start_col, end_col) in datetime_range_tables.items():
            if table_name not in data or data[table_name].empty:
                continue
                
            df = data[table_name]
            if start_col not in df.columns or end_col not in df.columns:
                continue
                
            # Check for invalid datetime ranges
            invalid_ranges = df[
                (df[start_col].notna()) & 
                (df[end_col].notna()) & 
                (df[start_col] > df[end_col])
            ]
            
            if len(invalid_ranges) > 0:
                self.add_error(f"Found {len(invalid_ranges)} records with start > end in {table_name}")
                self.validation_report[f"{table_name}_invalid_datetime_ranges"] = len(invalid_ranges)
        
        # Validate events fall within episode boundaries
        self._validate_episode_boundaries(data)
    
    def _validate_observation_period_coverage(self, obs_periods: pd.DataFrame, episodes: pd.DataFrame) -> None:
        """Validate that observation periods encompass all episodes.
        
        Args:
            obs_periods: Observation period DataFrame
            episodes: Episode DataFrame
        """
        if obs_periods.empty or episodes.empty:
            return
            
        # Group by person_id and check coverage
        for person_id in episodes['person_id'].unique():
            person_episodes = episodes[episodes['person_id'] == person_id]
            person_obs_periods = obs_periods[obs_periods['person_id'] == person_id]
            
            if person_obs_periods.empty:
                self.add_error(f"Person {person_id} has episodes but no observation periods")
                continue
            
            # Check if all episodes fall within observation periods
            for _, episode in person_episodes.iterrows():
                episode_start = pd.to_datetime(episode['episode_start_datetime'])
                episode_end = pd.to_datetime(episode['episode_end_datetime'])
                
                covered = False
                for _, obs_period in person_obs_periods.iterrows():
                    period_start = pd.to_datetime(obs_period['observation_period_start_date'])
                    period_end = pd.to_datetime(obs_period['observation_period_end_date'])
                    
                    # Treat observation period end date as end of day (23:59:59)
                    # since episodes can have specific times within the day
                    period_end_eod = period_end.replace(hour=23, minute=59, second=59, microsecond=999999)
                    
                    if period_start <= episode_start and episode_end <= period_end_eod:
                        covered = True
                        break
                
                if not covered:
                    self.add_error(f"Episode {episode['episode_id']} falls outside observation periods")
    
    def _validate_episode_boundaries(self, data: Dict[str, pd.DataFrame]) -> None:
        """Validate that all events fall within episode boundaries.
        
        Args:
            data: Dictionary of table name to DataFrame
        """
        if 'episode' not in data or data['episode'].empty:
            return
            
        episodes = data['episode']
        
        # Tables with events that should fall within episodes
        event_tables = {
            'visit_occurrence': 'visit_start_datetime',
            'survey_conduct': 'survey_start_datetime',
            'observation': 'observation_datetime',
            'measurement': 'measurement_datetime',
            'condition_occurrence': 'condition_start_datetime',
            'drug_exposure': 'drug_exposure_start_datetime',
            'procedure_occurrence': 'procedure_datetime',
            'device_exposure': 'device_exposure_start_datetime'
        }
        
        for table_name, datetime_col in event_tables.items():
            if table_name not in data or data[table_name].empty:
                continue
                
            df = data[table_name]
            if datetime_col not in df.columns:
                continue
                
            # Check each event against episode boundaries
            violations = 0
            for _, event in df.iterrows():
                person_id = event['person_id']
                event_datetime = pd.to_datetime(event[datetime_col])
                
                # Find episodes for this person
                person_episodes = episodes[episodes['person_id'] == person_id]
                
                # Check if event falls within any episode
                within_episode = False
                for _, episode in person_episodes.iterrows():
                    episode_start = pd.to_datetime(episode['episode_start_datetime'])
                    episode_end = pd.to_datetime(episode['episode_end_datetime'])
                    
                    if episode_start <= event_datetime <= episode_end:
                        within_episode = True
                        break
                
                if not within_episode:
                    violations += 1
            
            if violations > 0:
                self.add_warning(f"Found {violations} events in {table_name} outside episode boundaries")
                self.validation_report[f"{table_name}_outside_episodes"] = violations
    
    def _validate_clinical_bounds(self, data: Dict[str, pd.DataFrame]) -> None:
        """Validate measurement values within physiological ranges.
        
        Args:
            data: Dictionary of table name to DataFrame
        """
        self.logger.info("Validating clinical bounds and measurement consistency")
        
        if 'measurement' not in data or data['measurement'].empty:
            return
            
        measurements = data['measurement']
        
        # Validate numeric measurements against clinical bounds
        numeric_measurements = measurements[measurements['value_as_number'].notna()]
        
        for _, measurement in numeric_measurements.iterrows():
            concept_id = measurement['measurement_concept_id']
            value = measurement['value_as_number']
            unit_concept_id = measurement.get('unit_concept_id')
            
            # Map concept IDs to measurement types (simplified mapping)
            measurement_type = self._get_measurement_type(concept_id)
            
            if measurement_type in self.clinical_bounds:
                min_val, max_val = self.clinical_bounds[measurement_type]
                
                if value < min_val or value > max_val:
                    self.add_warning(
                        f"Measurement value {value} for {measurement_type} outside normal range "
                        f"({min_val}-{max_val})"
                    )
        
        # Validate unit consistency
        self._validate_unit_consistency(measurements)
        
        # Validate proper usage of value_as_number vs value_as_concept_id
        self._validate_value_usage(measurements)
    
    def _get_measurement_type(self, concept_id: int) -> str:
        """Map concept ID to measurement type for bounds checking.
        
        Args:
            concept_id: OMOP concept ID
            
        Returns:
            Measurement type string
        """
        # Simplified mapping - in practice this would use the concept dictionary
        concept_mapping = {
            # Standard OMOP concept IDs for common measurements
            3027018: 'heart_rate',
            3004249: 'systolic_bp',
            3012888: 'diastolic_bp',
            3020891: 'temperature',
            3016502: 'oxygen_saturation',
            3024171: 'respiratory_rate',
            3025315: 'weight',
            3036277: 'height',
            3038553: 'bmi',
            3004501: 'glucose',
            3000963: 'hemoglobin',
            3016723: 'creatinine',
            3019550: 'sodium',
            3023103: 'potassium'
        }
        
        return concept_mapping.get(concept_id, 'unknown')
    
    def _validate_unit_consistency(self, measurements: pd.DataFrame) -> None:
        """Validate unit consistency per measurement concept.
        
        Args:
            measurements: Measurement DataFrame
        """
        # Group by measurement concept and check unit consistency
        for concept_id in measurements['measurement_concept_id'].unique():
            concept_measurements = measurements[measurements['measurement_concept_id'] == concept_id]
            units = concept_measurements['unit_concept_id'].dropna().unique()
            
            if len(units) > 1:
                self.add_warning(f"Inconsistent units for measurement concept {concept_id}: {units}")
    
    def _validate_value_usage(self, measurements: pd.DataFrame) -> None:
        """Validate proper usage of value_as_number vs value_as_concept_id.
        
        Args:
            measurements: Measurement DataFrame
        """
        # Check for measurements with both numeric and concept values
        both_values = measurements[
            measurements['value_as_number'].notna() & 
            measurements['value_as_concept_id'].notna()
        ]
        
        if len(both_values) > 0:
            self.add_warning(f"Found {len(both_values)} measurements with both numeric and concept values")
        
        # Check for measurements with neither value
        no_values = measurements[
            measurements['value_as_number'].isna() & 
            measurements['value_as_concept_id'].isna()
        ]
        
        if len(no_values) > 0:
            self.add_error(f"Found {len(no_values)} measurements with no value specified")
    
    def _generate_coverage_report(self, data: Dict[str, pd.DataFrame]) -> None:
        """Generate coverage and quality reporting.
        
        Args:
            data: Dictionary of table name to DataFrame
        """
        self.logger.info("Generating coverage and quality report")
        
        report = {}
        
        # Calculate survey completion rates per episode
        if 'episode' in data and 'survey_conduct' in data and not data['episode'].empty and not data['survey_conduct'].empty:
            report['survey_completion'] = self._calculate_survey_completion(
                data['episode'], data['survey_conduct']
            )
        
        # Report intervention frequencies by archetype
        if 'episode' in data and not data['episode'].empty:
            report['intervention_frequencies'] = self._calculate_intervention_frequencies(data)
        
        # Generate outlier detection summaries
        if 'measurement' in data and not data['measurement'].empty:
            report['measurement_outliers'] = self._detect_measurement_outliers(data['measurement'])
        
        # Calculate data completeness
        report['data_completeness'] = self._calculate_data_completeness(data)
        
        self.validation_report['coverage_report'] = report
        
        # Log key metrics
        if 'survey_completion' in report:
            avg_completion = np.mean(list(report['survey_completion'].values()))
            self.logger.info(f"Average survey completion rate: {avg_completion:.1%}")
        
        if 'data_completeness' in report:
            for table, completeness in report['data_completeness'].items():
                self.logger.info(f"Data completeness for {table}: {completeness:.1%}")
    
    def _calculate_survey_completion(self, episodes: pd.DataFrame, surveys: pd.DataFrame) -> Dict[int, float]:
        """Calculate survey completion rates per episode.
        
        Args:
            episodes: Episode DataFrame
            surveys: Survey conduct DataFrame
            
        Returns:
            Dictionary of episode_id to completion rate
        """
        completion_rates = {}
        
        for _, episode in episodes.iterrows():
            episode_id = episode['episode_id']
            episode_start = pd.to_datetime(episode['episode_start_datetime'])
            episode_end = pd.to_datetime(episode['episode_end_datetime'])
            
            # Calculate expected number of daily surveys
            episode_days = (episode_end - episode_start).days + 1
            expected_surveys = episode_days
            
            # Count actual surveys for this episode
            episode_surveys = surveys[
                (surveys['person_id'] == episode['person_id']) &
                (pd.to_datetime(surveys['survey_start_datetime']) >= episode_start) &
                (pd.to_datetime(surveys['survey_start_datetime']) <= episode_end)
            ]
            
            actual_surveys = len(episode_surveys)
            completion_rate = actual_surveys / expected_surveys if expected_surveys > 0 else 0
            completion_rates[episode_id] = completion_rate
        
        return completion_rates
    
    def _calculate_intervention_frequencies(self, data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """Calculate intervention frequencies by archetype.
        
        Args:
            data: Dictionary of table name to DataFrame
            
        Returns:
            Dictionary of intervention type to frequency
        """
        frequencies = {}
        
        # Count different types of interventions
        intervention_tables = ['drug_exposure', 'procedure_occurrence', 'device_exposure']
        
        for table_name in intervention_tables:
            if table_name in data and not data[table_name].empty:
                frequencies[table_name] = len(data[table_name])
        
        return frequencies
    
    def _detect_measurement_outliers(self, measurements: pd.DataFrame) -> Dict[str, List[float]]:
        """Detect outliers in measurement values.
        
        Args:
            measurements: Measurement DataFrame
            
        Returns:
            Dictionary of measurement type to outlier values
        """
        outliers = {}
        
        # Group by measurement concept and detect outliers using IQR method
        for concept_id in measurements['measurement_concept_id'].unique():
            concept_measurements = measurements[
                (measurements['measurement_concept_id'] == concept_id) &
                (measurements['value_as_number'].notna())
            ]
            
            if len(concept_measurements) < 4:  # Need at least 4 values for IQR
                continue
                
            values = concept_measurements['value_as_number']
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_values = values[(values < lower_bound) | (values > upper_bound)]
            
            if len(outlier_values) > 0:
                measurement_type = self._get_measurement_type(concept_id)
                outliers[f"{measurement_type}_{concept_id}"] = outlier_values.tolist()
        
        return outliers
    
    def _calculate_data_completeness(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate data completeness for each table.
        
        Args:
            data: Dictionary of table name to DataFrame
            
        Returns:
            Dictionary of table name to completeness percentage
        """
        completeness = {}
        
        for table_name, df in data.items():
            if df.empty:
                completeness[table_name] = 0.0
                continue
                
            # Calculate percentage of non-null values across all columns
            total_cells = df.size
            non_null_cells = df.count().sum()
            completeness[table_name] = non_null_cells / total_cells if total_cells > 0 else 0.0
        
        return completeness
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get the complete validation report.
        
        Returns:
            Dictionary containing validation results and metrics
        """
        return {
            'errors': self.validation_errors,
            'warnings': self.validation_warnings,
            'details': self.validation_report,
            'summary': {
                'total_errors': len(self.validation_errors),
                'total_warnings': len(self.validation_warnings),
                'validation_passed': not self.has_errors
            }
        }