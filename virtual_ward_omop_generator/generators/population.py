"""Population generator for person demographics and observation periods."""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from ..models.base import BaseGenerator
from ..exceptions import ConfigurationError
from ..validation.concept_validator import ConceptValidator
from ..utils.temporal_coordinator import TemporalCoordinator


class PopulationGenerator(BaseGenerator):
    """Generates person demographics and observation periods."""
    
    def __init__(self, config: Dict[str, Any], concept_dict: pd.DataFrame, rng: np.random.Generator):
        """Initialize the PopulationGenerator with concept validation and temporal coordination.
        
        Args:
            config: Configuration dictionary
            concept_dict: DataFrame containing valid concept IDs
            rng: Random number generator
        """
        super().__init__(config, concept_dict, rng)
        self.concept_validator = ConceptValidator(concept_dict)
        self.temporal_coordinator = TemporalCoordinator()
    
    def _validate_config(self) -> None:
        """Validate population generator configuration."""
        if 'dataset' not in self.config:
            raise ConfigurationError("Missing 'dataset' section in configuration")
        
        dataset_config = self.config['dataset']
        
        # Validate persons count
        if 'persons' not in dataset_config:
            raise ConfigurationError("Missing 'persons' in dataset configuration")
        
        persons = dataset_config['persons']
        if not isinstance(persons, int) or persons < 1 or persons > 20000:
            raise ConfigurationError("persons must be an integer between 1 and 20000")
        
        # Validate types section exists for concept IDs
        if 'types' not in self.config:
            raise ConfigurationError("Missing 'types' section in configuration")
        
        types_config = self.config['types']
        if 'period_type_virtual_ward' not in types_config:
            raise ConfigurationError("Missing 'period_type_virtual_ward' in types configuration")
    
    def generate(self, locations: pd.DataFrame, **kwargs: Any) -> Dict[str, pd.DataFrame]:
        """Generate population data including persons and observation periods.
        
        Returns:
            Dictionary containing 'person' and 'observation_period' DataFrames
        """
        persons_count = self.config['dataset']['persons']
        
        # Generate person demographics
        persons_df = self._generate_persons(persons_count)
        persons_df['location_id'] = locations.location_id.sample(n=persons_count, replace=True).reset_index(drop=True)
        self._log_generation_stats(persons_df, "person")
        
        # Generate observation periods
        observation_periods_df = self._generate_observation_periods(persons_df)
        self._log_generation_stats(observation_periods_df, "observation_period")
        
        per_person_obs = self._generate_per_person_observations(persons_df)
        self._log_generation_stats(per_person_obs, 'per_person_obs')

        return {
            'person': persons_df,
            'observation_period': observation_periods_df,
            'per_person_obs': per_person_obs
        }
    
    def _generate_weight_from_bmi(self, bmi_distribution: dict, height: float) -> float:
        """Generate weight based on BMI distribution and person's actual height.
        Args:
            bmi_distribution: Dictionary with BMI category probabilities
            height: Person's height in meters
        Returns:
            Generated weight in kg based on BMI distribution
        """
        # BMI category ranges (in kg/m²)
        bmi_categories = {
            'underweight': (16.0, 18.5),
            'healthy_weight': (18.5, 25.0),
            'overweight': (25.0, 30.0),
            'obese': (30.0, 40.0),
            'morbidly_obese': (40.0, 50.0)
        }
        # Sample BMI category based on distribution
        categories = list(bmi_categories.keys())
        probabilities = [bmi_distribution.get(cat, 0.0) for cat in categories]
        # Normalize probabilities to sum to 1
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            # Fallback to healthy weight if no valid distribution
            probabilities = [0.0, 1.0, 0.0, 0.0, 0.0]
        # Choose category
        category = self.rng.choice(categories, p=probabilities)
        bmi_range = bmi_categories[category]
        # Sample BMI within the chosen category range
        bmi = self.rng.uniform(bmi_range[0], bmi_range[1])
        # Calculate weight: weight = BMI × height²
        weight = bmi * (height ** 2)
        # Apply safety bounds only for extreme outliers
        weight = max(30, min(220, weight))

        return round(weight, 1)

    def _generate_per_person_observations(self, persons_df: pd.DataFrame) -> pd.DataFrame:
        if not self.config['per_person_obs']:
            return pd.DataFrame()
        observations = []
        observation_type_id = self.config['types']['observation_type_clinician_reported']

        yes_concept_id = self.config['types'].get('yes', 2100000020)  # Default fallback
        no_concept_id = self.config['types'].get('no', 2100000021)   # Default fallback
        unable_concept_id = self.config['types'].get('unable', 2100000022)  # Default fallback

        concept_name_map = {
            yes_concept_id: 'yes',
            no_concept_id: 'no',
            unable_concept_id: 'unable'
        }

        survey_id = 1
        observation_id = 1

        for _, row in persons_df.iterrows():
            # First pass: generate all numeric observations and store height for weight calculation
            person_height = None
            person_numeric_values = {}

            for var in self.config['per_person_obs'].get('numeric', []):
                concept_id = var['id']
                # Check if this is weight with BMI distribution
                if concept_id == 2100000106 and 'bmi_distribution' in var:
                    # Skip weight for now, will generate after height
                    person_numeric_values[concept_id] = var
                    continue
                # Generate normal numeric value (height, LTC count, etc.)
                mean_value = var['mean']
                sd = var['sd']
                min_bound = var['min']
                max_bound = var['max']

                numeric_value = self.rng.normal(mean_value, sd)
                numeric_value = max(min_bound, min(max_bound, numeric_value))
                numeric_value = round(numeric_value, 2)

                # Store height for later use
                if concept_id == 3036277:  # Height concept
                    person_height = numeric_value

                # Treat per-person observations as taken on the users 18th birthday by the GP for now
                obs_dt = datetime(row.year_of_birth + 18, row.month_of_birth, row.day_of_birth)

                observations.append({
                    'observation_id': observation_id,
                    'person_id': row.person_id,
                    'observation_concept_id': concept_id,
                    'observation_date': obs_dt.date(),
                    'observation_datetime': obs_dt,
                    'observation_type_concept_id': observation_type_id,
                    'value_as_number': numeric_value,
                    'value_as_string': None,
                    'value_as_concept_id': None,
                    'qualifier_concept_id': None,
                    'unit_concept_id': None,
                    'provider_id': None,
                    'visit_occurrence_id': None,
                    'visit_detail_id': None,
                    'observation_source_value': f"NUMERIC_{concept_id}",
                    'observation_source_concept_id': concept_id,
                    'unit_source_value': 'm' if concept_id == 3036277 else None,
                    'qualifier_source_value': None,
                    'value_source_value': str(numeric_value),
                    'observation_event_id': None,
                    'obs_event_field_concept_id': None,
                    'survey_conduct_id': survey_id,
                })
                observation_id += 1

            # Second pass: generate weight using BMI distribution and actual height
            for concept_id, var in person_numeric_values.items():
                if concept_id == 2100000106 and person_height is not None:
                    # Generate weight from BMI distribution using person's actual height
                    weight_value = self._generate_weight_from_bmi(var['bmi_distribution'], person_height)

                    obs_dt = datetime(row.year_of_birth + 18, row.month_of_birth, row.day_of_birth)

                    observations.append({
                        'observation_id': observation_id,
                        'person_id': row.person_id,
                        'observation_concept_id': concept_id,
                        'observation_date': obs_dt.date(),
                        'observation_datetime': obs_dt,
                        'observation_type_concept_id': observation_type_id,
                        'value_as_number': weight_value,
                        'value_as_string': None,
                        'value_as_concept_id': None,
                        'qualifier_concept_id': None,
                        'unit_concept_id': None,
                        'provider_id': None,
                        'visit_occurrence_id': None,
                        'visit_detail_id': None,
                        'observation_source_value': f"WEIGHT_{concept_id}",
                        'observation_source_concept_id': concept_id,
                        'unit_source_value': 'kg',
                        'qualifier_source_value': None,
                        'value_source_value': str(weight_value),
                        'observation_event_id': None,
                        'obs_event_field_concept_id': None,
                        'survey_conduct_id': survey_id,
                    })
                    observation_id += 1
                    # Generate baseline BMI observation from weight and height
                    baseline_bmi = weight_value / (person_height ** 2)
                    baseline_bmi = round(baseline_bmi, 2)

                    observations.append({
                        'observation_id': observation_id,
                        'person_id': row.person_id,
                        'observation_concept_id': 3038553,  # Body mass index (OMOP standard)
                        'observation_date': obs_dt.date(),
                        'observation_datetime': obs_dt,
                        'observation_type_concept_id': observation_type_id,
                        'value_as_number': baseline_bmi,
                        'value_as_string': None,
                        'value_as_concept_id': None,
                        'qualifier_concept_id': None,
                        'unit_concept_id': None,
                        'provider_id': None,
                        'visit_occurrence_id': None,
                        'visit_detail_id': None,
                        'observation_source_value': f"BMI_BASELINE_{row.person_id}",
                        'observation_source_concept_id': 3038553,
                        'unit_source_value': 'kg/m²',
                        'qualifier_source_value': None,
                        'value_source_value': str(baseline_bmi),
                        'observation_event_id': None,
                        'obs_event_field_concept_id': None,
                        'survey_conduct_id': survey_id,
                    })
                    observation_id += 1
                
            for var in self.config['per_person_obs'].get('boolean', []):
                concept_id = var['id']
                base_prob = var['base_prob']

                # Small chance of "Unable to Rate" response
                unable_prob = 0.02
                
                if self.rng.random() < unable_prob:
                    value_concept_id = unable_concept_id
                elif self.rng.random() < base_prob:
                    value_concept_id = yes_concept_id
                else:
                    value_concept_id = no_concept_id
                
                # Treat per-person observations as taken on the users 18th birthday by the GP for now
                obs_dt = datetime(row.year_of_birth + 18, row.month_of_birth, row.day_of_birth)

                observations.append({
                    'observation_id': observation_id,
                    'person_id': row.person_id,
                    'observation_concept_id': concept_id,
                    'observation_date': obs_dt.date(),
                    'observation_datetime': obs_dt,
                    'observation_type_concept_id': observation_type_id,
                    'value_as_number': None,
                    'value_as_string': None,
                    'value_as_concept_id': value_concept_id,
                    'qualifier_concept_id': None,
                    'unit_concept_id': None,
                    'provider_id': None,
                    'visit_occurrence_id': None,
                    'visit_detail_id': None,
                    'observation_source_value': f"PROM_{concept_id}",
                    'observation_source_concept_id': concept_id,
                    'unit_source_value': None,
                    'qualifier_source_value': None,
                    'value_source_value': concept_name_map[value_concept_id],
                    'observation_event_id': None,
                    'obs_event_field_concept_id': None,
                    'survey_conduct_id': survey_id,
                })
                
                observation_id += 1
        
            survey_id += 1

        return pd.DataFrame(observations)

    def _generate_persons(self, count: int) -> pd.DataFrame:
        """Generate person demographics.
        
        Args:
            count: Number of persons to generate
            
        Returns:
            DataFrame with person demographics
        """
        # Generate person IDs starting from 1
        person_ids = np.arange(1, count + 1)
        
        # Generate gender
        male_concept = self.concept_validator.validate_concept(8507, 'male')
        female_concept = self.concept_validator.validate_concept(8532, 'female')
        gender_concepts = self.rng.choice([male_concept, female_concept], size=count, p=[0.385, 0.615])
        
        # Generate birth dates using TemporalCoordinator
        birth_years = np.zeros(count, dtype=int)
        birth_months = np.zeros(count, dtype=int)
        birth_days = np.zeros(count, dtype=int)
        
        for i in range(count):
            year, month, day = self.temporal_coordinator.generate_birth_date(
                self.rng, 
                self.config['dataset']['age_ranges']
            )
            birth_years[i] = year
            birth_months[i] = month
            birth_days[i] = day
        
        # Generate race concept IDs with realistic US distribution using validated concepts
        race_concepts = self._generate_validated_race_concepts(count)
        
        # Generate ethnicity concept IDs using validated concepts
        # Note: UK doesn't use Hispanic/Not Hispanic categories like US
        # Using "Not Hispanic" as proxy for UK population
        hispanic_concept = self.concept_validator.validate_concept(2100001020, 'hispanic')
        not_hispanic_concept = self.concept_validator.validate_concept(2100001021, 'not_hispanic')

        ethnicity_concepts = self.rng.choice([hispanic_concept, not_hispanic_concept], size=count, p=[0.005, 0.995])
        
        persons_df = pd.DataFrame({
            'person_id': person_ids,
            'gender_concept_id': gender_concepts,
            'year_of_birth': birth_years,
            'month_of_birth': birth_months,
            'day_of_birth': birth_days,
            'race_concept_id': race_concepts,
            'ethnicity_concept_id': ethnicity_concepts
        })

        return persons_df
    
    def _generate_validated_race_concepts(self, count: int) -> np.ndarray:
        """Generate race concept IDs

        Args:
            count: Number of race concepts to generate

        Returns:
            Array of validated race concept IDs
        """
        # VH_VOCAB race concept IDs
        raw_race_options = [
            2100001010,   # White
            2100001011,   # Black or African American
            2100001012,   # Asian
            2100001013,   # American Indian or Alaska Native (not applicable to UK, kept minimal for compatibility)
            2100001014,   # Native Hawaiian or Other Pacific Islander (not applicable to UK, kept minimal)
            2100001015    # Other Race/Mixed
        ]

        # Validate each race concept
        validated_race_options = []
        for race_concept in raw_race_options:
            validated_concept = self.concept_validator.validate_concept(race_concept)
            validated_race_options.append(validated_concept)

        # Note: US race categories don't perfectly map to UK
        race_probs = [0.918, 0.019, 0.028, 0.0005, 0.0005, 0.034]

        return self.rng.choice(validated_race_options, size=count, p=race_probs)
    
    def _generate_observation_periods(self, persons_df: pd.DataFrame) -> pd.DataFrame:
        """Generate observation periods spanning all episodes for each person.
        
        Args:
            persons_df: DataFrame containing person data
            
        Returns:
            DataFrame with observation periods
        """
        # Get and validate the virtual ward period type concept ID from config
        raw_period_type_concept_id = self.config['types']['period_type_virtual_ward']
        period_type_concept_id = self.concept_validator.validate_concept(
            raw_period_type_concept_id, 'period_type'
        )
        
        # Generate observation period IDs
        observation_period_ids = np.arange(1, len(persons_df) + 1)
        
        # Create placeholder observation periods that will be updated after episodes are generated
        # Use temporal coordinator's base year for realistic date ranges
        base_year = self.temporal_coordinator.base_year
        base_start_date = date(base_year - 1, 1, 1)  # Start from previous year
        base_end_date = date(base_year, 12, 31)      # End at current year
        
        # Generate start dates within the first quarter of the base period
        start_date_offsets = self.rng.integers(0, 90, size=len(persons_df))  # 0-89 days from base
        start_dates = [base_start_date + timedelta(days=int(offset)) for offset in start_date_offsets]
        
        # Generate end dates ensuring they're after start dates and within reasonable range
        # Minimum observation period of 30 days, maximum of full range
        min_duration_days = 30
        max_duration_days = (base_end_date - base_start_date).days
        
        end_dates = []
        for start_date in start_dates:
            max_possible_duration = (base_end_date - start_date).days
            actual_max_duration = min(max_duration_days, max_possible_duration)
            duration = self.rng.integers(min_duration_days, actual_max_duration + 1)
            end_dates.append(start_date + timedelta(days=int(duration)))
        
        observation_periods_df = pd.DataFrame({
            'observation_period_id': observation_period_ids,
            'person_id': persons_df['person_id'].values,
            'observation_period_start_date': start_dates,
            'observation_period_end_date': end_dates,
            'period_type_concept_id': period_type_concept_id
        })
        
        return observation_periods_df
    
    def update_observation_periods_for_episodes(
        self, 
        observation_periods_df: pd.DataFrame, 
        episodes_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Update observation periods to ensure they span all episodes for each person.
        
        This method should be called after episodes are generated to ensure proper coverage.
        
        Args:
            observation_periods_df: Current observation periods DataFrame
            episodes_df: Episodes DataFrame with start/end datetimes
            
        Returns:
            Updated observation periods DataFrame
        """
        updated_periods = observation_periods_df.copy()
        
        # Group episodes by person to find min start and max end dates
        episode_bounds = episodes_df.groupby('person_id').agg({
            'episode_start_datetime': 'min',
            'episode_end_datetime': 'max'
        }).reset_index()
        
        # Convert datetime to date for observation periods
        episode_bounds['min_start_date'] = episode_bounds['episode_start_datetime'].dt.date
        episode_bounds['max_end_date'] = episode_bounds['episode_end_datetime'].dt.date
        
        # Merge with observation periods
        merged = updated_periods.merge(episode_bounds, on='person_id', how='left')
        
        # Update start dates to be no later than first episode start
        merged['observation_period_start_date'] = np.minimum(
            merged['observation_period_start_date'],
            merged['min_start_date']
        )
        
        # Update end dates to be no earlier than last episode end
        merged['observation_period_end_date'] = np.maximum(
            merged['observation_period_end_date'],
            merged['max_end_date']
        )
        
        # Return only the observation period columns
        return merged[['observation_period_id', 'person_id', 'observation_period_start_date', 
                      'observation_period_end_date', 'period_type_concept_id']]