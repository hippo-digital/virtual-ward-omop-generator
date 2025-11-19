"""Signal generator for PROM and device measurement data."""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from ..models.base import BaseGenerator
from ..exceptions import DataGenerationError

logger = logging.getLogger(__name__)


class SignalGenerator(BaseGenerator):
    """Generates PROM and device measurement signals for virtual ward episodes."""
    
    def __init__(self, config: Dict[str, Any], concept_dict: pd.DataFrame, 
                 random_generator: np.random.Generator) -> None:
        """Initialize signal generator.
        
        Args:
            config: Configuration dictionary
            concept_dict: Concept dictionary DataFrame
            random_generator: NumPy random generator for reproducible results
        """
        super().__init__(config, concept_dict, random_generator)
        
        # Initialize concept validator and temporal coordinator
        from ..validation.concept_validator import ConceptValidator
        from ..utils.temporal_coordinator import TemporalCoordinator
        
        self.concept_validator = ConceptValidator(concept_dict)
        self.temporal_coordinator = TemporalCoordinator()
        
        # Cache concept mappings for performance
        self._cache_concept_mappings()
        
    def _validate_config(self) -> None:
        """Validate signal generator configuration."""
        # Check for required top-level sections
        if 'dataset' not in self.config:
            raise DataGenerationError("Missing required config section: dataset")
        if 'proms' not in self.config:
            raise DataGenerationError("Missing required config section: proms")
        if 'measurements' not in self.config:
            raise DataGenerationError("Missing required config section: measurements")
                
        # Validate survey completion probabilities (can be in dataset or top-level)
        survey_config = self.config.get('survey_completion') or self.config['dataset'].get('survey_completion')
        if survey_config:
            if not (0 <= survey_config.get('weekday', 0) <= 1):
                raise DataGenerationError("Survey weekday completion probability must be between 0 and 1")
            if not (0 <= survey_config.get('weekend', 0) <= 1):
                raise DataGenerationError("Survey weekend completion probability must be between 0 and 1")
            
        self.logger.info("Signal generator configuration validated")
        
    def _cache_concept_mappings(self) -> None:
        """Cache concept ID mappings for performance."""
        self.concept_map = {}
        for _, row in self.concept_dict.iterrows():
            self.concept_map[row['concept_id']] = {
                'name': row['concept_name'],
                'domain': row['domain_id']
            }
            
    def generate_survey_schedule(self, episodes: pd.DataFrame, starting_survey_id: int, death_lookup: Dict[int, Any] = None) -> pd.DataFrame:
        """Generate survey schedule with weekday/weekend patterns.

        Args:
            episodes: DataFrame with episode information
            starting_survey_id: Starting ID for survey records
            death_lookup: Optional dictionary mapping person_id to death_date

        Returns:
            DataFrame with survey_conduct records
        """
        self.logger.info(f"Generating survey schedule for {len(episodes)} episodes")

        survey_records = []
        survey_id = starting_survey_id
        
        # Get survey completion probabilities (can be in dataset or top-level)
        survey_config = self.config.get('survey_completion') or self.config['dataset'].get('survey_completion', {})
        weekday_prob = survey_config.get('weekday', 0.9)
        weekend_prob = survey_config.get('weekend', 0.8)
        
        # Collection methods and respondent types from config
        collection_methods = self.config['types']['collection_methods']
        respondent_types = self.config['types']['respondent_types']
        survey_concept_id = self.config['types'].get('survey_concept_id', 2100000010)

        # Prepare weighted distributions if configured
        collection_method_probs = None
        if 'collection_method_distribution' in self.config['types']:
            dist = self.config['types']['collection_method_distribution']
            collection_method_probs = np.array([dist.get(method, 1.0/len(collection_methods))
                                               for method in collection_methods])
            collection_method_probs = collection_method_probs / collection_method_probs.sum()  # Normalize

        respondent_type_probs = None
        if 'respondent_type_distribution' in self.config['types']:
            dist = self.config['types']['respondent_type_distribution']
            respondent_type_probs = np.array([dist.get(rtype, 1.0/len(respondent_types))
                                             for rtype in respondent_types])
            respondent_type_probs = respondent_type_probs / respondent_type_probs.sum()  # Normalize
        
        for _, episode in episodes.iterrows():
            person_id = episode['person_id']
            episode_id = episode['episode_id']
            start_date = pd.to_datetime(episode['episode_start_datetime'])
            end_date = pd.to_datetime(episode['episode_end_datetime'])

            # Skip episodes that start after death date
            if death_lookup is not None:
                death_date = death_lookup.get(person_id)
                if death_date and start_date.date() > death_date:
                    self.logger.debug(f"Skipping episode {episode_id} for person {person_id} - starts after death on {death_date}")
                    continue
            
            # Generate daily survey opportunities
            current_date = start_date.date()
            end_date_only = end_date.date()
            
            while current_date <= end_date_only:
                # Determine if it's a weekend (Saturday=5, Sunday=6)
                is_weekend = current_date.weekday() >= 5
                completion_prob = weekend_prob if is_weekend else weekday_prob
                
                # Decide if survey is completed
                if self.rng.random() < completion_prob:
                    # Generate survey timestamp around 09:00 ± 2h
                    base_time = datetime.combine(current_date, datetime.min.time().replace(hour=9))
                    time_offset = self.rng.normal(0, 2 * 60)  # ±2 hours in minutes
                    survey_time = base_time + timedelta(minutes=time_offset)
                    
                    # Ensure survey time is within episode bounds
                    survey_time = max(survey_time, start_date)
                    survey_time = min(survey_time, end_date)

                    # Weighted collection method and respondent type selection
                    if collection_method_probs is not None:
                        collection_method = self.rng.choice(collection_methods, p=collection_method_probs)
                    else:
                        collection_method = self.rng.choice(collection_methods)

                    if respondent_type_probs is not None:
                        respondent_type = self.rng.choice(respondent_types, p=respondent_type_probs)
                    else:
                        respondent_type = self.rng.choice(respondent_types)
                    
                    survey_records.append({
                        'survey_conduct_id': survey_id,
                        'person_id': person_id,
                        'survey_concept_id': survey_concept_id,
                        'survey_start_date': survey_time.date(),
                        'survey_start_datetime': survey_time,
                        'survey_end_date': survey_time.date(),
                        'survey_end_datetime': survey_time + timedelta(minutes=int(self.rng.integers(5, 20))),
                        'provider_id': None,
                        'assisted_concept_id': None,
                        'respondent_type_concept_id': respondent_type,
                        'timing_concept_id': None,
                        'collection_method_concept_id': collection_method,
                        'assisted_source_value': None,
                        'respondent_type_source_value': None,
                        'timing_source_value': None,
                        'collection_method_source_value': None,
                        'survey_source_value': f"VW_DAILY_{current_date.strftime('%Y%m%d')}",
                        'survey_source_concept_id': None,
                        'survey_source_identifier': f"VW_{person_id}_{current_date.strftime('%Y%m%d')}",
                        'validated_survey_concept_id': survey_concept_id,
                        'validated_survey_source_value': None,
                        'survey_version_number': "1.0",
                        'visit_occurrence_id': None,
                        'response_visit_occurrence_id': None,
                        'episode_id': episode_id
                    })
                    
                    survey_id += 1
                
                current_date += timedelta(days=1)
        
        survey_df = pd.DataFrame(survey_records)
        self.logger.info(f"Generated {len(survey_df)} survey conduct records")
        
        return survey_df
    
    def generate_prom_observations(self, surveys: pd.DataFrame, episodes: pd.DataFrame, per_person_obs: pd.DataFrame, first_observation_id: int) -> pd.DataFrame:
        """Generate PROM observations with realistic distributions.
        
        Args:
            surveys: DataFrame with survey_conduct records
            episodes: DataFrame with episode information for condition-specific baselines
            
        Returns:
            DataFrame with observation records for PROMs
        """
        self.logger.info(f"Generating PROM observations for {len(surveys)} surveys")
        
        observation_records = []
        observation_id = first_observation_id
        
        # Get PROM configurations
        numeric_proms = self.config['proms']['numeric']
        boolean_proms = self.config['proms']['boolean']
        
        # Type concept IDs
        observation_type_id = self.config['types']['observation_type_patient_reported']
        
        # Debug: check what's in types config
        self.logger.debug(f"Types config keys: {list(self.config['types'].keys())}")
        
        yes_concept_id = self.config['types'].get('yes', 2100000020)  # Default fallback
        no_concept_id = self.config['types'].get('no', 2100000021)   # Default fallback
        unable_concept_id = self.config['types'].get('unable', 2100000022)  # Default fallback
        
        # Create episode condition mapping for baselines
        episode_conditions = episodes.set_index('episode_id')['index_condition_concept_id'].to_dict()

        person_smoking = {}
        if per_person_obs is not None and not per_person_obs.empty:
            tobacco_obs = per_person_obs[per_person_obs['observation_concept_id'] == 4041306]
            for _, obs in tobacco_obs.iterrows():
                person_id = obs['person_id']
                value = obs['value_as_concept_id']
                if value == 2100000020:
                    person_smoking[person_id] = True 
                else:
                    person_smoking[person_id] = False

        for _, survey in surveys.iterrows():
            person_id = survey['person_id']
            episode_id = survey['episode_id']
            survey_id = survey['survey_conduct_id']
            survey_datetime = survey['survey_start_datetime']
            condition_id = episode_conditions.get(episode_id)

            is_smoker = person_smoking.get(person_id, False)
            if is_smoker:
                smoking_mult = 1.25
            else:
                smoking_mult = 1.0
            # Generate numeric PROM observations
            for prom_config in numeric_proms:
                concept_id = prom_config['id']
                min_val = prom_config['min']
                max_val = prom_config['max']
                
                # Get condition-specific baseline or default
                if 'baseline_mean_by_condition' in prom_config:
                    baseline_mean = prom_config['baseline_mean_by_condition'].get(
                        condition_id, prom_config['baseline_mean_by_condition']['default']
                    )
                else:
                    baseline_mean = prom_config['baseline_mean']

                # Apply smoking status multiplier to baseline (worse outcomes for smokers)
                baseline_mean = baseline_mean * smoking_mult

                baseline_sd = prom_config['baseline_sd']

                # Generate value with truncated normal distribution
                value = self._generate_truncated_normal(
                    baseline_mean, baseline_sd, min_val, max_val
                )
                
                observation_records.append({
                    'observation_id': observation_id,
                    'person_id': person_id,
                    'observation_concept_id': concept_id,
                    'observation_date': survey_datetime.date(),
                    'observation_datetime': survey_datetime,
                    'observation_type_concept_id': observation_type_id,
                    'value_as_number': value,
                    'value_as_string': None,
                    'value_as_concept_id': None,
                    'qualifier_concept_id': None,
                    'unit_concept_id': None,  # Numeric scales don't have units
                    'provider_id': None,
                    'visit_occurrence_id': None,
                    'visit_detail_id': None,
                    'observation_source_value': f"PROM_{concept_id}",
                    'observation_source_concept_id': concept_id,
                    'unit_source_value': None,
                    'qualifier_source_value': None,
                    'value_source_value': str(value),
                    'observation_event_id': None,
                    'obs_event_field_concept_id': None,
                    'survey_conduct_id': survey_id,
                    'episode_id': episode_id
                })
                
                observation_id += 1
            
            # Generate boolean PROM observations
            for prom_config in boolean_proms:
                concept_id = prom_config['id']
                
                # Get condition-specific probability or default
                if 'base_prob_by_condition' in prom_config:
                    base_prob = prom_config['base_prob_by_condition'].get(
                        condition_id, prom_config['base_prob_by_condition']['default']
                    )
                else:
                    base_prob = prom_config['base_prob']

                # Apply smoking status multiplier to probability (higher chance of symptoms for smokers)
                base_prob = min(base_prob * smoking_mult, 0.95)  # Cap at 95% to keep some variability

                # Small chance of "Unable to Rate" response
                unable_prob = 0.02
                
                if self.rng.random() < unable_prob:
                    value_concept_id = unable_concept_id
                elif self.rng.random() < base_prob:
                    value_concept_id = yes_concept_id
                else:
                    value_concept_id = no_concept_id
                
                observation_records.append({
                    'observation_id': observation_id,
                    'person_id': person_id,
                    'observation_concept_id': concept_id,
                    'observation_date': survey_datetime.date(),
                    'observation_datetime': survey_datetime,
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
                    'value_source_value': self.concept_map[value_concept_id]['name'],
                    'observation_event_id': None,
                    'obs_event_field_concept_id': None,
                    'survey_conduct_id': survey_id,
                    'episode_id': episode_id
                })
                
                observation_id += 1
        
        observation_df = pd.DataFrame(observation_records)
        self.logger.info(f"Generated {len(observation_df)} PROM observation records")
        
        return observation_df
    
    def _generate_truncated_normal(self, mean: float, sd: float, min_val: float, max_val: float) -> float:
        """Generate a value from truncated normal distribution.
        
        Args:
            mean: Mean of the distribution
            sd: Standard deviation
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Generated value within bounds
        """
        # Use rejection sampling for simplicity
        max_attempts = 100
        for _ in range(max_attempts):
            value = self.rng.normal(mean, sd)
            if min_val <= value <= max_val:
                return round(value, 1)
        
        # Fallback: clip to bounds
        value = self.rng.normal(mean, sd)
        return round(np.clip(value, min_val, max_val), 1)
    
    def generate_device_measurements(self, episodes: pd.DataFrame, visits: pd.DataFrame,
                                   device_assignments: pd.DataFrame, starting_observation_id: int = 1, death_lookup: Dict[int, Any] = None) -> pd.DataFrame:
        """Generate device measurements with proper temporal sequencing within visit timeframes.

        Args:
            episodes: DataFrame with episode information
            visits: DataFrame with visit_occurrence records for temporal boundaries
            device_assignments: DataFrame with device assignments per episode
            starting_observation_id: Starting ID for BMI observations (default: 1)
            death_lookup: Optional dictionary mapping person_id to death_date

        Returns:
            DataFrame with measurement records
        """
        self.logger.info(f"Generating device measurements for {len(episodes)} episodes")

        measurement_records = []
        measurement_id = 1
        bmi_observations = []
        observation_id = starting_observation_id
        
        # Get measurement configurations
        measurement_configs = {config['id']: config for config in self.config['measurements']}
        units_map = self.config['units']
        
        # Use validated concept IDs
        measurement_type_id = self.concept_validator.validate_concept(
            self.config['types']['measurement_type_patient_device'], 
            'measurement_type'
        )
        observation_type_id = self.config['types'].get('observation_type_clinician_reported', 2100000015)
        
        # Create episode condition mapping
        episode_conditions = episodes.set_index('episode_id')['index_condition_concept_id'].to_dict()
        
        # Group visits by episode for temporal boundaries
        visits_by_episode = visits.groupby('episode_id')
        
        for _, episode in episodes.iterrows():
            person_id = episode['person_id']
            episode_id = episode['episode_id']
            condition_id = episode_conditions[episode_id]

            # Skip episodes that start after death date
            if death_lookup is not None:
                death_date = death_lookup.get(person_id)
                if death_date and episode['episode_start_datetime'].date() > death_date:
                    self.logger.debug(f"Skipping episode {episode_id} for person {person_id} - starts after death on {death_date}")
                    continue

            # Get device assignments for this episode
            episode_devices = device_assignments[device_assignments['episode_id'] == episode_id]
            
            # Get visits for this episode for temporal boundaries
            episode_visits = visits_by_episode.get_group(episode_id) if episode_id in visits_by_episode.groups else pd.DataFrame()
            
            if episode_visits.empty:
                self.logger.warning(f"No visits found for episode {episode_id}, skipping measurements")
                continue
            
            for _, device_assignment in episode_devices.iterrows():
                device_concept_id = device_assignment['device_concept_id']
                
                # Map device to measurement concepts
                measurement_concepts = self._get_measurement_concepts_for_device(device_concept_id)
                
                for measurement_concept_id in measurement_concepts:
                    # Validate measurement concept ID
                    validated_concept_id = self.concept_validator.validate_concept(
                        measurement_concept_id, 'measurement'
                    )
                    
                    if validated_concept_id not in measurement_configs:
                        self.logger.warning(f"No config found for measurement concept {validated_concept_id}")
                        continue
                        
                    config = measurement_configs[validated_concept_id]
                    
                    # Generate measurements within visit timeframes
                    for _, visit in episode_visits.iterrows():
                        visit_start = pd.to_datetime(visit['visit_start_datetime'])
                        visit_end = pd.to_datetime(visit['visit_end_datetime'])
                        
                        # Generate 1-3 measurements per visit
                        num_measurements = self.rng.integers(1, 4)
                        
                        for _ in range(num_measurements):
                            # Generate measurement time within visit boundaries
                            measurement_time = self.temporal_coordinator.generate_measurement_datetime(
                                visit_start, visit_end, self.rng
                            )

                            # Generate measurement value with realistic bounds (pass person_id for weight)
                            value = self._generate_measurement_value(config, condition_id, person_id)
                            
                            # Validate unit concept ID
                            unit_concept_id = self.concept_validator.validate_concept(
                                units_map.get(validated_concept_id, 0), 'unit'
                            )
                            
                            bounds = config.get('bounds', [None, None])
                            
                            measurement_records.append({
                                'measurement_id': measurement_id,
                                'person_id': person_id,
                                'measurement_concept_id': validated_concept_id,
                                'measurement_date': measurement_time.date(),
                                'measurement_datetime': measurement_time,
                                'measurement_time': measurement_time.strftime('%H:%M:%S'),
                                'measurement_type_concept_id': measurement_type_id,
                                'operator_concept_id': None,
                                'value_as_number': value,
                                'value_as_concept_id': None,
                                'unit_concept_id': unit_concept_id if unit_concept_id != 0 else None,
                                'range_low': bounds[0] if bounds and bounds[0] is not None else None,
                                'range_high': bounds[1] if bounds and bounds[1] is not None else None,
                                'provider_id': None,
                                'visit_occurrence_id': visit['visit_occurrence_id'],
                                'visit_detail_id': None,
                                'measurement_source_value': f"DEVICE_{validated_concept_id}",
                                'measurement_source_concept_id': validated_concept_id,
                                'unit_source_value': None,
                                'unit_source_concept_id': unit_concept_id if unit_concept_id != 0 else None,
                                'value_source_value': str(value),
                                'measurement_event_id': None,
                                'meas_event_field_concept_id': None,
                                'episode_id': episode_id
                            })
                            
                            measurement_id += 1

                            if validated_concept_id == 2100000106:
                                if person_id in self.person_heights:
                                    height = self.person_heights[person_id]
                                    bmi = value / (height ** 2) # BMI = weight(kg) / height(m)²
                                    bmi = round(bmi, 2)

                                    # Create BMI observation record
                                    bmi_observations.append({
                                        'observation_id': observation_id,
                                        'person_id': person_id,
                                        'observation_concept_id': 3038553,
                                        'observation_date': measurement_time.date(),
                                        'observation_datetime': measurement_time,
                                        'observation_type_concept_id': observation_type_id,
                                        'value_as_number': bmi,
                                        'value_as_string': None,
                                        'value_as_concept_id': None,
                                        'qualifier_concept_id': None,
                                        'unit_concept_id': None,  # kg/m² unit
                                        'provider_id': None,
                                        'visit_occurrence_id': visit['visit_occurrence_id'],
                                        'visit_detail_id': None,
                                        'observation_source_value': f"BMI_CALCULATED_{person_id}",
                                        'observation_source_concept_id': 3038553,
                                        'unit_source_value': 'kg/m²',
                                        'qualifier_source_value': None,
                                        'value_source_value': str(bmi),
                                        'observation_event_id': None,
                                        'obs_event_field_concept_id': None,
                                        'episode_id': episode_id
                                    })
                                    observation_id += 1
                                    
        measurement_df = pd.DataFrame(measurement_records)
        bmi_df = pd.DataFrame(bmi_observations)
        self.logger.info(f"Generated {len(measurement_df)} device measurement records")
        self.logger.info(f"Generated {len(bmi_df)} BMI observation records")
        
        return measurement_df, bmi_df
    
    def _get_measurement_concepts_for_device(self, device_concept_id: int) -> List[int]:
        """Map device concept to measurement concepts.
        
        Args:
            device_concept_id: Device concept ID
            
        Returns:
            List of measurement concept IDs for this device
        """
        # Device to measurement mapping based on concept dictionary
        device_mapping = {
            2100000600: [2100000100, 2100000101, 2100000105], 
            2100000601: [4292062, 4268883, 4232915, 4248524],  # BP Cuff -> SBP, DBP
            2100000602: [2100000104],              # Thermometer -> Temp
            2100000603: [2100000106],              # Scale -> Weight
        }
        
        return device_mapping.get(device_concept_id, [])
    
    def _generate_weight_from_bmi(self, config: Dict[str, Any], condition_id: Optional[int], person_id: Optional[int] = None) -> float:
        """Generate weight based on BMI distribution and person's actual height.
        Args:
            config: Measurement configuration with bmi_distribution
            condition_id: Condition concept ID (unused for now, but available for future use)
            person_id: Person ID to look up actual height (if available)
        Returns:
            Generated weight in kg based on BMI distribution
        """
        bmi_distribution = config['bmi_distribution']

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

        # Use person's actual height if available, otherwise generate realistic height
        if person_id is not None and person_id in self.person_heights:
            height = self.person_heights[person_id]
        else:
            # Fallback: Generate realistic height for elderly population
            # Mean age 78, 61.5% female (1.58m), 38.5% male (1.72m)
            # Weighted mean: 1.634m
            height = self.rng.normal(1.634, 0.09)
            # Ensure height is within reasonable bounds for elderly
            height = np.clip(height, 1.4, 1.95)

        # Calculate weight: weight = BMI × height²
        weight = bmi * (height ** 2)
        # Apply safety bounds only for extreme outliers
        # Minimum: BMI 15 × shortest height 1.4m = 29.4kg
        # Maximum: BMI 50 × tallest height 1.95m = 190kg
        weight = max(30, min(200, weight))

        return round(weight, 1)

    def _generate_measurement_value(self, config: Dict[str, Any], condition_id: Optional[int], person_id: Optional[int] = None) -> float:
        """Generate a measurement value based on configuration and condition with realistic bounds.

        Supports baseline weight + condition-specific variation for weight measurements (concept_id 2100000106).
        Args:
            config: Measurement configuration
            condition_id: Condition concept ID for condition-specific baselines
            person_id: Person ID for baseline weight lookup (required for weight measurements)
        Returns:
            Generated measurement value within realistic bounds
        """
        # Check if this is weight measurement - use baseline + variation
        if config.get('id') == 2100000106:
            # Try to use baseline weight + condition-specific variation
            if person_id is not None and person_id in self.person_baseline_weights:
                baseline_weight = self.person_baseline_weights[person_id]
                self.logger.debug(f"Weight generation: person_id={person_id}, baseline={baseline_weight:.1f}kg, condition={condition_id}")

                # Add condition-specific weight variation
                # Based on NHS virtual ward estimations

                if condition_id == 2100000301:  # Heart Failure - acute decompensation
                    # Fluid retention/edema is hallmark of HF decompensation
                    # Patients can gain 3-7kg in days from fluid overload
                    # Weight monitoring is key HF management metric
                    variation = self.rng.normal(4.0, 2.0)  # Mean +4kg, SD 2kg
                    variation = max(0, min(8, variation))  # Clamp to 0-8kg gain

                elif condition_id == 2100000302:  # Infection/Sepsis
                    # Acute illness causes weight loss: dehydration, poor oral intake, catabolism
                    # Typical acute infection weight loss: 2-5kg
                    variation = self.rng.normal(-3.0, 1.5)  # Mean -3kg, SD 1.5kg
                    variation = max(-6, min(0, variation))  # Clamp to -6 to 0kg loss

                else:  # COPD Exacerbation (2100000300) or default
                    # COPD exacerbation: increased work of breathing, poor appetite
                    # Usually stable weight or slight loss
                    # Chronic COPD patients may be cachectic OR obese (sedentary)
                    variation = self.rng.normal(-0.5, 1.5)  # Mean -0.5kg, SD 1.5kg
                    variation = max(-3, min(2, variation))  # Clamp to -3 to +2kg

                episode_weight = baseline_weight + variation

                # Apply safety bounds (must remain physiologically possible)
                episode_weight = max(35, min(200, episode_weight))
                return round(episode_weight, 1)

            # Fallback to old BMI distribution method if no baseline available
            elif 'bmi_distribution' in config:
                return self._generate_weight_from_bmi(config, condition_id, person_id)
            # Fallback to normal generation if no BMI distribution either
            else:
                pass  # Continue to normal generation below

        # Get condition-specific mean or default
        if 'mean_by_condition' in config:
            mean = config['mean_by_condition'].get(condition_id, config['mean_by_condition']['default'])
        else:
            mean = config['mean']

        sd = config['sd']
        bounds = config.get('bounds', [None, None])
        
        # Ensure we have realistic bounds for all measurements
        if bounds and bounds[0] is not None and bounds[1] is not None:
            # Use truncated normal distribution to stay within bounds
            value = self._generate_truncated_normal(mean, sd, bounds[0], bounds[1])
        else:
            # Generate value and apply reasonable clinical bounds based on measurement type
            value = self.rng.normal(mean, sd)
            
            # Apply default clinical bounds if not specified
            if bounds is None or (bounds[0] is None and bounds[1] is None):
                # Apply reasonable bounds based on common measurement ranges
                if mean > 80:  # Likely SpO2 percentage or large values
                    value = max(70, min(100, value))
                elif 50 <= mean < 200:  # Likely heart rate or blood pressure
                    value = max(30, min(250, value))
                elif mean < 50:  # Likely temperature (mean ~37°C)
                    value = max(30, min(45, value))
                else:  # Other measurements
                    value = max(0, value)
            else:
                # Apply partial bounds
                if bounds[0] is not None:
                    value = max(bounds[0], value)
                if bounds[1] is not None:
                    value = min(bounds[1], value)
            
        return round(value, 1)
    
    def _apply_temporal_patterns_parallel(self, episodes: pd.DataFrame, 
                                        observations: pd.DataFrame, 
                                        measurements: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply temporal patterns to episodes in parallel for better performance.
        
        Args:
            episodes: DataFrame with episode data including trajectory_archetype
            observations: DataFrame with observation data
            measurements: DataFrame with measurement data
            
        Returns:
            Tuple of (updated_observations, updated_measurements)
        """
        # Determine optimal number of threads (don't exceed CPU count)
        max_workers = min(multiprocessing.cpu_count(), len(episodes), 8)  # Cap at 8 threads
        
        # Create copies to avoid modifying originals during parallel processing
        updated_observations = observations.copy()
        updated_measurements = measurements.copy()
        
        def process_episode(episode_data):
            """Process a single episode's temporal patterns."""
            episode_id = episode_data['episode_id']
            trajectory = episode_data.get('trajectory_archetype', 'stable')
            
            # Get data for this episode
            episode_obs = observations[observations['episode_id'] == episode_id].copy()
            episode_meas = measurements[measurements['episode_id'] == episode_id].copy()
            
            # Apply patterns if data exists
            if not episode_obs.empty:
                episode_obs = self.apply_temporal_patterns(episode_obs, trajectory, 'observation')
            
            if not episode_meas.empty:
                episode_meas = self.apply_temporal_patterns(episode_meas, trajectory, 'measurement')
            
            return episode_id, episode_obs, episode_meas
        
        # Process episodes in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all episodes for processing
            future_to_episode = {
                executor.submit(process_episode, episode_data): episode_data['episode_id']
                for _, episode_data in episodes.iterrows()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_episode):
                episode_id = future_to_episode[future]
                try:
                    episode_id, episode_obs, episode_meas = future.result()
                    
                    # Update the main dataframes with processed data
                    if not episode_obs.empty:
                        updated_observations.loc[updated_observations['episode_id'] == episode_id] = episode_obs
                    
                    if not episode_meas.empty:
                        updated_measurements.loc[updated_measurements['episode_id'] == episode_id] = episode_meas
                        
                except Exception as e:
                    self.logger.error(f"Error processing episode {episode_id}: {str(e)}")
                    # Continue with other episodes
        
        return updated_observations, updated_measurements
    
    def _apply_temporal_patterns_vectorized(self, episodes: pd.DataFrame, 
                                          observations: pd.DataFrame, 
                                          measurements: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply temporal patterns using vectorized operations for better performance.
        
        Args:
            episodes: DataFrame with episode data including trajectory_archetype
            observations: DataFrame with observation data
            measurements: DataFrame with measurement data
            
        Returns:
            Tuple of (updated_observations, updated_measurements)
        """
        # Create trajectory mapping for quick lookup
        trajectory_map = episodes.set_index('episode_id')['trajectory_archetype'].to_dict()
        
        # Process observations
        if not observations.empty and 'value_as_number' in observations.columns:
            observations = observations.copy()
            # Add trajectory info to observations
            observations['trajectory'] = observations['episode_id'].map(trajectory_map)
            
            # Apply patterns vectorized by trajectory type
            for trajectory in observations['trajectory'].unique():
                if pd.isna(trajectory):
                    continue
                    
                mask = observations['trajectory'] == trajectory
                trajectory_obs = observations[mask]
                
                if not trajectory_obs.empty and not trajectory_obs['value_as_number'].isna().all():
                    # Apply simplified temporal patterns vectorized
                    observations.loc[mask, 'value_as_number'] = self._apply_vectorized_patterns(
                        trajectory_obs, trajectory, 'observation'
                    )
            
            # Remove temporary trajectory column
            observations = observations.drop('trajectory', axis=1)
        
        # Process measurements
        if not measurements.empty and 'value_as_number' in measurements.columns:
            measurements = measurements.copy()
            # Add trajectory info to measurements
            measurements['trajectory'] = measurements['episode_id'].map(trajectory_map)
            for trajectory in measurements['trajectory'].unique():
                if pd.isna(trajectory):
                    continue

                # Group by measurement type within each trajectory
                for concept_id in measurements['measurement_concept_id'].unique():
                    mask = (measurements['trajectory'] == trajectory) & \
                           (measurements['measurement_concept_id'] == concept_id)
                    trajectory_meas = measurements[mask]

                    if not trajectory_meas.empty and not trajectory_meas['value_as_number'].isna().all():
                        # Apply simplified temporal patterns vectorized
                        measurements.loc[mask, 'value_as_number'] = self._apply_vectorized_patterns(
                            trajectory_meas, trajectory, 'measurement'
                        )

            # Remove temporary trajectory column
            measurements = measurements.drop('trajectory', axis=1)
        
        return observations, measurements
    
    def _apply_vectorized_patterns(self, data: pd.DataFrame, trajectory: str, data_type: str) -> pd.Series:
        """Apply temporal patterns using vectorized operations.

        Args:
            data: DataFrame with signal data
            trajectory: Episode trajectory archetype
            data_type: Type of data ('measurement' or 'observation')

        Returns:
            Series with modified values
        """
        # Remember which values are originally NA
        na_mask = data['value_as_number'].isna()

        values = data['value_as_number'].fillna(0)

        # Taking the std of a series with only 1 value causes pandas to produce
        # NaN (numpy returns 0 instead) which causes NaN to be incorrectly stored
        # in the database.  Workaround this by returning the unmodified values.
        if len(values) == 1:
            return values

        # Check if this is temperature (use additive, not multiplicative)
        is_temperature = False
        if data_type == 'measurement' and not data.empty:
            concept_id = data.iloc[0].get('measurement_concept_id')
            is_temperature = (concept_id == 2100000104)  # Temperature concept ID

        if is_temperature:
            # For temperature, additive adjustments (fever during flares)
            # Normal: 36.7°C, Mild fever: +0.5°C, Moderate: +1°C, Severe: +1.5°C
            trajectory_adjustments = {
                'stable': 0.0,
                'flare_mild': 0.5,
                'flare_moderate': 1.0,
                'flare_severe': 1.5,
                'noisy_reporter': 0.0
            }
            adjustment = trajectory_adjustments.get(trajectory, 0.0)
            modified_values = values + adjustment
        else:
            # For other measurements, use multiplicative (worse readings during flares)
            trajectory_multipliers = {
                'stable': 1.0,
                'flare_mild': 1.1,
                'flare_moderate': 1.2,
                'flare_severe': 1.4,
                'noisy_reporter': 1.0  # Will add noise instead
            }
            multiplier = trajectory_multipliers.get(trajectory, 1.0)
            modified_values = values * multiplier
        
        # Add trajectory-specific noise
        if trajectory == 'noisy_reporter':
            # Add more random noise for noisy reporters
            noise = self.rng.normal(0, values.std() * 0.3, len(values))
            modified_values += noise
        elif trajectory in ['flare_moderate', 'flare_severe']:
            # Add some variability for flare trajectories
            noise = self.rng.normal(0, values.std() * 0.1, len(values))
            modified_values += noise
        
        # Apply bounds if this is measurement data
        if data_type == 'measurement' and not data.empty:
            concept_id = data.iloc[0].get('measurement_concept_id')
            if concept_id:
                bounds = self._get_measurement_bounds(concept_id)
                if bounds:
                    modified_values = np.clip(modified_values, bounds[0], bounds[1])
        

        # To avoid spirious values being added to the value_as_number field for non
        # numeric value types - we re-apply the NAs from the original data to the 
        # modified data
        modified_values.where(~na_mask, np.nan, inplace=True)

        return modified_values
    
    def _get_measurement_bounds(self, concept_id: int) -> Optional[Tuple[float, float]]:
        """Get measurement bounds for a concept ID."""
        measurements_config = self.config.get('measurements', [])
        for measurement in measurements_config:
            if measurement.get('id') == concept_id:
                bounds = measurement.get('bounds')
                if bounds and len(bounds) == 2:
                    return (bounds[0], bounds[1])
        return None
    
    def apply_temporal_patterns(self, data: pd.DataFrame, trajectory: str, 
                              data_type: str = 'measurement') -> pd.DataFrame:
        """Apply temporal patterns and noise to signal data.
        
        Args:
            data: DataFrame with measurement or observation data
            trajectory: Episode trajectory archetype
            data_type: Type of data ('measurement' or 'observation')
            
        Returns:
            DataFrame with temporal patterns applied
        """
        self.logger.info(f"Applying temporal patterns for trajectory: {trajectory}")
        
        if data.empty:
            return data
            
        data = data.copy()
        
        # Group by person and concept for pattern application
        if data_type == 'measurement':
            datetime_col = 'measurement_datetime'
            concept_col = 'measurement_concept_id'
            value_col = 'value_as_number'
        else:  # observation
            datetime_col = 'observation_datetime'
            concept_col = 'observation_concept_id'
            value_col = 'value_as_number'
        
        # Apply patterns by person and concept
        for (person_id, concept_id), group in data.groupby(['person_id', concept_col]):
            if group[value_col].isna().all():
                continue  # Skip if all values are null (boolean observations)
                
            # Generate subject-specific baseline drift
            baseline_drift = self._generate_baseline_drift(group, trajectory)
            
            # Add day-of-week noise
            dow_noise = self._generate_day_of_week_noise(group)
            
            # Add time-of-day noise
            tod_noise = self._generate_time_of_day_noise(group)
            
            # Create micro-event shocks based on trajectory
            event_shocks = self._generate_micro_event_shocks(group, trajectory)
            
            # Apply all patterns to the values
            mask = (data['person_id'] == person_id) & (data[concept_col] == concept_id)
            original_values = data.loc[mask, value_col].fillna(0)
            
            modified_values = (original_values + 
                             baseline_drift + 
                             dow_noise + 
                             tod_noise + 
                             event_shocks)
            
            # Apply bounds if available (for measurements)
            if data_type == 'measurement':
                # Get bounds from config
                measurement_configs = {config['id']: config for config in self.config['measurements']}
                if concept_id in measurement_configs:
                    bounds = measurement_configs[concept_id].get('bounds')
                    if bounds:
                        modified_values = np.clip(modified_values, bounds[0], bounds[1])
            
            # Update the data
            data.loc[mask, value_col] = modified_values.round(1)
        
        return data
    
    def _generate_baseline_drift(self, group: pd.DataFrame, trajectory: str) -> np.ndarray:
        """Generate subject-specific baseline with slow drift.
        
        Args:
            group: Data group for a person-concept combination
            trajectory: Episode trajectory archetype
            
        Returns:
            Array of drift values
        """
        n_points = len(group)
        if n_points <= 1:
            return np.zeros(n_points)
        
        # Trajectory-specific drift parameters
        drift_params = {
            'stable': {'magnitude': 0.1, 'volatility': 0.05},
            'flare_mild': {'magnitude': 0.3, 'volatility': 0.1},
            'flare_moderate': {'magnitude': 0.5, 'volatility': 0.15},
            'flare_severe': {'magnitude': 0.8, 'volatility': 0.2},
            'noisy_reporter': {'magnitude': 0.2, 'volatility': 0.3}
        }
        
        params = drift_params.get(trajectory, drift_params['stable'])
        
        # Generate slow linear drift
        time_points = np.arange(n_points)
        linear_drift = self.rng.normal(0, params['magnitude']) * time_points / n_points
        
        # Add random walk component
        random_walk = np.cumsum(self.rng.normal(0, params['volatility'], n_points))
        
        return linear_drift + random_walk
    
    def _generate_day_of_week_noise(self, group: pd.DataFrame) -> np.ndarray:
        """Generate day-of-week noise component.
        
        Args:
            group: Data group for a person-concept combination
            
        Returns:
            Array of day-of-week noise values
        """
        datetime_col = 'measurement_datetime' if 'measurement_datetime' in group.columns else 'observation_datetime'
        
        # Day of week effects (0=Monday, 6=Sunday)
        dow_effects = {
            0: 0.0,   # Monday - baseline
            1: 0.05,  # Tuesday
            2: 0.0,   # Wednesday
            3: -0.05, # Thursday
            4: 0.1,   # Friday
            5: 0.15,  # Saturday - weekend effect
            6: 0.1    # Sunday - weekend effect
        }
        
        noise = []
        for _, row in group.iterrows():
            dow = pd.to_datetime(row[datetime_col]).weekday()
            base_effect = dow_effects.get(dow, 0.0)
            # Add random variation
            noise_value = self.rng.normal(base_effect, 0.1)
            noise.append(noise_value)
            
        return np.array(noise)
    
    def _generate_time_of_day_noise(self, group: pd.DataFrame) -> np.ndarray:
        """Generate time-of-day noise component.
        
        Args:
            group: Data group for a person-concept combination
            
        Returns:
            Array of time-of-day noise values
        """
        datetime_col = 'measurement_datetime' if 'measurement_datetime' in group.columns else 'observation_datetime'
        
        noise = []
        for _, row in group.iterrows():
            hour = pd.to_datetime(row[datetime_col]).hour
            
            # Circadian-like pattern (peak around 14:00, trough around 06:00)
            circadian_effect = 0.1 * np.sin(2 * np.pi * (hour - 6) / 24)
            
            # Add random variation
            noise_value = self.rng.normal(circadian_effect, 0.05)
            noise.append(noise_value)
            
        return np.array(noise)
    
    def _generate_micro_event_shocks(self, group: pd.DataFrame, trajectory: str) -> np.ndarray:
        """Generate micro-event shocks for trajectory realism.
        
        Args:
            group: Data group for a person-concept combination
            trajectory: Episode trajectory archetype
            
        Returns:
            Array of shock values
        """
        n_points = len(group)
        shocks = np.zeros(n_points)
        
        # Trajectory-specific shock parameters
        shock_params = {
            'stable': {'prob': 0.05, 'magnitude': 0.2},
            'flare_mild': {'prob': 0.15, 'magnitude': 0.5},
            'flare_moderate': {'prob': 0.25, 'magnitude': 0.8},
            'flare_severe': {'prob': 0.35, 'magnitude': 1.2},
            'noisy_reporter': {'prob': 0.4, 'magnitude': 0.6}
        }
        
        params = shock_params.get(trajectory, shock_params['stable'])
        
        # Generate random shocks
        for i in range(n_points):
            if self.rng.random() < params['prob']:
                # Shock magnitude with decay
                shock_magnitude = self.rng.normal(0, params['magnitude'])
                
                # Apply shock with exponential decay over next few points
                decay_length = min(5, n_points - i)
                for j in range(decay_length):
                    if i + j < n_points:
                        decay_factor = np.exp(-j / 2)  # Exponential decay
                        shocks[i + j] += shock_magnitude * decay_factor
        
        return shocks
    
    def apply_missingness_and_outliers(self, observations: pd.DataFrame, 
                                     measurements: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply MNAR missingness model and inject outliers.
        
        Args:
            observations: DataFrame with PROM observations
            measurements: DataFrame with device measurements
            
        Returns:
            Tuple of (observations, measurements) with missingness and outliers applied
        """
        self.logger.info("Applying missingness and outlier injection")
        
        # Apply MNAR missingness to surveys (affects all PROMs in that survey)
        observations = self._apply_survey_mnar_missingness(observations)
        
        # Apply device gap patterns to measurements
        measurements = self._apply_device_gap_patterns(measurements)
        
        # Inject outliers
        observations = self._inject_outliers(observations, 'observation')
        measurements = self._inject_outliers(measurements, 'measurement')
        
        return observations, measurements
    
    def _apply_survey_mnar_missingness(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Apply MNAR missingness model based on symptom severity.
        
        Args:
            observations: DataFrame with PROM observations
            
        Returns:
            DataFrame with MNAR missingness applied
        """
        if observations.empty:
            return observations
            
        observations = observations.copy()
        
        # Get missingness model parameters
        mnar_config = self.config.get('missingness_model', {}).get('survey_mnar', {})
        dyspnea_coeff = mnar_config.get('dyspnea_coeff', 0.2)
        severe_offset = mnar_config.get('severe_symptom_offset', -0.1)
        
        # Group by survey to apply missingness at survey level
        surveys_to_remove = set()
        
        for survey_id, survey_group in observations.groupby('survey_conduct_id'):
            # Find dyspnea score for this survey (concept_id 2100000201)
            dyspnea_obs = survey_group[survey_group['observation_concept_id'] == 2100000201]
            
            if not dyspnea_obs.empty and not pd.isna(dyspnea_obs.iloc[0]['value_as_number']):
                dyspnea_score = dyspnea_obs.iloc[0]['value_as_number']
                
                # Calculate missingness probability using logistic model
                base_rate = 0.1  # Base missingness rate
                logit = np.log(base_rate / (1 - base_rate)) + dyspnea_coeff * dyspnea_score
                
                # Counter-effect: severe symptoms can increase reporting
                if dyspnea_score >= 3:
                    logit += severe_offset
                    
                missingness_prob = 1 / (1 + np.exp(-logit))
                
                # Decide if this survey should be missing
                if self.rng.random() < missingness_prob:
                    surveys_to_remove.add(survey_id)
        
        # Remove selected surveys
        if surveys_to_remove:
            observations = observations[~observations['survey_conduct_id'].isin(surveys_to_remove)]
            self.logger.info(f"Applied MNAR missingness: removed {len(surveys_to_remove)} surveys")
        
        return observations
    
    def _apply_device_gap_patterns(self, measurements: pd.DataFrame) -> pd.DataFrame:
        """Apply device gap patterns with weekend uplift.
        
        Args:
            measurements: DataFrame with device measurements
            
        Returns:
            DataFrame with device gaps applied
        """
        if measurements.empty:
            return measurements
            
        measurements = measurements.copy()
        
        # Get device gap parameters
        gap_config = self.config.get('missingness_model', {}).get('device_gaps', {})
        base_rate = gap_config.get('per_metric_base', 0.08)
        weekend_uplift = gap_config.get('weekend_uplift', 0.05)
        
        # Apply gaps per measurement
        measurements_to_remove = []
        
        for idx, row in measurements.iterrows():
            measurement_datetime = pd.to_datetime(row['measurement_datetime'])
            is_weekend = measurement_datetime.weekday() >= 5
            
            # Calculate gap probability
            gap_prob = base_rate
            if is_weekend:
                gap_prob += weekend_uplift
                
            # Decide if this measurement should be missing
            if self.rng.random() < gap_prob:
                measurements_to_remove.append(idx)
        
        # Remove selected measurements
        if measurements_to_remove:
            measurements = measurements.drop(measurements_to_remove)
            self.logger.info(f"Applied device gaps: removed {len(measurements_to_remove)} measurements")
        
        return measurements
    
    def _inject_outliers(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Inject plausible and implausible outliers.
        
        Args:
            data: DataFrame with observations or measurements
            data_type: Type of data ('observation' or 'measurement')
            
        Returns:
            DataFrame with outliers injected
        """
        if data.empty:
            return data
            
        data = data.copy()
        
        # Get outlier rates
        outlier_config = self.config.get('outliers', {})
        plausible_rate = outlier_config.get('plausible_rate', 0.003)
        implausible_rate = outlier_config.get('implausible_rate', 0.0005)
        
        value_col = 'value_as_number'
        
        # Only apply to numeric values
        numeric_mask = data[value_col].notna()
        numeric_data = data[numeric_mask]
        
        if numeric_data.empty:
            return data
        
        n_numeric = len(numeric_data)
        n_plausible = int(n_numeric * plausible_rate)
        n_implausible = int(n_numeric * implausible_rate)
        
        # Select random indices for outliers
        outlier_indices = self.rng.choice(numeric_data.index, 
                                        size=min(n_plausible + n_implausible, n_numeric), 
                                        replace=False)
        
        plausible_indices = outlier_indices[:n_plausible]
        implausible_indices = outlier_indices[n_plausible:n_plausible + n_implausible]
        
        # Inject plausible outliers (extreme but within clinical bounds)
        for idx in plausible_indices:
            original_value = data.loc[idx, value_col]
            concept_id = data.loc[idx, f'{data_type}_concept_id']
            
            # Get bounds for this concept
            bounds = self._get_concept_bounds(concept_id, data_type)
            if bounds:
                # Generate extreme value within bounds
                if self.rng.random() < 0.5:
                    # High outlier
                    outlier_value = self.rng.uniform(bounds[1] * 0.9, bounds[1])
                else:
                    # Low outlier
                    outlier_value = self.rng.uniform(bounds[0], bounds[0] * 1.1)
                    
                data.loc[idx, value_col] = round(outlier_value, 1)
        
        # Inject implausible outliers (outside clinical bounds)
        for idx in implausible_indices:
            original_value = data.loc[idx, value_col]
            concept_id = data.loc[idx, f'{data_type}_concept_id']

            # Get bounds for this concept
            bounds = self._get_concept_bounds(concept_id, data_type)
            if bounds:
                # Generate implausible value outside bounds
                if concept_id == 2100000100:  # SpO2 - CANNOT exceed 100%
                    # For SpO2, implausible means severe hypoxia, never >100%
                    if self.rng.random() < 0.3:
                        outlier_value = self.rng.uniform(50, 70)  # Severe hypoxia
                    else:
                        outlier_value = self.rng.uniform(bounds[0] * 0.7, bounds[0])

                elif concept_id == 4248524:  # Diastolic Sitting BP - max realistic ~140 mmHg
                    # Diastolic BP >130 mmHg is hypertensive emergency (very rare)
                    # Absolute max in living patient: ~140 mmHg
                    if self.rng.random() < 0.5:
                        outlier_value = self.rng.uniform(125, 135)  # Hypertensive crisis
                    else:
                        outlier_value = self.rng.uniform(bounds[0] * 0.5, bounds[0] * 0.8)

                elif concept_id == 4232915:  # Systolic Sitting BP - max realistic ~220 mmHg
                    # Systolic BP >200 mmHg is severe hypertensive emergency
                    if self.rng.random() < 0.5:
                        outlier_value = self.rng.uniform(210, 220)  # Severe hypertensive crisis
                    else:
                        outlier_value = self.rng.uniform(bounds[0] * 0.5, bounds[0] * 0.8)

                else:
                    # For other measurements, allow moderate outliers outside bounds
                    if self.rng.random() < 0.5:
                        # High implausible (max 1.3x, not 2x - more realistic)
                        outlier_value = bounds[1] * self.rng.uniform(1.1, 1.3)
                    else:
                        # Low implausible
                        outlier_value = bounds[0] * self.rng.uniform(0.5, 0.9)

                data.loc[idx, value_col] = round(outlier_value, 1)
        
        if n_plausible + n_implausible > 0:
            self.logger.info(f"Injected {n_plausible} plausible and {n_implausible} implausible outliers in {data_type}s")
        
        return data
    
    def _get_concept_bounds(self, concept_id: int, data_type: str) -> Optional[Tuple[float, float]]:
        """Get clinical bounds for a concept.
        
        Args:
            concept_id: Concept ID
            data_type: Type of data ('observation' or 'measurement')
            
        Returns:
            Tuple of (min, max) bounds or None if not found
        """
        if data_type == 'measurement':
            measurement_configs = {config['id']: config for config in self.config['measurements']}
            if concept_id in measurement_configs:
                return measurement_configs[concept_id].get('bounds')
        elif data_type == 'observation':
            # For PROM observations, get bounds from config
            numeric_proms = {config['id']: config for config in self.config['proms']['numeric']}
            if concept_id in numeric_proms:
                config = numeric_proms[concept_id]
                return (config['min'], config['max'])
        
        return None
    
    def generate(self, episodes: pd.DataFrame, visits: pd.DataFrame = None, device_assignments: pd.DataFrame = None, per_person_obs: pd.DataFrame = None, starting_survey_id: int = 1, starting_observation_id: int = 1, death_lookup: Dict[int, Any] = None, **kwargs: Any) -> Dict[str, pd.DataFrame]:
        """Generate all signal data (surveys, PROMs, and device measurements).

        Args:
            episodes: DataFrame with episode information
            visits: DataFrame with visit_occurrence records for temporal boundaries
            device_assignments: DataFrame with device assignments
            per_person_obs: DataFrame with per-person observations (height, baseline weight)
            starting_survey_id: Starting ID for survey records
            starting_observation_id: Starting ID for observation records
            death_lookup: Optional dictionary mapping person_id to death_date
            **kwargs: Additional arguments

        Returns:
            Dictionary with generated data tables
        """
        self.logger.info("Starting signal generation")
        self.person_heights = {}
        self.person_baseline_weights = {}

        if per_person_obs is not None and not per_person_obs.empty:
            # Filter for height observations (concept_id 3036277)
            height_obs = per_person_obs[per_person_obs['observation_concept_id'] == 3036277]
            if not height_obs.empty:
                self.person_heights = dict(zip(
                    height_obs['person_id'],
                    height_obs['value_as_number']
                ))
                self.logger.info(f"Loaded height data for {len(self.person_heights)} persons")
            else:
                self.logger.warning("No height observations found in per_person_obs")

            # Filter for baseline weight observations (concept_id 2100000106)
            # These are the weights generated from BMI distribution in population.py
            weight_obs = per_person_obs[per_person_obs['observation_concept_id'] == 2100000106]
            # Additional filter: only WEIGHT_ source values (not BMI observations which also use 2100000106)
            if not weight_obs.empty and 'observation_source_value' in weight_obs.columns:
                weight_obs = weight_obs[weight_obs['observation_source_value'].astype(str).str.startswith('WEIGHT_')]

            if not weight_obs.empty:
                self.person_baseline_weights = dict(zip(
                    weight_obs['person_id'],
                    weight_obs['value_as_number']
                ))
                self.logger.info(f"Loaded baseline weight data for {len(self.person_baseline_weights)} persons")
            else:
                self.logger.warning("No baseline weight observations found in per_person_obs")
        else:
            self.logger.info("No per_person_obs provided - BMI calculations will not be performed")
        # Generate survey schedule
        surveys = self.generate_survey_schedule(episodes, starting_survey_id, death_lookup)

        # Generate PROM observations
        observations = self.generate_prom_observations(surveys, episodes, per_person_obs, starting_observation_id)

        # Generate device measurements (requires visits for temporal alignment)
        measurements = pd.DataFrame()  # Empty by default
        bmi_observations = pd.DataFrame()
        if visits is not None and device_assignments is not None:
            # Calculate next observation ID based on PROM observations
            next_observation_id = observations['observation_id'].max() + 1 if not observations.empty else starting_observation_id
            measurements, bmi_observations = self.generate_device_measurements(
                episodes, visits, device_assignments, starting_observation_id=next_observation_id, death_lookup=death_lookup
            )
        elif device_assignments is not None:
            self.logger.warning("Device assignments provided but no visits - measurements will not be generated")
        else:
            self.logger.info("No device assignments provided - skipping measurement generation")
        if not bmi_observations.empty:
            observations = pd.concat([observations, bmi_observations], ignore_index=False)
            observations = observations.reset_index(drop=True)
            self.logger.info(f"Merged {len(bmi_observations)} BMI observations with PROM observations")
        
        # Apply temporal patterns (requires trajectory information from episodes)
        if 'trajectory_archetype' in episodes.columns:
            observations, measurements = self._apply_temporal_patterns_vectorized(
                episodes, observations, measurements
            )
        
        # Apply missingness and outliers
        observations, measurements = self.apply_missingness_and_outliers(observations, measurements)
        
        self.logger.info("Signal generation completed")
        
        return {
            'survey_conduct': surveys,
            'observation': observations,
            'measurement': measurements
        }