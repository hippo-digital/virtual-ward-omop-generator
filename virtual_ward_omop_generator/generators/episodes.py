"""Episode manager for virtual ward episodes and trajectories."""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..models.base import BaseGenerator
from ..exceptions import ConfigurationError
from ..utils.temporal_coordinator import TemporalCoordinator


class EpisodeManager(BaseGenerator):
    """Manages virtual ward episodes and trajectories."""
    
    def __init__(self, config: Dict[str, Any], concept_dict: pd.DataFrame, rng: np.random.Generator):
        """Initialize the episode manager with temporal coordination.
        
        Args:
            config: Configuration dictionary
            concept_dict: Concept dictionary DataFrame
            rng: Random number generator
        """
        super().__init__(config, concept_dict, rng)
        self.temporal_coordinator = TemporalCoordinator()
    
    def _validate_config(self) -> None:
        """Validate episode manager configuration."""
        if 'dataset' not in self.config:
            raise ConfigurationError("Missing 'dataset' section in configuration")
        
        dataset_config = self.config['dataset']
        
        # Validate episodes per person
        if 'episodes_per_person' not in dataset_config:
            raise ConfigurationError("Missing 'episodes_per_person' in dataset configuration")
        
        episodes_config = dataset_config['episodes_per_person']
        if 'min' not in episodes_config or 'max' not in episodes_config:
            raise ConfigurationError("episodes_per_person must have 'min' and 'max' values")
        
        if episodes_config['min'] < 1 or episodes_config['max'] < episodes_config['min']:
            raise ConfigurationError("Invalid episodes_per_person range")
        
        # Validate episode length distribution
        if 'episode_length_days' not in dataset_config:
            raise ConfigurationError("Missing 'episode_length_days' in dataset configuration")
        
        length_config = dataset_config['episode_length_days']
        if 'modes' not in length_config or 'probs' not in length_config:
            raise ConfigurationError("episode_length_days must have 'modes' and 'probs'")
        
        if len(length_config['modes']) != len(length_config['probs']):
            raise ConfigurationError("episode_length_days modes and probs must have same length")
        
        if abs(sum(length_config['probs']) - 1.0) > 0.01:
            raise ConfigurationError("episode_length_days probabilities must sum to 1.0")
        
        # Validate condition mix
        if 'condition_mix' not in dataset_config:
            raise ConfigurationError("Missing 'condition_mix' in dataset configuration")
        
        condition_mix = dataset_config['condition_mix']
        if not isinstance(condition_mix, dict) or len(condition_mix) == 0:
            raise ConfigurationError("condition_mix must be a non-empty dictionary")
        
        # Validate archetype mix
        if 'episode_archetype_mix' not in dataset_config:
            raise ConfigurationError("Missing 'episode_archetype_mix' in dataset configuration")
        
        archetype_mix = dataset_config['episode_archetype_mix']
        valid_archetypes = ['stable', 'flare_mild', 'flare_moderate', 'flare_severe', 'noisy_reporter']
        
        # Check that all specified archetypes are valid
        for archetype in archetype_mix.keys():
            if archetype not in valid_archetypes:
                raise ConfigurationError(f"Invalid archetype '{archetype}' in episode_archetype_mix. Valid options: {valid_archetypes}")
        
        # Require at least 'stable' archetype
        if 'stable' not in archetype_mix:
            raise ConfigurationError("'stable' archetype is required in episode_archetype_mix")
        
        if abs(sum(archetype_mix.values()) - 1.0) > 0.01:
            raise ConfigurationError("episode_archetype_mix probabilities must sum to 1.0")
        
        # Validate device assignment
        if 'device_assignment' not in self.config:
            raise ConfigurationError("Missing 'device_assignment' section in configuration")
        
        # Validate required concept IDs
        if 'episode_and_visit_concepts' not in self.config:
            raise ConfigurationError("Missing 'episode_and_visit_concepts' section in configuration")
        
        if 'types' not in self.config:
            raise ConfigurationError("Missing 'types' section in configuration")
    
    def generate(
            self, 
            persons_df: pd.DataFrame, 
            observation_periods_df: pd.DataFrame, 
            starting_survey_id: int,
            starting_observation_id: int, 
            **kwargs: Any) -> Dict[str, pd.DataFrame]:
        """Generate episode data including episodes, conditions, and device exposures.
        
        Args:
            persons_df: DataFrame containing person data
            observation_periods_df: DataFrame containing observation periods
            
        Returns:
            Dictionary containing episode-related DataFrames
        """
        # Generate episodes using temporal coordination
        episodes_df = self._generate_episodes(persons_df, observation_periods_df)
        self._log_generation_stats(episodes_df, "episode")
        
        # Update observation periods to ensure they span all episodes
        updated_observation_periods_df = self._update_observation_periods_for_episodes(
            observation_periods_df, episodes_df
        )
        
        # Assign trajectory archetypes
        episodes_df = self._assign_trajectory_archetypes(episodes_df)
        
        # Assign index conditions
        episodes_df = self._assign_index_conditions(episodes_df)
        
        # Generate condition occurrences
        condition_occurrences_df = self._generate_condition_occurrences(episodes_df)
        self._log_generation_stats(condition_occurrences_df, "condition_occurrence")
        
        # Generate device exposures
        device_exposures_df = self._generate_device_exposures(episodes_df)
        self._log_generation_stats(device_exposures_df, "device_exposure")
        
        # Generate per-episode clinical observations
        per_episode_df = self._generate_per_episode_observations(episodes_df, starting_survey_id, starting_observation_id)
        self._log_generation_stats(per_episode_df, "per_episode_observations")

        # Generate Condition Era's based on episodes
        condition_era_df = episodes_df.groupby(['person_id', 'index_condition_concept_id']).agg({
            'episode_start_date': 'min',
            'episode_end_date': 'max'
        }).reset_index().assign(
            condition_era_id=lambda df: range(1, len(df)+1),
        ).rename(columns={
            'index_condition_concept_id': 'condition_concept_id',
            'episode_start_date': 'condition_era_start_date',
            'episode_end_date': 'condition_era_end_date'
        })

        return {
            'episode': episodes_df,
            'condition_occurrence': condition_occurrences_df,
            'device_exposure': device_exposures_df,
            'observation_period': updated_observation_periods_df,  # Include updated observation periods
            'per_episode_obs': per_episode_df,
            'condition_era': condition_era_df
        }
    
    def _generate_per_episode_observations(
            self, 
            episiodes_df: pd.DataFrame, 
            starting_survey_id: int, 
            starting_observation_id: int
        ) -> pd.DataFrame:
        "Generate per episode observations based on logic from SignalGenerator"


        if not self.config['per_episode_obs']:
            return pd.DataFrame()
        
        observations = []
        observation_type_id = self.config['types']['observation_type_clinician_reported']

        survey_id = starting_survey_id
        observation_id = starting_observation_id

        for _, row in episiodes_df.iterrows():
            for var in self.config['per_episode_obs'].get('numeric', []):
                val = self.rng.normal(var['baseline_mean'], var['baseline_sd'])
                if val < var['min'] or val > var['max']:
                    val = np.clip(val, var['min'], var['max'])

                val = round(val)

                concept_id = var['id']
                observations.append({
                    'observation_id': observation_id,
                    'person_id': row.person_id,
                    'observation_concept_id': concept_id,
                    'observation_date': row.episode_start_datetime.date(),
                    'observation_datetime': row.episode_start_datetime,
                    'observation_type_concept_id': observation_type_id,
                    'value_as_number': val,
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
                    'value_source_value': str(val),
                    'observation_event_id': None,
                    'obs_event_field_concept_id': None,
                    'survey_conduct_id': survey_id,
                    'episode_id': row.episode_id
                })
                observation_id += 1
                
            for var in self.config['per_episode_obs'].get('boolean', []):
                raise NotImplementedError()
        
        survey_id += 1

        return pd.DataFrame(observations)

    def _generate_episodes(self, persons_df: pd.DataFrame, observation_periods_df: pd.DataFrame) -> pd.DataFrame:
        """Generate episodes with 1-2 per person distribution and tri-modal length distribution.
        
        Args:
            persons_df: DataFrame containing person data
            observation_periods_df: DataFrame containing observation periods
            
        Returns:
            DataFrame with episode data
        """
        episodes_config = self.config['dataset']['episodes_per_person']
        length_config = self.config['dataset']['episode_length_days']
        episode_concept_id = self.config['episode_and_visit_concepts']['episode']
        
        episodes_list = []
        episode_id = 1
        
        for _, person in persons_df.iterrows():
            person_id = person['person_id']

            # Get observation period for this person
            obs_period = observation_periods_df[observation_periods_df['person_id'] == person_id].iloc[0]
            obs_start = obs_period['observation_period_start_date']
            obs_end = obs_period['observation_period_end_date']
            
            # Determine number of episodes for this person (1-2)
            num_episodes = self.rng.integers(episodes_config['min'], episodes_config['max'] + 1)
            
            # Generate episodes for this person
            for episode_num in range(num_episodes):
                # Sample episode length from tri-modal distribution
                # Normalize probabilities to handle floating point precision issues
                probs = np.array(length_config['probs'])
                if abs(probs.sum() - 1.0) > 0.1:  # 10% tolerance
                    self.logger.warning(f"Episode length probabilities sum to {probs.sum()}, normalizing")
                probs = probs / probs.sum()
                episode_length = self.rng.choice(length_config['modes'], p=probs)
                
                # Generate episode start datetime within observation period
                # Leave room for the episode to fit within the observation period
                episode_length = int(episode_length)  # Convert numpy int to Python int
                max_start_date = obs_end - timedelta(days=episode_length)
                if max_start_date < obs_start:
                    # If episode is too long for observation period, adjust
                    max_start_date = obs_start
                    episode_length = min(episode_length, (obs_end - obs_start).days)
                
                # Generate random start date within valid range
                date_range_days = (max_start_date - obs_start).days
                if date_range_days > 0:
                    start_offset_days = int(self.rng.integers(0, date_range_days + 1))
                else:
                    start_offset_days = 0
                
                episode_start_date = obs_start + timedelta(days=start_offset_days)
                
                # Generate start time (random time during the day)
                start_hour = self.rng.integers(0, 24)
                start_minute = self.rng.integers(0, 60)
                start_second = self.rng.integers(0, 60)
                
                episode_start_datetime = datetime.combine(
                    episode_start_date,
                    datetime.min.time().replace(hour=start_hour, minute=start_minute, second=start_second)
                )
                
                # Calculate end datetime
                episode_end_datetime = episode_start_datetime + timedelta(days=int(episode_length))
                
                episode_record = {
                    'episode_id': episode_id,
                    'person_id': person_id,
                    'episode_concept_id': episode_concept_id,
                    'episode_start_date': episode_start_datetime.date(),
                    'episode_start_datetime': episode_start_datetime,
                    'episode_end_date': episode_end_datetime.date(),
                    'episode_end_datetime': episode_end_datetime,
                    'episode_number': episode_num + 1,
                    'episode_length_days': episode_length,
                    'episode_type_concept_id': 32842 # EHR Referral Record
                }

                episodes_list.append(episode_record)
                
                episode_id += 1
        
        return pd.DataFrame(episodes_list)
    
    def _assign_trajectory_archetypes(self, episodes_df: pd.DataFrame) -> pd.DataFrame:
        """Assign trajectory archetypes to episodes with archetype-specific patterns.
        
        Args:
            episodes_df: DataFrame with episode data
            
        Returns:
            DataFrame with trajectory archetypes and patterns assigned
        """
        archetype_mix = self.config['dataset']['episode_archetype_mix']
        
        # Extract archetype names and probabilities
        archetypes = list(archetype_mix.keys())
        probabilities = list(archetype_mix.values())
        
        # Assign archetypes to episodes
        # Normalize probabilities to handle floating point precision issues
        probabilities = np.array(probabilities)
        if abs(probabilities.sum() - 1.0) > 0.1:  # 10% tolerance
            self.logger.warning(f"Archetype probabilities sum to {probabilities.sum()}, normalizing")
        probabilities = probabilities / probabilities.sum()
        assigned_archetypes = self.rng.choice(archetypes, size=len(episodes_df), p=probabilities)
        
        episodes_df = episodes_df.copy()
        episodes_df['trajectory_archetype'] = assigned_archetypes
        
        # Assign archetype-specific signal patterns and intervention rates
        episodes_df = self._assign_archetype_patterns(episodes_df)
        
        # Schedule micro-events within trajectories
        episodes_df = self._schedule_micro_events(episodes_df)
        
        return episodes_df
    
    def _assign_archetype_patterns(self, episodes_df: pd.DataFrame) -> pd.DataFrame:
        """Assign archetype-specific signal patterns and intervention rates.
        
        Args:
            episodes_df: DataFrame with episode data and archetypes
            
        Returns:
            DataFrame with archetype patterns assigned
        """
        episodes_df = episodes_df.copy()
        
        # Define archetype-specific characteristics
        archetype_patterns = {
            'stable': {
                'signal_variance_multiplier': 0.7,  # Lower variance in signals
                'intervention_probability': 0.05,   # Very low intervention rate
                'deterioration_probability': 0.1,   # Low chance of deterioration
                'recovery_rate': 0.9                # High recovery rate
            },
            'flare_mild': {
                'signal_variance_multiplier': 1.2,  # Slightly higher variance
                'intervention_probability': 0.4,    # Moderate intervention rate (1 outpatient intervention)
                'deterioration_probability': 0.3,   # Moderate deterioration
                'recovery_rate': 0.8                # Good recovery rate
            },
            'flare_moderate': {
                'signal_variance_multiplier': 1.5,  # Higher variance
                'intervention_probability': 0.6,    # Higher intervention rate
                'deterioration_probability': 0.5,   # Moderate-high deterioration
                'recovery_rate': 0.6                # Moderate recovery rate
            },
            'flare_severe': {
                'signal_variance_multiplier': 2.0,  # High variance
                'intervention_probability': 0.8,    # High intervention rate
                'deterioration_probability': 0.7,   # High deterioration
                'recovery_rate': 0.4                # Lower recovery rate
            },
            'noisy_reporter': {
                'signal_variance_multiplier': 2.5,  # Very high variance (inconsistent reporting)
                'intervention_probability': 0.2,    # Low intervention rate despite noise
                'deterioration_probability': 0.2,   # Low actual deterioration
                'recovery_rate': 0.7                # Moderate recovery rate
            }
        }
        
        # Assign patterns to each episode
        for archetype, patterns in archetype_patterns.items():
            mask = episodes_df['trajectory_archetype'] == archetype
            for pattern_key, pattern_value in patterns.items():
                episodes_df.loc[mask, pattern_key] = pattern_value
        
        return episodes_df
    
    def _schedule_micro_events(self, episodes_df: pd.DataFrame) -> pd.DataFrame:
        """Schedule 1-3 micro-events within trajectories for realism.
        
        Args:
            episodes_df: DataFrame with episode data and archetype patterns
            
        Returns:
            DataFrame with micro-events scheduled
        """
        episodes_df = episodes_df.copy()
        
        # Define micro-event types and their effects
        micro_event_types = [
            'viral_day',        # Temporary worsening of symptoms
            'missed_meds_day',  # Temporary deterioration due to missed medication
            'good_day',         # Temporary improvement
            'stress_day',       # Stress-related symptom increase
            'weather_day'       # Weather-related symptom changes
        ]
        
        micro_events_list = []
        
        for idx, episode in episodes_df.iterrows():
            episode_length = episode['episode_length_days']
            archetype = episode['trajectory_archetype']
            
            # Determine number of micro-events based on archetype and episode length
            if archetype == 'stable':
                max_events = 1
            elif archetype in ['flare_mild', 'noisy_reporter']:
                max_events = 2
            else:  # flare_moderate, flare_severe
                max_events = 3
            
            # Adjust for episode length
            if episode_length <= 7:
                max_events = min(max_events, 1)
            elif episode_length <= 14:
                max_events = min(max_events, 2)
            
            num_events = self.rng.integers(1, max_events + 1)
            
            # Schedule events at random times during the episode
            event_days = self.rng.choice(
                range(episode_length), 
                size=min(num_events, episode_length), 
                replace=False
            )
            
            episode_events = []
            for day in sorted(event_days):
                event_type = self.rng.choice(micro_event_types)
                event_datetime = episode['episode_start_datetime'] + timedelta(days=int(day))
                
                # Add some random hours to the event time
                event_datetime += timedelta(hours=int(self.rng.integers(0, 24)))
                
                episode_events.append({
                    'event_type': event_type,
                    'event_datetime': event_datetime,
                    'event_day': day
                })
            
            micro_events_list.append(episode_events)
        
        # Store micro-events as a list column (will be used by signal generators)
        episodes_df['micro_events'] = micro_events_list
        
        return episodes_df
    
    def _assign_index_conditions(self, episodes_df: pd.DataFrame) -> pd.DataFrame:
        """Assign index conditions based on specified mix (COPD 45%, HF 45%, Post-op 10%).
        
        Args:
            episodes_df: DataFrame with episode data
            
        Returns:
            DataFrame with index conditions assigned
        """
        condition_mix = self.config['dataset']['condition_mix']
        
        # Extract condition concept IDs and probabilities
        condition_ids = list(condition_mix.keys())
        probabilities = [condition_mix[cid] for cid in condition_ids]  # Probabilities are already in decimal form
        
        # Assign conditions to episodes
        # Normalize probabilities to handle floating point precision issues
        probabilities = np.array(probabilities)
        if abs(probabilities.sum() - 1.0) > 0.1:  # 10% tolerance
            self.logger.warning(f"Condition probabilities sum to {probabilities.sum()}, normalizing")
        probabilities = probabilities / probabilities.sum()
        assigned_conditions = self.rng.choice(condition_ids, size=len(episodes_df), p=probabilities)
        
        episodes_df = episodes_df.copy()
        episodes_df['index_condition_concept_id'] = assigned_conditions
        episodes_df['episode_object_concept_id'] = assigned_conditions
        
        return episodes_df
    
    def _update_observation_periods_for_episodes(self, observation_periods_df: pd.DataFrame, episodes_df: pd.DataFrame) -> pd.DataFrame:
        """Update observation periods to ensure they span all episodes for each person.
        
        Args:
            observation_periods_df: DataFrame containing observation periods
            episodes_df: DataFrame containing episodes
            
        Returns:
            Updated DataFrame with observation periods that span all episodes
        """
        updated_periods = observation_periods_df.copy()
        
        # Group episodes by person to find min/max dates
        episode_bounds = episodes_df.groupby('person_id').agg({
            'episode_start_datetime': 'min',
            'episode_end_datetime': 'max'
        }).reset_index()
        
        # Update observation periods to span all episodes
        for _, bounds in episode_bounds.iterrows():
            person_id = bounds['person_id']
            episode_start = bounds['episode_start_datetime']
            episode_end = bounds['episode_end_datetime']
            
            # Find the observation period for this person
            person_mask = updated_periods['person_id'] == person_id
            
            if person_mask.any():
                # Update start date if episode starts before observation period
                current_start = pd.to_datetime(updated_periods.loc[person_mask, 'observation_period_start_date'].iloc[0])
                if episode_start < current_start:
                    updated_periods.loc[person_mask, 'observation_period_start_date'] = episode_start.date()
                
                # Update end date if episode ends after observation period
                current_end = pd.to_datetime(updated_periods.loc[person_mask, 'observation_period_end_date'].iloc[0])
                if episode_end.date() > current_end.date():
                    # Extend observation period to include the full day when episode ends
                    updated_periods.loc[person_mask, 'observation_period_end_date'] = episode_end.date()
                elif episode_end.date() == current_end.date() and episode_end.time() > current_end.time():
                    # If episode ends on the same day but later in the day, extend to end of day
                    updated_periods.loc[person_mask, 'observation_period_end_date'] = episode_end.date()
        
        return updated_periods
    
    def _generate_condition_occurrences(self, episodes_df: pd.DataFrame) -> pd.DataFrame:
        """Generate condition_occurrence records linked to episodes.
        
        Args:
            episodes_df: DataFrame with episode data including index conditions
            
        Returns:
            DataFrame with condition occurrence records
        """
        condition_type_concept_id = self.config['types']['condition_type_index']
        
        condition_occurrences = []
        condition_occurrence_id = 1
        
        for _, episode in episodes_df.iterrows():
            # Create condition occurrence starting at or before episode start
            # Condition typically starts before or at the beginning of virtual ward episode
            max_days_before = 30  # Condition can start up to 30 days before episode
            days_before = int(self.rng.integers(0, max_days_before + 1))
            
            condition_start_datetime = episode['episode_start_datetime'] - timedelta(days=days_before)
            
            # Condition typically continues through the episode (chronic conditions)
            # End datetime can be None (ongoing) or extend beyond episode
            condition_end_datetime = None  # Most conditions are ongoing
            
            condition_occurrences.append({
                'condition_occurrence_id': condition_occurrence_id,
                'person_id': episode['person_id'],
                'condition_concept_id': episode['index_condition_concept_id'],
                'condition_start_date': condition_start_datetime.date(),
                'condition_start_datetime': condition_start_datetime,
                'condition_end_date': condition_end_datetime.date() if condition_end_datetime else None,
                'condition_end_datetime': condition_end_datetime,
                'condition_type_concept_id': condition_type_concept_id,
                'condition_status_concept_id': None,  # Not specified in requirements
                'stop_reason': None,
                'provider_id': None,
                'visit_occurrence_id': None,
                'visit_detail_id': None,
                'condition_source_value': None,
                'condition_source_concept_id': None,
                'condition_status_source_value': None,
                'episode_id': episode['episode_id']
            })
            
            condition_occurrence_id += 1
        
        return pd.DataFrame(condition_occurrences)
    
    def _generate_device_exposures(self, episodes_df: pd.DataFrame) -> pd.DataFrame:
        """Generate device exposures based on condition type and device assignment probabilities.
        
        Args:
            episodes_df: DataFrame with episode data including index conditions
            
        Returns:
            DataFrame with device exposure records
        """
        device_assignment = self.config['device_assignment']
        
        device_exposures = []
        device_exposure_id = 1
        
        # Device concept ID mapping
        # Maps measurement concept IDs to device concept IDs
        device_concept_mapping = self.get_device_mapping()
        
        for _, episode in episodes_df.iterrows():
            condition_id = episode['index_condition_concept_id']
            
            # Get device assignment probabilities for this condition
            if condition_id in device_assignment:
                assignment_probs = device_assignment[condition_id]
                
                # Assign devices based on probabilities
                assigned_devices = set()
                assigned_measurements = []  # Track which measurements are assigned
                
                for measurement_id, prob in assignment_probs.items():
                    if self.rng.random() < prob:
                        device_id = device_concept_mapping.get(measurement_id)
                        if device_id:
                            assigned_devices.add(device_id)
                            assigned_measurements.append(measurement_id)
                
                # Create device exposure records
                for device_id in assigned_devices:
                    # Generate unique device identifier
                    unique_device_id = f"DEV_{episode['person_id']}_{device_id}_{episode['episode_id']}"
                    
                    device_exposures.append({
                        'device_exposure_id': device_exposure_id,
                        'person_id': episode['person_id'],
                        'device_concept_id': device_id,
                        'device_exposure_start_date': episode['episode_start_datetime'].date(),
                        'device_exposure_start_datetime': episode['episode_start_datetime'],
                        'device_exposure_end_date': episode['episode_end_datetime'].date(),
                        'device_exposure_end_datetime': episode['episode_end_datetime'],
                        'device_type_concept_id': self.config['types']['measurement_type_patient_device'],
                        'unique_device_id': unique_device_id,
                        'production_id': None,
                        'quantity': 1,
                        'provider_id': None,
                        'visit_occurrence_id': None,
                        'visit_detail_id': None,
                        'device_source_value': None,
                        'device_source_concept_id': None,
                        'unit_concept_id': None,
                        'unit_source_value': None,
                        'unit_source_concept_id': None,
                        'episode_id': episode['episode_id']
                    })
                    
                    device_exposure_id += 1
                
        # Store assigned measurements for later use by signal generators
        # Initialize the column first
        if 'assigned_measurements' not in episodes_df.columns:
            episodes_df['assigned_measurements'] = [[] for _ in range(len(episodes_df))]
        
        # Update assigned measurements for each episode
        for _, episode in episodes_df.iterrows():
            condition_id = episode['index_condition_concept_id']
            assigned_measurements = []
            
            if condition_id in device_assignment:
                assignment_probs = device_assignment[condition_id]
                for measurement_id, prob in assignment_probs.items():
                    if self.rng.random() < prob:
                        assigned_measurements.append(measurement_id)
            
            # Update the specific row
            idx = episodes_df[episodes_df['episode_id'] == episode['episode_id']].index[0]
            episodes_df.at[idx, 'assigned_measurements'] = assigned_measurements
        
        return pd.DataFrame(device_exposures)
    
    def get_device_mapping(self) -> Dict[int, int]:
        """Get the mapping from measurement concept IDs to device concept IDs.
        
        Returns:
            Dictionary mapping measurement concept ID to device concept ID
        """
        return {
            2100000100: 2100000600,  # SpO2 -> Pulse Oximeter
            2100000101: 2100000600,  # HR -> Pulse Oximeter (same device)
            4292062: 2100000601,     # Systolic Standing -> Blood Pressure Cuff
            4268883: 2100000601,     # Diastolic Standing (same device)
            4232915: 2100000601,     # Systolic Sitting (same device)
            4248524: 2100000601,     # Diastolic Sitting (same device)
            2100000104: 2100000602,  # Temperature -> Digital Thermometer
            2100000105: 2100000600,  # RR -> Pulse Oximeter (some devices measure RR)
            2100000106: 2100000603,  # Weight -> Body Weight Scale
        }
    
    def get_condition_device_assignments(self, condition_id: int) -> List[int]:
        """Get the list of measurement concept IDs assigned to a condition.
        
        Args:
            condition_id: Condition concept ID
            
        Returns:
            List of measurement concept IDs that should be assigned to this condition
        """
        device_assignment = self.config.get('device_assignment', {})
        if condition_id in device_assignment:
            return list(device_assignment[condition_id].keys())
        return []