"""Intervention engine for generating clinical interventions based on PROM and device signals."""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from ..models.base import BaseGenerator
from ..exceptions import DataGenerationError

logger = logging.getLogger(__name__)


class InterventionEngine(BaseGenerator):
    """Generates clinical interventions based on trigger evaluation logic."""
    
    def __init__(self, config: Dict[str, Any], concept_dict: pd.DataFrame, 
                 random_generator: np.random.Generator) -> None:
        """Initialize intervention engine.
        
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
        
        # Initialize ID counters for unique record IDs
        self.drug_id_counter = 1
        self.procedure_id_counter = 1
        
        # Cache concept mappings for performance
        self._cache_concept_mappings()
        
    def _validate_config(self) -> None:
        """Validate intervention engine configuration."""
        if 'interventions' not in self.config:
            raise DataGenerationError("Missing required config section: interventions")
            
        interventions_config = self.config['interventions']
        valid_interventions = ['urgent_review', 'conveyance_ed', 'start_antibiotic', 
                              'increase_diuretic', 'start_oxygen', 'unplanned_admission']
        
        # Check that all specified interventions are valid
        for intervention in interventions_config.keys():
            if intervention not in valid_interventions:
                raise DataGenerationError(f"Invalid intervention '{intervention}'. Valid options: {valid_interventions}")
        
        # Require at least 'urgent_review' intervention
        if 'urgent_review' not in interventions_config:
            raise DataGenerationError("'urgent_review' intervention is required")
                
        self.logger.info("Intervention engine configuration validated")
        
    def _cache_concept_mappings(self) -> None:
        """Cache concept ID mappings for performance."""
        self.concept_map = {}
        for _, row in self.concept_dict.iterrows():
            self.concept_map[row['concept_id']] = {
                'name': row['concept_name'],
                'domain': row['domain_id']
            }
    
    def generate(self, episodes: pd.DataFrame, observations: pd.DataFrame, 
                measurements: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate intervention data (drug exposures and procedures).
        
        This is the main entry point for the intervention engine that implements
        the abstract generate method from BaseGenerator.
        
        Args:
            episodes: DataFrame with episode information
            observations: DataFrame with PROM observations
            measurements: DataFrame with device measurements
            
        Returns:
            Dictionary containing 'drug_exposure' and 'procedure_occurrence' DataFrames
        """
        drug_exposure_df, procedure_occurrence_df = self.apply_intervention_logic(episodes, observations, measurements)

        # compute drug_era's from the provided drug_exposures
        ingredient_mapping = self.config['drug_ingredient_mapping']
        drug_eras = []
        for _, episode in episodes.iterrows():
            exposures = drug_exposure_df[
                (drug_exposure_df.person_id == episode.person_id) 
                & (drug_exposure_df.drug_exposure_start_datetime >= episode.episode_start_datetime)
                & (drug_exposure_df.drug_exposure_end_datetime <= episode.episode_end_datetime)
            ].groupby('drug_concept_id').agg({ 'drug_exposure_start_datetime': 'min', 'drug_exposure_end_datetime': 'max' })

            if not exposures.empty:
                for drug_concept_id, exposure in exposures.iterrows():
                    drug_eras.append({
                        'drug_era_id': len(drug_eras) + 1,
                        'person_id': episode.person_id,
                        'drug_concept_id': ingredient_mapping[drug_concept_id], # This MUST be an ingredient
                        'drug_era_start_date': exposure.drug_exposure_start_datetime.date(),
                        'drug_era_start_datetime': exposure.drug_exposure_start_datetime,
                        'drug_era_end_date': exposure.drug_exposure_end_datetime.date(),
                        'drug_era_end_datetime': exposure.drug_exposure_end_datetime,
                    })

        return {
            'drug_exposure': drug_exposure_df,
            'drug_era': pd.DataFrame(drug_eras),
            'procedure_occurrence': procedure_occurrence_df
        }
    
    def evaluate_triggers(self, patient_data: pd.DataFrame, episode_id: int, 
                         current_time: datetime) -> List[Dict[str, Any]]:
        """Evaluate intervention triggers for a patient at a specific time.
        
        Args:
            patient_data: DataFrame with patient's PROM and measurement data
            episode_id: Episode ID being evaluated
            current_time: Current evaluation timestamp
            
        Returns:
            List of triggered interventions with metadata
        """
        triggered_interventions = []
        interventions_config = self.config['interventions']
        
        # Evaluate each intervention type
        for intervention_name, intervention_config in interventions_config.items():
            if intervention_name in ['urgent_review', 'conveyance_ed', 'start_antibiotic', 
                                   'increase_diuretic', 'start_oxygen']:
                
                trigger_result = self._evaluate_intervention_triggers(
                    intervention_name, intervention_config, patient_data, current_time
                )
                
                if trigger_result['triggered']:
                    # Apply adherence probability
                    adherence = intervention_config.get('adherence', intervention_config.get('prob', 0.8))
                    
                    if self.rng.random() < adherence:
                        triggered_interventions.append({
                            'intervention_type': intervention_name,
                            'concept_id': intervention_config['concept_id'],
                            'episode_id': episode_id,
                            'trigger_time': current_time,
                            'trigger_reasons': trigger_result['reasons'],
                            'config': intervention_config
                        })
                        
                        self.logger.debug(f"Triggered {intervention_name} for episode {episode_id}")
        
        return triggered_interventions
    
    def _evaluate_intervention_triggers(self, intervention_name: str, config: Dict[str, Any], 
                                      patient_data: pd.DataFrame, current_time: datetime) -> Dict[str, Any]:
        """Evaluate triggers for a specific intervention type.
        
        Args:
            intervention_name: Name of the intervention
            config: Intervention configuration
            patient_data: Patient data for evaluation
            current_time: Current evaluation time
            
        Returns:
            Dictionary with trigger evaluation results
        """
        triggered = False
        reasons = []
        
        if intervention_name == 'urgent_review':
            triggered, reasons = self._evaluate_urgent_review_triggers(config, patient_data, current_time)
            
        elif intervention_name == 'conveyance_ed':
            triggered, reasons = self._evaluate_ed_conveyance_triggers(config, patient_data, current_time)
            
        elif intervention_name == 'start_antibiotic':
            triggered, reasons = self._evaluate_antibiotic_triggers(config, patient_data, current_time)
            
        elif intervention_name == 'increase_diuretic':
            triggered, reasons = self._evaluate_diuretic_triggers(config, patient_data, current_time)
            
        elif intervention_name == 'start_oxygen':
            triggered, reasons = self._evaluate_oxygen_triggers(config, patient_data, current_time)
        
        return {'triggered': triggered, 'reasons': reasons}
    
    def _evaluate_urgent_review_triggers(self, config: Dict[str, Any], 
                                       patient_data: pd.DataFrame, current_time: datetime) -> Tuple[bool, List[str]]:
        """Evaluate urgent review triggers.
        
        Triggers:
        - SpO2 ≤ 92 or drop ≥ 3 points from 7-day baseline
        - Dyspnea ≥ 3 or New chest pain = Yes
        - HR ≥ 110 sustained 6h+ and Fatigue ≥ 7
        """
        reasons = []
        
        # Get recent data (last 24 hours for evaluation)
        recent_data = self._get_recent_data(patient_data, current_time, hours=24)
        
        if recent_data.empty:
            return False, reasons
        
        # SpO2 absolute threshold (≤ 92)
        spo2_data = recent_data[recent_data['concept_id'] == 2100000100]  # SpO2
        if not spo2_data.empty:
            latest_spo2 = spo2_data.iloc[-1]['value_as_number']
            if latest_spo2 <= 92:
                reasons.append(f"SpO2 ≤ 92% (current: {latest_spo2}%)")
                return True, reasons
        
        # SpO2 drop from 7-day baseline (≥ 3 points)
        if not spo2_data.empty:
            baseline_data = self._get_recent_data(patient_data, current_time, hours=168)  # 7 days
            baseline_spo2_data = baseline_data[baseline_data['concept_id'] == 2100000100]
            if not baseline_spo2_data.empty:
                baseline_mean = baseline_spo2_data['value_as_number'].mean()
                latest_spo2 = spo2_data.iloc[-1]['value_as_number']
                drop = baseline_mean - latest_spo2
                if drop >= 3:
                    reasons.append(f"SpO2 drop ≥ 3 points from baseline (drop: {drop:.1f})")
                    return True, reasons
        
        # Dyspnea ≥ 3
        dyspnea_data = recent_data[recent_data['concept_id'] == 2100000201]  # Dyspnea
        if not dyspnea_data.empty:
            latest_dyspnea = dyspnea_data.iloc[-1]['value_as_number']
            if latest_dyspnea >= 3:
                reasons.append(f"Dyspnea ≥ 3 (current: {latest_dyspnea})")
                return True, reasons
        
        # New chest pain = Yes
        chest_pain_data = recent_data[recent_data['concept_id'] == 2100000203]  # New Chest Pain
        if not chest_pain_data.empty:
            latest_chest_pain = chest_pain_data.iloc[-1]['value_as_concept_id']
            if latest_chest_pain == 2100000020:  # Yes
                reasons.append("New chest pain reported")
                return True, reasons
        
        # HR ≥ 110 sustained 6h+ and Fatigue ≥ 7
        hr_data = recent_data[recent_data['concept_id'] == 2100000101]  # Heart Rate
        fatigue_data = recent_data[recent_data['concept_id'] == 2100000207]  # Fatigue
        
        if not hr_data.empty and not fatigue_data.empty:
            # Check for sustained HR ≥ 110 over 6 hours
            hr_6h_data = self._get_recent_data(patient_data, current_time, hours=6)
            hr_6h_data = hr_6h_data[hr_6h_data['concept_id'] == 2100000101]
            
            if not hr_6h_data.empty:
                sustained_high_hr = (hr_6h_data['value_as_number'] >= 110).all()
                latest_fatigue = fatigue_data.iloc[-1]['value_as_number']
                
                if sustained_high_hr and latest_fatigue >= 7:
                    reasons.append(f"Sustained HR ≥ 110 for 6h+ and Fatigue ≥ 7 (fatigue: {latest_fatigue})")
                    return True, reasons
        
        return False, reasons
    
    def _evaluate_ed_conveyance_triggers(self, config: Dict[str, Any], 
                                       patient_data: pd.DataFrame, current_time: datetime) -> Tuple[bool, List[str]]:
        """Evaluate ED conveyance triggers.
        
        Triggers:
        - SpO2 ≤ 88
        - New chest pain = Yes and HR ≥ 120
        - SBP < 90 and Dizziness = Yes
        """
        reasons = []
        
        # Get recent data (last 24 hours)
        recent_data = self._get_recent_data(patient_data, current_time, hours=24)
        
        if recent_data.empty:
            return False, reasons
        
        # SpO2 ≤ 88
        spo2_data = recent_data[recent_data['concept_id'] == 2100000100]  # SpO2
        if not spo2_data.empty:
            latest_spo2 = spo2_data.iloc[-1]['value_as_number']
            if latest_spo2 <= 88:
                reasons.append(f"SpO2 ≤ 88% (current: {latest_spo2}%)")
                return True, reasons
        
        # New chest pain = Yes and HR ≥ 120
        chest_pain_data = recent_data[recent_data['concept_id'] == 2100000203]  # New Chest Pain
        hr_data = recent_data[recent_data['concept_id'] == 2100000101]  # Heart Rate
        
        if not chest_pain_data.empty and not hr_data.empty:
            latest_chest_pain = chest_pain_data.iloc[-1]['value_as_concept_id']
            latest_hr = hr_data.iloc[-1]['value_as_number']
            
            if latest_chest_pain == 2100000020 and latest_hr >= 120:  # Yes and HR ≥ 120
                reasons.append(f"New chest pain + HR ≥ 120 (HR: {latest_hr})")
                return True, reasons
        
        # SBP < 90 and Dizziness = Yes
        sbp_data = recent_data[recent_data['concept_id'] == 4232915]  # Systolic Sitting BP
        dizziness_data = recent_data[recent_data['concept_id'] == 2100000205]  # Dizziness
        
        if not sbp_data.empty and not dizziness_data.empty:
            latest_sbp = sbp_data.iloc[-1]['value_as_number']
            latest_dizziness = dizziness_data.iloc[-1]['value_as_concept_id']
            
            if latest_sbp < 90 and latest_dizziness == 2100000020:  # SBP < 90 and Yes
                reasons.append(f"Hypotension + dizziness (SBP: {latest_sbp})")
                return True, reasons
        
        return False, reasons
    
    def _evaluate_antibiotic_triggers(self, config: Dict[str, Any], 
                                    patient_data: pd.DataFrame, current_time: datetime) -> Tuple[bool, List[str]]:
        """Evaluate antibiotic triggers.
        
        Triggers:
        - Cough ≥ 2 and Temp ≥ 37.8 for 2 consecutive days and Appetite reduced = Yes
        """
        reasons = []
        
        # Get recent data (last 48 hours for 2-day evaluation)
        recent_data = self._get_recent_data(patient_data, current_time, hours=48)
        
        if recent_data.empty:
            return False, reasons
        
        # Check current symptoms
        current_data = self._get_recent_data(patient_data, current_time, hours=24)
        
        # Cough ≥ 2
        cough_data = current_data[current_data['concept_id'] == 2100000202]  # Cough
        if cough_data.empty:
            return False, reasons
        
        latest_cough = cough_data.iloc[-1]['value_as_number']
        if latest_cough < 2:
            return False, reasons
        
        # Appetite reduced = Yes
        appetite_data = current_data[current_data['concept_id'] == 2100000204]  # Appetite Reduced
        if appetite_data.empty:
            return False, reasons
        
        latest_appetite = appetite_data.iloc[-1]['value_as_concept_id']
        if latest_appetite != 2100000020:  # Not Yes
            return False, reasons
        
        # Temperature ≥ 37.8 for 2 consecutive days
        temp_data = recent_data[recent_data['concept_id'] == 2100000104]  # Temperature
        if temp_data.empty:
            return False, reasons
        
        # Check if temperature has been elevated for 2 days
        elevated_temp_days = 0
        temp_data_sorted = temp_data.sort_values('datetime')
        
        for i in range(len(temp_data_sorted)):
            if temp_data_sorted.iloc[i]['value_as_number'] >= 37.8:
                elevated_temp_days += 1
            else:
                elevated_temp_days = 0  # Reset counter if temp drops
        
        if elevated_temp_days >= 2:
            reasons.append(f"Cough ≥ 2, elevated temp ≥ 37.8°C for 2+ days, appetite reduced")
            return True, reasons
        
        return False, reasons
    
    def _evaluate_diuretic_triggers(self, config: Dict[str, Any], 
                                  patient_data: pd.DataFrame, current_time: datetime) -> Tuple[bool, List[str]]:
        """Evaluate diuretic triggers.
        
        Triggers:
        - Weight gain ≥ 1.5 kg over 48h OR
        - Ankle swelling = Yes with Dyspnea ≥ 2
        """
        reasons = []
        
        # Get recent data (last 48 hours)
        recent_data = self._get_recent_data(patient_data, current_time, hours=48)
        
        if recent_data.empty:
            return False, reasons
        
        # Weight gain ≥ 1.5 kg over 48h
        weight_data = recent_data[recent_data['concept_id'] == 2100000106]  # Weight
        if not weight_data.empty and len(weight_data) >= 2:
            weight_data_sorted = weight_data.sort_values('datetime')
            latest_weight = weight_data_sorted.iloc[-1]['value_as_number']
            baseline_weight = weight_data_sorted.iloc[0]['value_as_number']
            weight_gain = latest_weight - baseline_weight
            
            if weight_gain >= 1.5:
                reasons.append(f"Weight gain ≥ 1.5kg over 48h (gain: {weight_gain:.1f}kg)")
                return True, reasons
        
        # Ankle swelling = Yes with Dyspnea ≥ 2
        current_data = self._get_recent_data(patient_data, current_time, hours=24)
        ankle_swelling_data = current_data[current_data['concept_id'] == 2100000208]  # Ankle Swelling
        dyspnea_data = current_data[current_data['concept_id'] == 2100000201]  # Dyspnea
        
        if not ankle_swelling_data.empty and not dyspnea_data.empty:
            latest_ankle_swelling = ankle_swelling_data.iloc[-1]['value_as_concept_id']
            latest_dyspnea = dyspnea_data.iloc[-1]['value_as_number']
            
            if latest_ankle_swelling == 2100000020 and latest_dyspnea >= 2:  # Yes and Dyspnea ≥ 2
                reasons.append(f"Ankle swelling + dyspnea ≥ 2 (dyspnea: {latest_dyspnea})")
                return True, reasons
        
        return False, reasons
    
    def _evaluate_oxygen_triggers(self, config: Dict[str, Any], 
                                patient_data: pd.DataFrame, current_time: datetime) -> Tuple[bool, List[str]]:
        """Evaluate oxygen therapy triggers.
        
        Triggers:
        - SpO2 89-92% for 12h window and Dyspnea ≥ 2
        """
        reasons = []
        
        # Get recent data (last 12 hours)
        recent_data = self._get_recent_data(patient_data, current_time, hours=12)
        
        if recent_data.empty:
            return False, reasons
        
        # SpO2 in range 89-92% for 12h window
        spo2_data = recent_data[recent_data['concept_id'] == 2100000100]  # SpO2
        if spo2_data.empty:
            return False, reasons
        
        # Check if SpO2 has been consistently in 89-92% range
        spo2_in_range = ((spo2_data['value_as_number'] >= 89) & 
                        (spo2_data['value_as_number'] <= 92)).all()
        
        if not spo2_in_range:
            return False, reasons
        
        # Dyspnea ≥ 2
        current_data = self._get_recent_data(patient_data, current_time, hours=24)
        dyspnea_data = current_data[current_data['concept_id'] == 2100000201]  # Dyspnea
        
        if dyspnea_data.empty:
            return False, reasons
        
        latest_dyspnea = dyspnea_data.iloc[-1]['value_as_number']
        if latest_dyspnea >= 2:
            reasons.append(f"SpO2 89-92% for 12h + dyspnea ≥ 2 (dyspnea: {latest_dyspnea})")
            return True, reasons
        
        return False, reasons
    
    def _get_recent_data(self, patient_data: pd.DataFrame, current_time: datetime, 
                        hours: int) -> pd.DataFrame:
        """Get patient data from the last N hours.
        
        Args:
            patient_data: Full patient data
            current_time: Current evaluation time
            hours: Number of hours to look back
            
        Returns:
            Filtered DataFrame with recent data
        """
        cutoff_time = current_time - timedelta(hours=hours)
        
        # Filter data within time window
        recent_data = patient_data[patient_data['datetime'] >= cutoff_time]
        recent_data = recent_data[recent_data['datetime'] <= current_time]
        
        return recent_data.sort_values('datetime')    

    def apply_intervention_logic(self, episodes: pd.DataFrame, observations: pd.DataFrame, 
                               measurements: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply intervention logic to generate drug exposures and procedures.
        
        Args:
            episodes: DataFrame with episode information
            observations: DataFrame with PROM observations
            measurements: DataFrame with device measurements
            
        Returns:
            Tuple of (drug_exposure_df, procedure_occurrence_df)
        """
        self.logger.info(f"Applying intervention logic for {len(episodes)} episodes")
        
        drug_exposures = []
        procedures = []
        
        # Process each episode
        for _, episode in episodes.iterrows():
            person_id = episode['person_id']
            episode_id = episode['episode_id']
            episode_start = pd.to_datetime(episode['episode_start_datetime'])
            episode_end = pd.to_datetime(episode['episode_end_datetime'])
            
            # Get patient data for this episode
            episode_observations = observations[
                (observations['person_id'] == person_id) & 
                (observations['episode_id'] == episode_id)
            ].copy()
            
            episode_measurements = measurements[
                (measurements['person_id'] == person_id) & 
                (measurements['episode_id'] == episode_id)
            ].copy()
            
            # Combine observations and measurements for trigger evaluation
            patient_data = self._combine_patient_data(episode_observations, episode_measurements)
            
            if patient_data.empty:
                continue
            
            # Evaluate interventions using rolling windows (24-48h)
            episode_interventions = self._evaluate_episode_interventions(
                patient_data, episode_id, episode_start, episode_end
            )
            
            # Generate drug exposures and procedures
            episode_drugs, episode_procedures = self._generate_intervention_records(
                episode_interventions, person_id, episode_id
            )
            
            drug_exposures.extend(episode_drugs)
            procedures.extend(episode_procedures)
        
        # Convert to DataFrames
        drug_df = pd.DataFrame(drug_exposures) if drug_exposures else pd.DataFrame()
        procedure_df = pd.DataFrame(procedures) if procedures else pd.DataFrame()
        
        self.logger.info(f"Generated {len(drug_df)} drug exposures and {len(procedure_df)} procedures")
        
        return drug_df, procedure_df
    
    def _combine_patient_data(self, observations: pd.DataFrame, 
                            measurements: pd.DataFrame) -> pd.DataFrame:
        """Combine observations and measurements into unified patient data.
        
        Args:
            observations: PROM observations
            measurements: Device measurements
            
        Returns:
            Combined DataFrame with standardized columns
        """
        combined_data = []
        
        # Add observations
        for _, obs in observations.iterrows():
            combined_data.append({
                'datetime': pd.to_datetime(obs['observation_datetime']),
                'concept_id': obs['observation_concept_id'],
                'value_as_number': obs['value_as_number'],
                'value_as_concept_id': obs['value_as_concept_id'],
                'data_type': 'observation'
            })
        
        # Add measurements
        for _, meas in measurements.iterrows():
            combined_data.append({
                'datetime': pd.to_datetime(meas['measurement_datetime']),
                'concept_id': meas['measurement_concept_id'],
                'value_as_number': meas['value_as_number'],
                'value_as_concept_id': meas['value_as_concept_id'],
                'data_type': 'measurement'
            })
        
        if not combined_data:
            return pd.DataFrame()
        
        return pd.DataFrame(combined_data).sort_values('datetime')
    
    def _evaluate_episode_interventions(self, patient_data: pd.DataFrame, episode_id: int,
                                      episode_start: datetime, episode_end: datetime) -> List[Dict[str, Any]]:
        """Evaluate interventions for an entire episode using rolling windows.
        
        Args:
            patient_data: Combined patient data
            episode_id: Episode ID
            episode_start: Episode start datetime
            episode_end: Episode end datetime
            
        Returns:
            List of triggered interventions with timing
        """
        interventions = []
        
        # Use 24-hour evaluation windows
        current_time = episode_start + timedelta(hours=24)  # Start evaluating after first day
        
        while current_time <= episode_end:
            # Evaluate triggers at this time point
            triggered = self.evaluate_triggers(patient_data, episode_id, current_time)
            
            for intervention in triggered:
                # Apply realistic delays
                delay_config = intervention['config'].get('delay_hours_range', [0, 6])
                delay_hours = self.rng.uniform(delay_config[0], delay_config[1])
                intervention_time = current_time + timedelta(hours=delay_hours)
                
                # Ensure intervention time is within episode bounds
                intervention_time = min(intervention_time, episode_end)
                
                intervention['intervention_time'] = intervention_time
                interventions.append(intervention)
            
            # Move to next evaluation window (advance by 12 hours for overlap)
            current_time += timedelta(hours=12)
        
        # Remove duplicate interventions (same type within 24 hours)
        interventions = self._deduplicate_interventions(interventions)
        
        # Handle escalation logic
        interventions = self._handle_escalations(interventions, episode_end)
        
        return interventions
    
    def _deduplicate_interventions(self, interventions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate interventions of the same type within 24 hours.
        
        Args:
            interventions: List of triggered interventions
            
        Returns:
            Deduplicated list of interventions
        """
        if not interventions:
            return interventions
        
        # Sort by intervention time
        interventions.sort(key=lambda x: x['intervention_time'])
        
        deduplicated = []
        last_intervention_times = {}
        
        for intervention in interventions:
            intervention_type = intervention['intervention_type']
            intervention_time = intervention['intervention_time']
            
            # Check if same intervention type occurred within last 24 hours
            if intervention_type in last_intervention_times:
                time_diff = intervention_time - last_intervention_times[intervention_type]
                if time_diff < timedelta(hours=24):
                    continue  # Skip duplicate
            
            deduplicated.append(intervention)
            last_intervention_times[intervention_type] = intervention_time
        
        return deduplicated
    
    def _handle_escalations(self, interventions: List[Dict[str, Any]], 
                          episode_end: datetime) -> List[Dict[str, Any]]:
        """Handle intervention escalation logic.
        
        Args:
            interventions: List of interventions
            episode_end: Episode end datetime
            
        Returns:
            List of interventions with escalations added
        """
        escalated_interventions = interventions.copy()
        
        # Handle urgent review → ED escalation
        urgent_reviews = [i for i in interventions if i['intervention_type'] == 'urgent_review']
        
        for urgent_review in urgent_reviews:
            # 20-40% chance of escalation to ED within 24h
            escalation_prob = self.rng.uniform(0.20, 0.40)
            
            if self.rng.random() < escalation_prob:
                escalation_time = urgent_review['intervention_time'] + timedelta(
                    hours=self.rng.uniform(1, 24)
                )
                
                # Ensure escalation is within episode bounds
                if escalation_time <= episode_end:
                    escalated_interventions.append({
                        'intervention_type': 'conveyance_ed',
                        'concept_id': 2100000502,  # ED conveyance
                        'episode_id': urgent_review['episode_id'],
                        'trigger_time': urgent_review['trigger_time'],
                        'intervention_time': escalation_time,
                        'trigger_reasons': ['Escalation from urgent review'],
                        'config': self.config['interventions']['conveyance_ed']
                    })
        
        # Handle ED → admission escalation
        ed_conveyances = [i for i in escalated_interventions if i['intervention_type'] == 'conveyance_ed']
        
        for ed_conveyance in ed_conveyances:
            # 30-50% chance of admission within 6h
            admission_prob = self.rng.uniform(0.30, 0.50)
            
            if self.rng.random() < admission_prob:
                admission_time = ed_conveyance['intervention_time'] + timedelta(
                    hours=self.rng.uniform(0, 6)
                )
                
                # Ensure admission is within episode bounds
                if admission_time <= episode_end:
                    escalated_interventions.append({
                        'intervention_type': 'unplanned_admission',
                        'concept_id': 2100000503,  # Unplanned admission
                        'episode_id': ed_conveyance['episode_id'],
                        'trigger_time': ed_conveyance['trigger_time'],
                        'intervention_time': admission_time,
                        'trigger_reasons': ['Escalation from ED conveyance'],
                        'config': {'concept_id': 2100000503}
                    })
        
        return escalated_interventions
    
    def _generate_intervention_records(self, interventions: List[Dict[str, Any]], 
                                     person_id: int, episode_id: int) -> Tuple[List[Dict], List[Dict]]:
        """Generate drug exposure and procedure occurrence records.
        
        Args:
            interventions: List of triggered interventions
            person_id: Person ID
            episode_id: Episode ID
            
        Returns:
            Tuple of (drug_records, procedure_records)
        """
        drug_records = []
        procedure_records = []
        
        for intervention in interventions:
            intervention_type = intervention['intervention_type']
            concept_id = intervention['concept_id']
            intervention_time = intervention['intervention_time']
            
            if intervention_type in ['start_antibiotic', 'increase_diuretic']:
                # Generate drug exposure
                drug_record = self._create_drug_exposure(
                    self.drug_id_counter, person_id, episode_id, concept_id, 
                    intervention_time, intervention['config']
                )
                drug_records.append(drug_record)
                self.drug_id_counter += 1
                
            else:
                # Generate procedure occurrence
                procedure_record = self._create_procedure_occurrence(
                    self.procedure_id_counter, person_id, episode_id, concept_id, intervention_time
                )
                procedure_records.append(procedure_record)
                self.procedure_id_counter += 1
        
        return drug_records, procedure_records
    
    def _create_drug_exposure(self, drug_id: int, person_id: int, episode_id: int,
                            concept_id: int, start_time: datetime, 
                            config: Dict[str, Any]) -> Dict[str, Any]:
        """Create drug exposure record.
        
        Args:
            drug_id: Drug exposure ID
            person_id: Person ID
            episode_id: Episode ID
            concept_id: Drug concept ID
            start_time: Drug start datetime
            config: Drug configuration
            
        Returns:
            Drug exposure record
        """
        # Get days supply range
        days_supply_range = config.get('days_supply_range', [5, 10])
        days_supply = self.rng.integers(days_supply_range[0], days_supply_range[1] + 1)
        
        end_time = start_time + timedelta(days=int(days_supply))
        
        return {
            'drug_exposure_id': drug_id,
            'person_id': person_id,
            'drug_concept_id': concept_id,
            'drug_exposure_start_date': start_time.date(),
            'drug_exposure_start_datetime': start_time,
            'drug_exposure_end_date': end_time.date(),
            'drug_exposure_end_datetime': end_time,
            'verbatim_end_date': end_time.date(),
            'drug_type_concept_id': 2100000017,  # Virtual ward drug type
            'stop_reason': None,
            'refills': 0,
            'quantity': None,
            'days_supply': days_supply,
            'sig': f"As prescribed for {days_supply} days",
            'route_concept_id': None,
            'lot_number': None,
            'provider_id': None,
            'visit_occurrence_id': None,
            'visit_detail_id': None,
            'drug_source_value': self.concept_map[concept_id]['name'],
            'drug_source_concept_id': concept_id,
            'route_source_value': None,
            'dose_unit_source_value': None,
            'episode_id': episode_id
        }
    
    def _create_procedure_occurrence(self, procedure_id: int, person_id: int, 
                                   episode_id: int, concept_id: int, 
                                   procedure_time: datetime) -> Dict[str, Any]:
        """Create procedure occurrence record.
        
        Args:
            procedure_id: Procedure occurrence ID
            person_id: Person ID
            episode_id: Episode ID
            concept_id: Procedure concept ID
            procedure_time: Procedure datetime
            
        Returns:
            Procedure occurrence record
        """
        return {
            'procedure_occurrence_id': procedure_id,
            'person_id': person_id,
            'procedure_concept_id': concept_id,
            'procedure_date': procedure_time.date(),
            'procedure_datetime': procedure_time,
            'procedure_end_date': procedure_time.date(),
            'procedure_end_datetime': procedure_time,
            'procedure_type_concept_id': 2100000016,  # Virtual ward procedure type
            'modifier_concept_id': None,
            'quantity': 1,
            'provider_id': None,
            'visit_occurrence_id': None,
            'visit_detail_id': None,
            'procedure_source_value': self.concept_map[concept_id]['name'],
            'procedure_source_concept_id': concept_id,
            'modifier_source_value': None,
            'episode_id': episode_id
        }