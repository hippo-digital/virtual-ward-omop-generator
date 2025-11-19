"""Visit generator for visit_occurrence records linked to episodes."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime
from ..models.base import BaseGenerator
from ..exceptions import ConfigurationError
from ..validation.concept_validator import ConceptValidator
from ..utils.temporal_coordinator import TemporalCoordinator


class VisitGenerator(BaseGenerator):
    """Generates visit_occurrence records for each episode."""
    
    def __init__(self, config: Dict[str, Any], concept_dict: pd.DataFrame, rng: np.random.Generator):
        """Initialize the VisitGenerator with concept validation and temporal coordination.
        
        Args:
            config: Configuration dictionary
            concept_dict: DataFrame containing valid concept IDs
            rng: Random number generator
        """
        super().__init__(config, concept_dict, rng)
        self.concept_validator = ConceptValidator(concept_dict)
        self.temporal_coordinator = TemporalCoordinator()
    
    def _validate_config(self) -> None:
        """Validate visit generator configuration."""
        if 'episode_and_visit_concepts' not in self.config:
            raise ConfigurationError("Missing 'episode_and_visit_concepts' section in configuration")
        
        visit_concepts = self.config['episode_and_visit_concepts']
        
        if 'visits' not in visit_concepts:
            raise ConfigurationError("Missing 'visits' in episode_and_visit_concepts configuration")
        
        visits_config = visit_concepts['visits']
        if not isinstance(visits_config, dict) or len(visits_config) == 0:
            raise ConfigurationError("visits configuration must be a non-empty dictionary")
        
        # Validate that we have at least one visit type
        if 'telehealth' not in visits_config:
            raise ConfigurationError("Missing 'telehealth' visit type in visits configuration")
        
        # Validate types section exists for visit type concept ID
        if 'types' not in self.config:
            raise ConfigurationError("Missing 'types' section in configuration")
    
    def generate(self, episodes_df: pd.DataFrame, death_lookup: Dict[int, Any] = None, **kwargs: Any) -> pd.DataFrame:
        """Generate visit_occurrence records for each episode.

        Args:
            episodes_df: DataFrame containing episode data
            death_lookup: Optional dictionary mapping person_id to death_date

        Returns:
            DataFrame with visit_occurrence records
        """
        if episodes_df.empty:
            self.logger.warning("No episodes provided, generating empty visit_occurrence table")
            return pd.DataFrame(columns=[
                'visit_occurrence_id', 'person_id', 'visit_concept_id',
                'visit_start_date', 'visit_start_datetime', 'visit_end_date', 'visit_end_datetime',
                'visit_type_concept_id', 'provider_id', 'care_site_id',
                'visit_source_value', 'visit_source_concept_id',
                'admitted_from_concept_id', 'admitted_from_source_value',
                'discharged_to_concept_id', 'discharged_to_source_value',
                'preceding_visit_occurrence_id', 'episode_id'
            ])

        visits_df = self._generate_visits(episodes_df, death_lookup)
        self._log_generation_stats(visits_df, "visit_occurrence")

        return visits_df
    
    def _generate_visits(self, episodes_df: pd.DataFrame, death_lookup: Dict[int, Any] = None) -> pd.DataFrame:
        """Generate visit_occurrence records for each episode.

        Args:
            episodes_df: DataFrame containing episode data
            death_lookup: Optional dictionary mapping person_id to death_date

        Returns:
            DataFrame with visit_occurrence records
        """
        visits_list = []
        visit_occurrence_id = 1

        # Get visit concept IDs from configuration
        visits_config = self.config['episode_and_visit_concepts']['visits']

        # For virtual ward, we primarily use telehealth visits
        telehealth_concept_id = self.concept_validator.validate_concept(
            visits_config['telehealth'], 'visit_telehealth'
        )

        # Get visit type concept ID (from types section)
        # This represents the type/source of the visit record
        visit_type_concept_id = self.concept_validator.validate_concept(
            self.config.get('types', {}).get('visit_type_virtual_ward', 0),
            'visit_type'
        )

        for _, episode in episodes_df.iterrows():
            person_id = episode['person_id']

            # Skip episodes that start after death date
            if death_lookup is not None:
                death_date = death_lookup.get(person_id)
                if death_date and episode['episode_start_datetime'].date() > death_date:
                    self.logger.debug(f"Skipping episode {episode['episode_id']} for person {person_id} - starts after death on {death_date}")
                    continue

            # Generate 1-3 visits per episode based on episode length
            episode_length = episode['episode_length_days']
            
            if episode_length <= 7:
                num_visits = 1  # Short episodes get 1 visit
            elif episode_length <= 14:
                num_visits = self.rng.integers(1, 3)  # Medium episodes get 1-2 visits
            else:
                num_visits = self.rng.integers(2, 4)  # Long episodes get 2-3 visits
            
            # Generate visits for this episode
            episode_visits = self._generate_episode_visits(
                episode, num_visits, telehealth_concept_id, visit_type_concept_id, visit_occurrence_id
            )
            
            visits_list.extend(episode_visits)
            visit_occurrence_id += len(episode_visits)
        
        return pd.DataFrame(visits_list)
    
    def _generate_episode_visits(
        self, 
        episode: pd.Series, 
        num_visits: int,
        visit_concept_id: int,
        visit_type_concept_id: int,
        starting_visit_id: int
    ) -> List[Dict[str, Any]]:
        """Generate visits for a single episode.
        
        Args:
            episode: Episode data as pandas Series
            num_visits: Number of visits to generate
            visit_concept_id: Concept ID for visit type
            visit_type_concept_id: Concept ID for visit type classification
            starting_visit_id: Starting ID for visit numbering
            
        Returns:
            List of visit dictionaries
        """
        visits = []
        episode_start = episode['episode_start_datetime']
        episode_end = episode['episode_end_datetime']
        
        # Distribute visits across the episode timeline
        if num_visits == 1:
            # Single visit - place it randomly within the episode
            visit_start, visit_end = self.temporal_coordinator.generate_visit_dates(
                episode_start, episode_end, self.rng
            )
            visit_times = [(visit_start, visit_end)]
        else:
            # Multiple visits - distribute them across the episode
            visit_times = self._distribute_visits_across_episode(
                episode_start, episode_end, num_visits
            )
        
        # Create visit records
        for i, (visit_start, visit_end) in enumerate(visit_times):
            visit_id = starting_visit_id + i
            
            # Determine preceding visit (if any)
            preceding_visit_id = None
            if i > 0:
                preceding_visit_id = starting_visit_id + i - 1
            
            visit_record = {
                'visit_occurrence_id': visit_id,
                'person_id': episode['person_id'],
                'visit_concept_id': visit_concept_id,
                'visit_start_date': visit_start.date(),
                'visit_start_datetime': visit_start,
                'visit_end_date': visit_end.date(),
                'visit_end_datetime': visit_end,
                'visit_type_concept_id': visit_type_concept_id,
                'provider_id': None,  # Not specified in requirements
                'care_site_id': None,  # Not specified in requirements
                'visit_source_value': 'virtual_ward_telehealth',
                'visit_source_concept_id': visit_concept_id,
                'admitted_from_concept_id': None,  # Not applicable for virtual ward
                'admitted_from_source_value': None,
                'discharged_to_concept_id': None,  # Not applicable for virtual ward
                'discharged_to_source_value': None,
                'preceding_visit_occurrence_id': preceding_visit_id,
                'episode_id': episode['episode_id']  # Link to episode
            }
            
            visits.append(visit_record)
        
        return visits
    
    def _distribute_visits_across_episode(
        self, 
        episode_start: datetime, 
        episode_end: datetime, 
        num_visits: int
    ) -> List[tuple]:
        """Distribute multiple visits across an episode timeline.
        
        Args:
            episode_start: Episode start datetime
            episode_end: Episode end datetime
            num_visits: Number of visits to distribute
            
        Returns:
            List of (visit_start, visit_end) tuples
        """
        episode_duration = (episode_end - episode_start).total_seconds()
        visit_times = []
        
        # Divide episode into segments for visit distribution
        segment_duration = episode_duration / num_visits
        
        for i in range(num_visits):
            # Calculate segment boundaries
            segment_start = episode_start + pd.Timedelta(seconds=i * segment_duration)
            segment_end = episode_start + pd.Timedelta(seconds=(i + 1) * segment_duration)
            
            # Ensure last segment doesn't exceed episode end
            if segment_end > episode_end:
                segment_end = episode_end
            
            # Generate visit within this segment
            visit_start, visit_end = self.temporal_coordinator.generate_visit_dates(
                segment_start, segment_end, self.rng
            )
            
            visit_times.append((visit_start, visit_end))
        
        # Sort visits by start time to ensure proper chronological order
        visit_times.sort(key=lambda x: x[0])
        
        return visit_times