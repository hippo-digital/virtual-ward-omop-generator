from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..models.base import BaseGenerator
from ..exceptions import ConfigurationError

class MortalityGenerator(BaseGenerator):
    """Generates death events based on age, gender, and episode severity."""
    def __init__(self, config: Dict[str, Any], concept_dict: pd.DataFrame, rng: np.random.Generator):
        super().__init__(config, concept_dict, rng)
        self.death_id_counter = 1

    def _validate_config(self) -> None:
        dataset_config = self.config.get('dataset', {})

        # Mortality config is optional
        if 'mortality' not in dataset_config:
            self.logger.info("No mortality configuration found - using defaults")
            self.mortality_config = {
                'base_rate': 0.005,  # 0.5% base mortality
                'age_multipliers': {},
                'gender_multipliers': {}
            }
        else:
            self.mortality_config = dataset_config['mortality']

    def generate(self, persons_df: pd.DataFrame, episodes_df: pd.DataFrame, 
                procedures_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate death records based on person demographics and episode outcomes.
        Args:
            persons_df: DataFrame with person demographics (must have year_of_birth, gender_concept_id)
            episodes_df: DataFrame with episodes
            procedures_df: DataFrame with procedures (to identify severe outcomes)   
        Returns:
            DataFrame with death records
        """
        deaths = []

        # Get current year for age calculation
        current_year = datetime.now().year

        # Get death type concept
        death_type_concept = self.config['types'].get('death_type', 2100000900)

        for _, person in persons_df.iterrows():
            person_id = person['person_id']
            age = current_year - person['year_of_birth']
            gender_concept = person['gender_concept_id']

            # Calculate death probability
            base_rate = self.mortality_config.get('base_rate', 0.005)

            # Apply age multiplier
            age_mult = self._get_age_multiplier(age)

            # Apply gender multiplier  
            gender_mult = self._get_gender_multiplier(gender_concept)

            # Check for severe episodes (unplanned admission = higher risk)
            person_episodes = episodes_df[episodes_df['person_id'] == person_id]

            # Higher risk if unplanned admission occurred
            severe_outcome_mult = 1.0
            if not procedures_df.empty and 'person_id' in procedures_df.columns:
                person_procedures = procedures_df[procedures_df['person_id'] == person_id]
                if len(person_procedures[person_procedures['procedure_concept_id'] == 2100000503]) > 0:
                    severe_outcome_mult = 2.0  # 2x risk if unplanned admission

            # Final death probability
            death_prob = base_rate * age_mult * gender_mult * severe_outcome_mult

            # Generate death event
            if self.rng.random() < death_prob:
                # Death occurs during or shortly after an episode
                if len(person_episodes) > 0:
                    person_episodes = person_episodes.sort_values("episode_end_datetime")
                    last_episode = person_episodes.iloc[-1]
                    # Death 0-7 days after episode end
                    days_after = int(self.rng.integers(0, 8))
                    death_datetime = last_episode['episode_end_datetime'] + timedelta(days=days_after)

                    deaths.append({
                        'person_id': int(person_id),
                        'death_date': death_datetime.date(),
                        'death_datetime': death_datetime,
                        'death_type_concept_id': int(death_type_concept),
                        'cause_concept_id': None,  # Could be linked to condition
                        'cause_source_value': None,
                        'cause_source_concept_id': None
                    })

        if len(deaths) == 0:
            # Return empty DataFrame with correct schema
            deaths_df = pd.DataFrame(columns=[
                'person_id', 'death_date', 'death_datetime', 'death_type_concept_id',
                'cause_concept_id', 'cause_source_value', 'cause_source_concept_id'
            ])
        else:
            deaths_df = pd.DataFrame(deaths)

        self._log_generation_stats(deaths_df, "death")

        if len(deaths_df) > 0:
            self.logger.info(f"Generated {len(deaths_df)} death events "
                           f"({len(deaths_df) / len(persons_df) * 100:.2f}% mortality rate)")

        return deaths_df

    def _get_age_multiplier(self, age: int) -> float:
        """Get mortality multiplier based on age."""
        age_multipliers = self.mortality_config.get('age_multipliers', {})

        if age >= 85:
            return age_multipliers.get('85+', 1.18)
        elif age >= 75:
            return age_multipliers.get('75-84', 1.20)
        elif age >= 65:
            return age_multipliers.get('65-74', 1.15)
        else:
            return 1.0

    def _get_gender_multiplier(self, gender_concept_id: int) -> float:
        """Get mortality multiplier based on gender."""
        gender_multipliers = self.mortality_config.get('gender_multipliers', {})
        if gender_concept_id == 8507:  # Male
            return gender_multipliers.get('male', 1.18)
        elif gender_concept_id == 8532:  # Female
            return gender_multipliers.get('female', 1.10)
        else:
            return 1.0