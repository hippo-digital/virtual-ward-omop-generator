"""Temporal coordination utility for proper date generation logic."""

import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional
import numpy as np
from ..exceptions import DataGenerationError

logger = logging.getLogger(__name__)


class TemporalCoordinator:
    """Coordinates temporal data generation to ensure proper date sequencing."""
    
    def __init__(self, base_year: int = 2024):
        """Initialize the temporal coordinator.

        Args:
            base_year: The current/base year for date calculations
        """
        self.base_year = base_year
        self.base_date = datetime(base_year, 1, 1)

        logger.info(f"Temporal Coordinator initialized with base year {base_year}")
    
    def generate_birth_date(self, rng: np.random.Generator, age_ranges) -> Tuple[int, int, int]:
        """Generate realistic birth year, month, and day matching Demographic Overview.

        Args:
            rng: Random number generator
        Returns:
            Tuple of (year, month, day)
        """
        try:
            year_ranges = []
            weights = []
            for r in age_ranges:
                year_ranges.append((self.base_year - r['max'], self.base_year - r['min']))
                weights.append(r['weight'])

            assert sum(weights) == 1.0, "Age range weights do not sum to 1"

            # Select age bracket
            bracket_idx = rng.choice(len(year_ranges), p=weights)
            min_year, max_year = year_ranges[bracket_idx]

            # Generate birth year within selected bracket
            birth_year = int(rng.integers(min_year, max_year + 1))

            # Generate month (1-12)
            birth_month = int(rng.integers(1, 13))

            # Generate day based on month (simplified - doesn't account for leap years)
            days_in_month = {
                1: 31,
                2: 28,
                3: 31,
                4: 30,
                5: 31,
                6: 30,
                7: 31,
                8: 31,
                9: 30,
                10: 31,
                11: 30,
                12: 31,
            }
            max_day = days_in_month[birth_month]
            birth_day = int(rng.integers(1, max_day + 1))

            logger.debug(f"Generated birth date: {birth_year}-{birth_month:02d}-{birth_day:02d}")
            return birth_year, birth_month, birth_day

        except Exception as e:
            raise DataGenerationError(
                f"Failed to generate birth date: {e}",
                context={
                    "base_year": self.base_year,
                    "age_ranges": age_ranges
                },
            )
    
    def generate_episode_dates(
        self, 
        person_birth_date: Tuple[int, int, int], 
        episode_length_days: int, 
        rng: np.random.Generator,
    ) -> Tuple[datetime, datetime]:
        """Generate episode start and end dates ensuring person is adult and dates are recent.
        
        Args:
            person_birth_date: Tuple of (year, month, day) for person's birth
            episode_length_days: Length of episode in days
            rng: Random number generator
            
        Returns:
            Tuple of (episode_start_datetime, episode_end_datetime)
        """
        try:
            birth_year, birth_month, birth_day = person_birth_date
            birth_date = datetime(birth_year, birth_month, birth_day)
            
            # Ensure person is at least 18 years old
            min_episode_date = birth_date + timedelta(days=18 * 365)
            
            # Episodes should be recent (within last 2 years)
            max_episode_start = self.base_date - timedelta(
                days=30
            )  # At least 30 days ago
            min_episode_start = self.base_date - timedelta(
                days=2 * 365
            )  # Within 2 years
            
            # Adjust if person is too young
            if min_episode_date > min_episode_start:
                min_episode_start = min_episode_date
            
            # Ensure we have a valid range
            if min_episode_start >= max_episode_start:
                max_episode_start = min_episode_start + timedelta(days=365)
            
            # Generate random start date
            date_range_days = (max_episode_start - min_episode_start).days
            if date_range_days <= 0:
                date_range_days = 30  # Fallback to 30 days
            
            random_days = int(rng.integers(0, date_range_days + 1))
            episode_start = min_episode_start + timedelta(days=random_days)
            
            # Calculate end date
            episode_end = episode_start + timedelta(days=episode_length_days)
            
            logger.debug(
                f"Generated episode dates: {episode_start.date()} to {episode_end.date()} "
                f"(length: {episode_length_days} days)"
            )
            
            return episode_start, episode_end
            
        except Exception as e:
            raise DataGenerationError(
                f"Failed to generate episode dates: {e}",
                context={
                    "person_birth_date": person_birth_date,
                    "episode_length_days": episode_length_days,
                    "base_year": self.base_year,
                },
            )
    
    def generate_visit_dates(
        self, episode_start: datetime, episode_end: datetime, rng: np.random.Generator
    ) -> Tuple[datetime, datetime]:
        """Generate visit dates within episode boundaries.
        
        Args:
            episode_start: Episode start datetime
            episode_end: Episode end datetime
            rng: Random number generator
            
        Returns:
            Tuple of (visit_start_datetime, visit_end_datetime)
        """
        try:
            # Visit should be within episode boundaries
            episode_duration = (episode_end - episode_start).total_seconds()
            
            # Generate visit start time (random point within episode)
            random_seconds = rng.uniform(0, episode_duration)
            visit_start = episode_start + timedelta(seconds=random_seconds)
            
            # Visit duration is typically short (1-4 hours for virtual ward)
            visit_duration_hours = rng.uniform(1, 4)
            visit_end = visit_start + timedelta(hours=visit_duration_hours)
            
            # Ensure visit doesn't extend beyond episode
            if visit_end > episode_end:
                visit_end = episode_end
                visit_start = max(episode_start, visit_end - timedelta(hours=1))
            
            logger.debug(
                f"Generated visit dates: {visit_start} to {visit_end} "
                f"(within episode {episode_start.date()} to {episode_end.date()})"
            )
            
            return visit_start, visit_end
            
        except Exception as e:
            raise DataGenerationError(
                f"Failed to generate visit dates: {e}",
                context={"episode_start": episode_start, "episode_end": episode_end},
            )
    
    def generate_measurement_datetime(
        self, visit_start: datetime, visit_end: datetime, rng: np.random.Generator
    ) -> datetime:
        """Generate measurement datetime within visit boundaries.
        
        Args:
            visit_start: Visit start datetime
            visit_end: Visit end datetime
            rng: Random number generator
            
        Returns:
            Measurement datetime
        """
        try:
            visit_duration = (visit_end - visit_start).total_seconds()
            
            if visit_duration <= 0:
                return visit_start
            
            # Generate random time within visit
            random_seconds = rng.uniform(0, visit_duration)
            measurement_datetime = visit_start + timedelta(seconds=random_seconds)
            
            logger.debug(
                f"Generated measurement datetime: {measurement_datetime} "
                f"(within visit {visit_start} to {visit_end})"
            )
            
            return measurement_datetime
            
        except Exception as e:
            raise DataGenerationError(
                f"Failed to generate measurement datetime: {e}",
                context={"visit_start": visit_start, "visit_end": visit_end},
            )
    
    def validate_temporal_sequence(
        self, 
        birth_date: Tuple[int, int, int],
        episode_start: datetime,
        episode_end: datetime,
        visit_start: Optional[datetime] = None,
        visit_end: Optional[datetime] = None,
    ) -> bool:
        """Validate that temporal sequence is logical.
        
        Args:
            birth_date: Person's birth date
            episode_start: Episode start datetime
            episode_end: Episode end datetime
            visit_start: Optional visit start datetime
            visit_end: Optional visit end datetime
            
        Returns:
            True if sequence is valid, False otherwise
        """
        try:
            birth_year, birth_month, birth_day = birth_date
            birth_datetime = datetime(birth_year, birth_month, birth_day)
            
            # Check basic sequence
            if episode_start >= episode_end:
                logger.error("Episode start is not before episode end")
                return False
            
            if episode_start <= birth_datetime:
                logger.error("Episode start is not after birth date")
                return False
            
            if visit_start and visit_end:
                if visit_start >= visit_end:
                    logger.error("Visit start is not before visit end")
                    return False
                
                if visit_start < episode_start or visit_end > episode_end:
                    logger.error("Visit is not within episode boundaries")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate temporal sequence: {e}")
            return False