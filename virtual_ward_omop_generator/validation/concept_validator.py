"""Concept validation utility for ensuring valid OMOP concept IDs."""

import logging
import pandas as pd
from typing import Dict, Set, Optional, Any
from ..exceptions import DataGenerationError

logger = logging.getLogger(__name__)


class ConceptValidator:
    """Validates concept IDs against the provided concept dictionary."""
    
    def __init__(self, concept_dict: pd.DataFrame):
        """Initialize the validator with a concept dictionary.
        
        Args:
            concept_dict: DataFrame containing concept_id and related concept information
        """
        self.concept_dict = concept_dict
        self.valid_concepts: Set[int] = set(concept_dict['concept_id'].values)
        self._fallback_concepts = self._initialize_fallback_concepts()
        
        logger.info(f"ConceptValidator initialized with {len(self.valid_concepts)} valid concepts")
    
    def _initialize_fallback_concepts(self) -> Dict[str, int]:
        """Initialize standard fallback concepts for common types."""
        fallbacks = {}
        
        # Find fallback concepts by searching the concept dictionary
        concept_mappings = {
            'episode': 'Virtual Ward Episode',
            'visit_telehealth': 'Telehealth Visit',
            'visit_home': 'Home Visit',
            'observation_type': 'Patient-reported Observation',
            'measurement_type': 'Patient Device Measurement',
            'period_type': 'Observation Period—Virtual Ward',
            'condition_type': 'Index Condition',
            'procedure_type': 'Virtual Ward Procedure',
            'drug_type': 'Virtual Ward Drug Exposure',
            'yes': 'Yes',
            'no': 'No',
            'copd': 'Chronic Obstructive Pulmonary Disease',
            'heart_failure': 'Heart Failure',
            'spo2': 'SpO2',
            'heart_rate': 'Heart Rate',
            'dyspnea': 'Dyspnea 0-4'
        }
        
        for key, concept_name in concept_mappings.items():
            matching_concepts = self.concept_dict[
                self.concept_dict['concept_name'] == concept_name
            ]
            if not matching_concepts.empty:
                fallbacks[key] = int(matching_concepts.iloc[0]['concept_id'])
            # Note: No longer logging warnings here since we want strict validation
        
        return fallbacks
    
    def validate_concept(self, concept_id: int, fallback_key: Optional[str] = None) -> int:
        """Validate a concept ID and return it if valid, otherwise raise exception.
        
        Args:
            concept_id: The concept ID to validate
            fallback_key: Key for fallback concept if validation fails (for error context)
            
        Returns:
            Valid concept ID (original only)
            
        Raises:
            DataGenerationError: If concept ID is invalid
        """
        if concept_id in self.valid_concepts:
            return concept_id
        
        # Raise exception for invalid concept - no fallbacks
        error_msg = f"Invalid concept ID {concept_id}"
        if fallback_key:
            error_msg += f" for {fallback_key}"
        
        # Check if we have a fallback concept defined but it's also invalid
        if fallback_key and fallback_key in self._fallback_concepts:
            fallback_id = self._fallback_concepts[fallback_key]
            if fallback_id not in self.valid_concepts:
                error_msg += f" (fallback concept {fallback_id} is also invalid)"
        
        raise DataGenerationError(
            error_msg,
            context={
                "concept_id": concept_id, 
                "fallback_key": fallback_key,
                "available_concepts": sorted(list(self.valid_concepts))[:10]  # Show first 10 for debugging
            }
        )
    
    def get_fallback_concept(self, key: str) -> Optional[int]:
        """Get a specific fallback concept by key.
        
        Args:
            key: The fallback concept key
            
        Returns:
            Concept ID if found, None otherwise
        """
        return self._fallback_concepts.get(key)
    
    def get_fallback_concepts(self) -> Dict[str, int]:
        """Return all available fallback concepts."""
        return self._fallback_concepts.copy()
    
    def is_valid_concept(self, concept_id: int) -> bool:
        """Check if a concept ID is valid.
        
        Args:
            concept_id: The concept ID to check
            
        Returns:
            True if valid, False otherwise
        """
        return concept_id in self.valid_concepts
    
    def validate_concept_with_fallback(self, concept_id: int, fallback_key: Optional[str] = None) -> int:
        """Validate a concept ID and return fallback if invalid (legacy behavior).
        
        Args:
            concept_id: The concept ID to validate
            fallback_key: Key for fallback concept if validation fails
            
        Returns:
            Valid concept ID (original or fallback)
        """
        if concept_id in self.valid_concepts:
            return concept_id
        
        # Try to use fallback
        if fallback_key and fallback_key in self._fallback_concepts:
            fallback_id = self._fallback_concepts[fallback_key]
            logger.warning(
                f"Invalid concept ID {concept_id}, using fallback {fallback_id} for {fallback_key}"
            )
            return fallback_id
        
        # Use generic fallback (first concept in dictionary)
        if self.valid_concepts:
            fallback_id = min(self.valid_concepts)
            logger.warning(
                f"Invalid concept ID {concept_id}, using generic fallback {fallback_id}"
            )
            return fallback_id
        
        # This should never happen if concept dictionary is loaded
        raise DataGenerationError(
            f"No valid concepts available for fallback. Invalid concept ID: {concept_id}",
            context={"concept_id": concept_id, "fallback_key": fallback_key}
        )
    
    def get_concept_info(self, concept_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a concept.
        
        Args:
            concept_id: The concept ID to look up
            
        Returns:
            Dictionary with concept information or None if not found
        """
        matching_concepts = self.concept_dict[
            self.concept_dict['concept_id'] == concept_id
        ]
        
        if matching_concepts.empty:
            return None
        
        concept_row = matching_concepts.iloc[0]
        return {
            'concept_id': int(concept_row['concept_id']),
            'concept_name': concept_row['concept_name'],
            'domain_id': concept_row['domain_id'],
            'vocabulary_id': concept_row['vocabulary_id'],
            'concept_class_id': concept_row['concept_class_id']
        }