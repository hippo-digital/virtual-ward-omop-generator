"""OMOP CDM table definitions and schemas."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import pandas as pd
from .base import BaseModel


@dataclass
class OMOPTableSchema:
    """Schema definition for an OMOP table."""
    
    name: str
    columns: Dict[str, str]  # column_name -> data_type
    primary_key: Optional[str] = None
    foreign_keys: Optional[Dict[str, str]] = None  # column -> referenced_table.column
    indexes: Optional[List[str]] = None


class OMOPTables(BaseModel):
    """OMOP CDM table definitions for virtual ward scenarios."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._define_schemas()
    
    def _define_schemas(self) -> None:
        """Define OMOP table schemas."""
        self.schemas = {
            "location": OMOPTableSchema(
                name="location",
                columns={
                    "location_id": "INTEGER", 
                    "address_1": "VARCHAR",
                    "address_2": "VARCHAR",
                    "city": "VARCHAR",
                    "state": "VARCHAR",
                    "county": "VARCHAR",
                    "zip": "VARCHAR(20)",
                    "country_concept_id": "INTEGER",
                    "latitude": "DOUBLE",
                    "longitude": "DOUBLE",
                    "location_source_value": "VARCHAR",
                    "country_source_value": "VARCHAR",
                },
            ),
            "person": OMOPTableSchema(
                name="person",
                columns={
                    'person_id': 'INTEGER',
                    'gender_concept_id': 'INTEGER',
                    'year_of_birth': 'INTEGER',
                    'month_of_birth': 'INTEGER',
                    'day_of_birth': 'INTEGER',
                    'birth_datetime': 'TIMESTAMP',
                    'race_concept_id': 'INTEGER',
                    'race_source_concept_id': 'INTEGER',
                    'care_site_id': 'INTEGER',
                    'gender_source_concept_id': 'INTEGER',
                    'gender_source_value': 'VARCHAR',
                    'person_source_value': 'VARCHAR',
                    'provider_id': 'INTEGER',
                    'race_concept_id': 'INTEGER',
                    'race_source_value': 'VARCHAR',
                    'ethnicity_concept_id': 'INTEGER',
                    'ethnicity_source_concept_id': 'INTEGER',
                    'ethnicity_source_value': 'VARCHAR',
                    'location_id': 'INTEGER'
                },
                primary_key='person_id'
            ),
            
            'observation_period': OMOPTableSchema(
                name='observation_period',
                columns={
                    'observation_period_id': 'INTEGER',
                    'person_id': 'INTEGER',
                    'observation_period_start_date': 'DATE',
                    'observation_period_end_date': 'DATE',
                    'period_type_concept_id': 'INTEGER'
                },
                primary_key='observation_period_id',
                foreign_keys={'person_id': 'person.person_id'},
                indexes=['person_id']
            ),
            
            'episode': OMOPTableSchema(
                name='episode',
                columns={
                    'episode_id': 'INTEGER',
                    'person_id': 'INTEGER',
                    'episode_concept_id': 'INTEGER',
                    'episode_start_date': 'DATE',
                    'episode_start_datetime': 'TIMESTAMP',
                    'episode_end_date': 'DATE',
                    'episode_end_datetime': 'TIMESTAMP',
                    'episode_parent_id': 'INTEGER',
                    'episode_number': 'INTEGER',
                    'episode_object_concept_id': 'INTEGER',
                    'episode_type_concept_id': 'INTEGER',
                    'episode_source_value': 'VARCHAR(50)',
                    'episode_source_concept_id': 'INTEGER'                
                },
                primary_key='episode_id',
                foreign_keys={'person_id': 'person.person_id'},
                indexes=['person_id', 'episode_start_datetime']
            ),
            
            'visit_occurrence': OMOPTableSchema(
                name='visit_occurrence',
                columns={
                    'visit_occurrence_id': 'INTEGER',
                    'person_id': 'INTEGER',
                    'visit_concept_id': 'INTEGER',
                    'visit_start_date': 'DATE',
                    'visit_start_datetime': 'TIMESTAMP',
                    'visit_end_date': 'DATE',
                    'visit_end_datetime': 'TIMESTAMP',
                    'visit_type_concept_id': 'INTEGER',
                    'provider_id': 'INTEGER',
                    'care_site_id': 'INTEGER',
                    'visit_source_value': 'VARCHAR(50)',
                    'visit_source_concept_id': 'INTEGER',
                    'admitted_from_concept_id': 'INTEGER',
                    'admitted_from_source_value': 'VARCHAR(50)',
                    'discharged_to_concept_id': 'INTEGER',
                    'discharged_to_source_value': 'VARCHAR(50)',
                    'preceding_visit_occurrence_id': 'INTEGER'
                },
                primary_key='visit_occurrence_id',
                foreign_keys={'person_id': 'person.person_id'},
                indexes=['person_id', 'visit_start_datetime']
            ),
            
            'survey_conduct': OMOPTableSchema(
                name='survey_conduct',
                columns={
                    'survey_conduct_id': 'INTEGER',
                    'person_id': 'INTEGER',
                    'survey_concept_id': 'INTEGER',
                    'survey_start_date': 'DATE',
                    'survey_start_datetime': 'TIMESTAMP',
                    'survey_end_date': 'DATE',
                    'survey_end_datetime': 'TIMESTAMP',
                    'provider_id': 'INTEGER',
                    'assisted_concept_id': 'INTEGER',
                    'respondent_type_concept_id': 'INTEGER',
                    'timing_concept_id': 'INTEGER',
                    'collection_method_concept_id': 'INTEGER',
                    'assisted_source_value': 'VARCHAR(50)',
                    'respondent_type_source_value': 'VARCHAR(50)',
                    'timing_source_value': 'VARCHAR(50)',
                    'collection_method_source_value': 'VARCHAR(50)',
                    'survey_source_value': 'VARCHAR(50)',
                    'survey_source_concept_id': 'INTEGER',
                    'survey_source_identifier': 'VARCHAR(50)',
                    'validated_survey_concept_id': 'INTEGER',
                    'validated_survey_source_value': 'VARCHAR(50)',
                    'survey_version_number': 'VARCHAR(20)',
                    'visit_occurrence_id': 'INTEGER',
                    'response_visit_occurrence_id': 'INTEGER'
                },
                primary_key='survey_conduct_id',
                foreign_keys={
                    'person_id': 'person.person_id',
                    'visit_occurrence_id': 'visit_occurrence.visit_occurrence_id'
                },
                indexes=['person_id', 'survey_start_datetime']
            ),
            
            'observation': OMOPTableSchema(
                name='observation',
                columns={
                    'observation_id': 'INTEGER',
                    'person_id': 'INTEGER',
                    'observation_concept_id': 'INTEGER',
                    'observation_date': 'DATE',
                    'observation_datetime': 'TIMESTAMP',
                    'observation_type_concept_id': 'INTEGER',
                    'value_as_number': 'DOUBLE',
                    'value_as_string': 'VARCHAR(60)',
                    'value_as_concept_id': 'INTEGER',
                    'qualifier_concept_id': 'INTEGER',
                    'unit_concept_id': 'INTEGER',
                    'provider_id': 'INTEGER',
                    'visit_occurrence_id': 'INTEGER',
                    'visit_detail_id': 'INTEGER',
                    'observation_source_value': 'VARCHAR(50)',
                    'observation_source_concept_id': 'INTEGER',
                    'unit_source_value': 'VARCHAR(50)',
                    'qualifier_source_value': 'VARCHAR(50)',
                    'value_source_value': 'VARCHAR(50)',
                    'observation_event_id': 'INTEGER',
                    'obs_event_field_concept_id': 'INTEGER'
                },
                primary_key='observation_id',
                foreign_keys={
                    'person_id': 'person.person_id',
                    'visit_occurrence_id': 'visit_occurrence.visit_occurrence_id'
                },
                indexes=['person_id', 'observation_datetime', 'observation_concept_id']
            ),
            
            'measurement': OMOPTableSchema(
                name='measurement',
                columns={
                    'measurement_id': 'INTEGER',
                    'person_id': 'INTEGER',
                    'measurement_concept_id': 'INTEGER',
                    'measurement_date': 'DATE',
                    'measurement_datetime': 'TIMESTAMP',
                    'measurement_time': 'VARCHAR(10)',
                    'measurement_type_concept_id': 'INTEGER',
                    'operator_concept_id': 'INTEGER',
                    'value_as_number': 'DOUBLE',
                    'value_as_concept_id': 'INTEGER',
                    'unit_concept_id': 'INTEGER',
                    'range_low': 'DOUBLE',
                    'range_high': 'DOUBLE',
                    'provider_id': 'INTEGER',
                    'visit_occurrence_id': 'INTEGER',
                    'visit_detail_id': 'INTEGER',
                    'measurement_source_value': 'VARCHAR(50)',
                    'measurement_source_concept_id': 'INTEGER',
                    'unit_source_value': 'VARCHAR(50)',
                    'unit_source_concept_id': 'INTEGER',
                    'value_source_value': 'VARCHAR(50)',
                    'measurement_event_id': 'INTEGER',
                    'meas_event_field_concept_id': 'INTEGER'
                },
                primary_key='measurement_id',
                foreign_keys={
                    'person_id': 'person.person_id',
                    'visit_occurrence_id': 'visit_occurrence.visit_occurrence_id'
                },
                indexes=['person_id', 'measurement_datetime', 'measurement_concept_id']
            ),
            
            'condition_occurrence': OMOPTableSchema(
                name='condition_occurrence',
                columns={
                    'condition_occurrence_id': 'INTEGER',
                    'person_id': 'INTEGER',
                    'condition_concept_id': 'INTEGER',
                    'condition_start_date': 'DATE',
                    'condition_start_datetime': 'TIMESTAMP',
                    'condition_end_date': 'DATE',
                    'condition_end_datetime': 'TIMESTAMP',
                    'condition_type_concept_id': 'INTEGER',
                    'condition_status_concept_id': 'INTEGER',
                    'stop_reason': 'VARCHAR(20)',
                    'provider_id': 'INTEGER',
                    'visit_occurrence_id': 'INTEGER',
                    'visit_detail_id': 'INTEGER',
                    'condition_source_value': 'VARCHAR(50)',
                    'condition_source_concept_id': 'INTEGER',
                    'condition_status_source_value': 'VARCHAR(50)'
                },
                primary_key='condition_occurrence_id',
                foreign_keys={
                    'person_id': 'person.person_id',
                    'visit_occurrence_id': 'visit_occurrence.visit_occurrence_id'
                },
                indexes=['person_id', 'condition_start_datetime', 'condition_concept_id']
            ),
            
            'drug_exposure': OMOPTableSchema(
                name='drug_exposure',
                columns={
                    'drug_exposure_id': 'INTEGER',
                    'person_id': 'INTEGER',
                    'drug_concept_id': 'INTEGER',
                    'drug_exposure_start_date': 'DATE',
                    'drug_exposure_start_datetime': 'TIMESTAMP',
                    'drug_exposure_end_date': 'DATE',
                    'drug_exposure_end_datetime': 'TIMESTAMP',
                    'verbatim_end_date': 'DATE',
                    'drug_type_concept_id': 'INTEGER',
                    'stop_reason': 'VARCHAR(20)',
                    'refills': 'INTEGER',
                    'quantity': 'DOUBLE',
                    'days_supply': 'INTEGER',
                    'sig': 'TEXT',
                    'route_concept_id': 'INTEGER',
                    'lot_number': 'VARCHAR(50)',
                    'provider_id': 'INTEGER',
                    'visit_occurrence_id': 'INTEGER',
                    'visit_detail_id': 'INTEGER',
                    'drug_source_value': 'VARCHAR(50)',
                    'drug_source_concept_id': 'INTEGER',
                    'route_source_value': 'VARCHAR(50)',
                    'dose_unit_source_value': 'VARCHAR(50)'
                },
                primary_key='drug_exposure_id',
                foreign_keys={
                    'person_id': 'person.person_id',
                    'visit_occurrence_id': 'visit_occurrence.visit_occurrence_id'
                },
                indexes=['person_id', 'drug_exposure_start_datetime', 'drug_concept_id']
            ),
            
            'procedure_occurrence': OMOPTableSchema(
                name='procedure_occurrence',
                columns={
                    'procedure_occurrence_id': 'INTEGER',
                    'person_id': 'INTEGER',
                    'procedure_concept_id': 'INTEGER',
                    'procedure_date': 'DATE',
                    'procedure_datetime': 'TIMESTAMP',
                    'procedure_end_date': 'DATE',
                    'procedure_end_datetime': 'TIMESTAMP',
                    'procedure_type_concept_id': 'INTEGER',
                    'modifier_concept_id': 'INTEGER',
                    'quantity': 'INTEGER',
                    'provider_id': 'INTEGER',
                    'visit_occurrence_id': 'INTEGER',
                    'visit_detail_id': 'INTEGER',
                    'procedure_source_value': 'VARCHAR(50)',
                    'procedure_source_concept_id': 'INTEGER',
                    'modifier_source_value': 'VARCHAR(50)'
                },
                primary_key='procedure_occurrence_id',
                foreign_keys={
                    'person_id': 'person.person_id',
                    'visit_occurrence_id': 'visit_occurrence.visit_occurrence_id'
                },
                indexes=['person_id', 'procedure_datetime', 'procedure_concept_id']
            ),
            
            'device_exposure': OMOPTableSchema(
                name='device_exposure',
                columns={
                    'device_exposure_id': 'INTEGER',
                    'person_id': 'INTEGER',
                    'device_concept_id': 'INTEGER',
                    'device_exposure_start_date': 'DATE',
                    'device_exposure_start_datetime': 'TIMESTAMP',
                    'device_exposure_end_date': 'DATE',
                    'device_exposure_end_datetime': 'TIMESTAMP',
                    'device_type_concept_id': 'INTEGER',
                    'unique_device_id': 'VARCHAR(255)',
                    'production_id': 'VARCHAR(255)',
                    'quantity': 'INTEGER',
                    'provider_id': 'INTEGER',
                    'visit_occurrence_id': 'INTEGER',
                    'visit_detail_id': 'INTEGER',
                    'device_source_value': 'VARCHAR(50)',
                    'device_source_concept_id': 'INTEGER',
                    'unit_concept_id': 'INTEGER',
                    'unit_source_value': 'VARCHAR(50)',
                    'unit_source_concept_id': 'INTEGER'
                },
                primary_key='device_exposure_id',
                foreign_keys={
                    'person_id': 'person.person_id',
                    'visit_occurrence_id': 'visit_occurrence.visit_occurrence_id'
                },
                indexes=['person_id', 'device_exposure_start_datetime', 'device_concept_id']
            ),
            
            'death': OMOPTableSchema(
                name='death',
                columns={
                    'person_id': 'INTEGER',
                    'death_date': 'DATE',
                    'death_datetime': 'TIMESTAMP',
                    'death_type_concept_id': 'INTEGER',
                    'cause_concept_id': 'INTEGER',
                    'cause_source_value': 'VARCHAR(50)',
                    'cause_source_concept_id': 'INTEGER'
                },
                primary_key='person_id',
                foreign_keys={'person_id': 'person.person_id'}
            ),

            "care_site": OMOPTableSchema(
                name="care_site",
                columns={
                    'care_site_id': 'INTEGER',
                    'care_site_name': 'VARCHAR(255)',
                    'place_of_service_concept_id': 'INTEGER',
                    'location_id': 'INTEGER',
                    'care_site_source_value': 'VARCHAR(50)',
                    'place_of_service_source_value': 'VARCHAR(50)'
                },
                primary_key='care_site_id'
            ),

            "cdm_source": OMOPTableSchema(
                name="cdm_source",
                columns={
                    'cdm_source_name': 'VARCHAR(255)',
                    'cdm_source_abbreviation': 'VARCHAR(25)',
                    'cdm_holder': 'VARCHAR(255)',
                    'source_description': 'TEXT',
                    'source_documentation_reference': 'VARCHAR(255)',
                    'cdm_etl_reference': 'VARCHAR(255)',
                    'source_release_date': 'DATE',
                    'cdm_release_date': 'DATE',
                    'cdm_version': 'VARCHAR(10)',
                    'cdm_version_concept_id': 'INTEGER',
                    'vocabulary_version': 'VARCHAR(20)'
                },
                primary_key='cdm_source_name' # Using cdm_source_name as a functional key
            ),

            "cohort": OMOPTableSchema(
                name="cohort",
                columns={
                    'cohort_definition_id': 'INTEGER',
                    'subject_id': 'INTEGER',
                    'cohort_start_date': 'DATE',
                    'cohort_end_date': 'DATE'
                },
                primary_key='cohort_definition_id' 
            ),
            "cohort_definition": OMOPTableSchema(
                name="cohort_definition",
                columns={
                    'cohort_definition_id': 'INTEGER',
                    'cohort_definition_name': 'VARCHAR(255)',
                    'cohort_definition_description': 'TEXT',
                    'definition_type_concept_id': 'INTEGER',
                    'cohort_definition_syntax': 'TEXT',
                    'subject_concept_id': 'INTEGER',
                    'cohort_initiation_date': 'DATE'
                },
                primary_key='cohort_definition_id'
            ),

            "concept": OMOPTableSchema(
                name="concept",
                columns={
                    'concept_id': 'INTEGER',
                    'concept_name': 'VARCHAR(255)',
                    'domain_id': 'VARCHAR(20)',
                    'vocabulary_id': 'VARCHAR(20)',
                    'concept_class_id': 'VARCHAR(20)',
                    'standard_concept': 'VARCHAR(1)',
                    'concept_code': 'VARCHAR(50)',
                    'valid_start_date': 'DATE',
                    'valid_end_date': 'DATE',
                    'invalid_reason': 'VARCHAR(1)'
                },
                primary_key='concept_id'
            ),

            "concept_ancestor": OMOPTableSchema(
                name="concept_ancestor",
                columns={
                    'ancestor_concept_id': 'INTEGER',
                    'descendant_concept_id': 'INTEGER',
                    'min_levels_of_separation': 'INTEGER',
                    'max_levels_of_separation': 'INTEGER'
                },
                primary_key='ancestor_concept_id' 
            ),

            "concept_class": OMOPTableSchema(
                name="concept_class",
                columns={
                    'concept_class_id': 'VARCHAR(20)',
                    'concept_class_name': 'VARCHAR(255)',
                    'concept_class_concept_id': 'INTEGER'
                },
                primary_key='concept_class_id'
            ),

            "concept_relationship": OMOPTableSchema(
                name="concept_relationship",
                columns={
                    'concept_id_1': 'INTEGER',
                    'concept_id_2': 'INTEGER',
                    'relationship_id': 'VARCHAR(20)',
                    'valid_start_date': 'DATE',
                    'valid_end_date': 'DATE',
                    'invalid_reason': 'VARCHAR(1)'
                },
                primary_key='concept_id_1' 
            ),

            "concept_synonym": OMOPTableSchema(
                name="concept_synonym",
                columns={
                    'concept_id': 'INTEGER',
                    'concept_synonym_name': 'VARCHAR(1000)',
                    'language_concept_id': 'INTEGER'
                },
                primary_key='concept_id' 
            ),

            "condition_era": OMOPTableSchema(
                name="condition_era",
                columns={
                    'condition_era_id': 'INTEGER',
                    'person_id': 'INTEGER',
                    'condition_concept_id': 'INTEGER',
                    'condition_era_start_date': 'DATE',
                    'condition_era_end_date': 'DATE',
                    'condition_occurrence_count': 'INTEGER'
                },
                primary_key='condition_era_id'
            ),
            "drug_era": OMOPTableSchema(
                name="drug_era",
                columns={
                    'drug_era_id': 'INTEGER',
                    'person_id': 'INTEGER',
                    'drug_concept_id': 'INTEGER',
                    'drug_era_start_date': 'DATE',
                    'drug_era_end_date': 'DATE',
                    'drug_exposure_count': 'INTEGER',
                    'gap_days': 'INTEGER'
                },
                primary_key='drug_era_id'
            ),
            "dose_era": OMOPTableSchema(
                name="dose_era",
                columns={
                    'dose_era_id': 'INTEGER',
                    'person_id': 'INTEGER',
                    'drug_concept_id': 'INTEGER',
                    'unit_concept_id': 'INTEGER',
                    'dose_value': 'NUMERIC',
                    'dose_era_start_date': 'DATE',
                    'dose_era_end_date': 'DATE'
                },
                primary_key='dose_era_id'
            ),

            "cost": OMOPTableSchema(
                name="cost",
                columns={
                    'cost_id': 'INTEGER',
                    'cost_event_id': 'INTEGER',
                    'cost_domain_id': 'VARCHAR(20)',
                    'cost_type_concept_id': 'INTEGER',
                    'currency_concept_id': 'INTEGER',
                    'total_charge': 'NUMERIC',
                    'total_cost': 'NUMERIC',
                    'total_paid': 'NUMERIC',
                    'paid_by_payer': 'NUMERIC',
                    'paid_by_patient': 'NUMERIC',
                    'paid_patient_copay': 'NUMERIC',
                    'paid_patient_coinsurance': 'NUMERIC',
                    'paid_patient_deductible': 'NUMERIC',
                    'paid_by_primary': 'NUMERIC',
                    'paid_ingredient_cost': 'NUMERIC',
                    'paid_dispensing_fee': 'NUMERIC',
                    'payer_plan_period_id': 'INTEGER',
                    'amount_allowed': 'NUMERIC',
                    'revenue_code_concept_id': 'INTEGER',
                    'revenue_code_source_value': 'VARCHAR(50)',
                    'drg_concept_id': 'INTEGER',
                    'drg_source_value': 'VARCHAR(3)'
                },
                primary_key='cost_id'
            ),

            "vocabulary": OMOPTableSchema(
                name="vocabulary",
                columns={
                    'vocabulary_id': 'VARCHAR(20)',
                    'vocabulary_name': 'VARCHAR(255)',
                    'vocabulary_reference': 'VARCHAR(255)',
                    'vocabulary_version': 'VARCHAR(255)',
                    'vocabulary_concept_id': 'INTEGER'
                },
                primary_key='vocabulary_id'
            ),

            "domain": OMOPTableSchema(
                name="domain",
                columns={
                    'domain_id': 'VARCHAR(20)',
                    'domain_name': 'VARCHAR(255)',
                    'domain_concept_id': 'INTEGER'
                },
                primary_key='domain_id'
            ),

            "drug_strength": OMOPTableSchema(
                name="drug_strength",
                columns={
                    'drug_concept_id': 'INTEGER',
                    'ingredient_concept_id': 'INTEGER',
                    'amount_value': 'NUMERIC',
                    'amount_unit_concept_id': 'INTEGER',
                    'numerator_value': 'NUMERIC',
                    'numerator_unit_concept_id': 'INTEGER',
                    'denominator_value': 'NUMERIC',
                    'denominator_unit_concept_id': 'INTEGER',
                    'box_size': 'INTEGER',
                    'valid_start_date': 'DATE',
                    'valid_end_date': 'DATE',
                    'invalid_reason': 'VARCHAR(1)'
                },
                primary_key='drug_concept_id' 
            ),

            "episode_event": OMOPTableSchema(
                name="episode_event",
                columns={
                    'episode_id': 'INTEGER',
                    'event_id': 'INTEGER',
                    'episode_event_field_concept_id': 'INTEGER'
                },
                primary_key='episode_id' 
            ),

            "fact_relationship": OMOPTableSchema(
                name="fact_relationship",
                columns={
                    'domain_concept_id_1': 'INTEGER',
                    'fact_id_1': 'INTEGER',
                    'domain_concept_id_2': 'INTEGER',
                    'fact_id_2': 'INTEGER',
                    'relationship_concept_id': 'INTEGER'
                },
                primary_key='domain_concept_id_1' 
            ),

            "metadata": OMOPTableSchema(
                name="metadata",
                columns={
                    'metadata_id': 'INTEGER',
                    'metadata_concept_id': 'INTEGER',
                    'metadata_type_concept_id': 'INTEGER',
                    'name': 'VARCHAR(250)',
                    'value_as_string': 'VARCHAR(250)',
                    'value_as_concept_id': 'INTEGER',
                    'value_as_number': 'NUMERIC',
                    'metadata_date': 'DATE',
                    'metadata_datetime': 'TIMESTAMP'
                },
                primary_key='metadata_id'
            ),

            "note": OMOPTableSchema(
                name="note",
                columns={
                    'note_id': 'INTEGER',
                    'person_id': 'INTEGER',
                    'note_date': 'DATE',
                    'note_datetime': 'TIMESTAMP',
                    'note_type_concept_id': 'INTEGER',
                    'note_class_concept_id': 'INTEGER',
                    'note_title': 'VARCHAR(250)',
                    'note_text': 'TEXT',
                    'encoding_concept_id': 'INTEGER',
                    'language_concept_id': 'INTEGER',
                    'provider_id': 'INTEGER',
                    'visit_occurrence_id': 'INTEGER',
                    'visit_detail_id': 'INTEGER',
                    'note_source_value': 'VARCHAR(50)',
                    'note_event_id': 'INTEGER',
                    'note_event_field_concept_id': 'INTEGER'
                },
                primary_key='note_id'
            ),
            "note_nlp": OMOPTableSchema(
                name="note_nlp",
                columns={
                    'note_nlp_id': 'INTEGER',
                    'note_id': 'INTEGER',
                    'section_concept_id': 'INTEGER',
                    'snippet': 'VARCHAR(250)',
                    'offset': 'VARCHAR(50)',
                    'lexical_variant': 'VARCHAR(250)',
                    'note_nlp_concept_id': 'INTEGER',
                    'note_nlp_source_concept_id': 'INTEGER',
                    'nlp_system': 'VARCHAR(250)',
                    'nlp_date': 'DATE',
                    'nlp_datetime': 'TIMESTAMP',
                    'term_exists': 'VARCHAR(1)',
                    'term_temporal': 'VARCHAR(50)',
                    'term_modifiers': 'VARCHAR(2000)'
                },
                primary_key='note_nlp_id'
            ),

            "provider": OMOPTableSchema(
                name="provider",
                columns={
                    'provider_id': 'INTEGER',
                    'provider_name': 'VARCHAR(255)',
                    'npi': 'VARCHAR(20)',
                    'dea': 'VARCHAR(20)',
                    'specialty_concept_id': 'INTEGER',
                    'care_site_id': 'INTEGER',
                    'year_of_birth': 'INTEGER',
                    'gender_concept_id': 'INTEGER',
                    'provider_source_value': 'VARCHAR(50)',
                    'specialty_source_value': 'VARCHAR(50)',
                    'specialty_source_concept_id': 'INTEGER',
                    'gender_source_value': 'VARCHAR(50)',
                    'gender_source_concept_id': 'INTEGER'
                },
                primary_key='provider_id'
            ),
            "payer_plan_period": OMOPTableSchema(
                name="payer_plan_period",
                columns={
                    'payer_plan_period_id': 'INTEGER',
                    'person_id': 'INTEGER',
                    'payer_plan_period_start_date': 'DATE',
                    'payer_plan_period_end_date': 'DATE',
                    'payer_concept_id': 'INTEGER',
                    'payer_source_value': 'VARCHAR(50)',
                    'payer_source_concept_id': 'INTEGER',
                    'plan_concept_id': 'INTEGER',
                    'plan_source_value': 'VARCHAR(50)',
                    'plan_source_concept_id': 'INTEGER',
                    'sponsor_concept_id': 'INTEGER',
                    'sponsor_source_value': 'VARCHAR(50)',
                    'sponsor_source_concept_id': 'INTEGER',
                    'family_source_value': 'VARCHAR(50)',
                    'stop_reason_concept_id': 'INTEGER',
                    'stop_reason_source_value': 'VARCHAR(50)',
                    'stop_reason_source_concept_id': 'INTEGER'
                },
                primary_key='payer_plan_period_id'
            ),

            "relationship": OMOPTableSchema(
                name="relationship",
                columns={
                    'relationship_id': 'VARCHAR(20)',
                    'relationship_name': 'VARCHAR(255)',
                    'is_hierarchical': 'VARCHAR(1)',
                    'defines_ancestry': 'VARCHAR(1)',
                    'reverse_relationship_id': 'VARCHAR(20)',
                    'relationship_concept_id': 'INTEGER'
                },
                primary_key='relationship_id'
            ),

            "source_to_concept_map": OMOPTableSchema(
                name="source_to_concept_map",
                columns={
                    'source_code': 'VARCHAR(50)',
                    'source_concept_id': 'INTEGER',
                    'source_vocabulary_id': 'VARCHAR(20)',
                    'source_code_description': 'VARCHAR(255)',
                    'target_concept_id': 'INTEGER',
                    'target_vocabulary_id': 'VARCHAR(20)',
                    'valid_start_date': 'DATE',
                    'valid_end_date': 'DATE',
                    'invalid_reason': 'VARCHAR(1)'
                },
                primary_key='source_code' 
            ),

            "specimen": OMOPTableSchema(
                name="specimen",
                columns={
                    'specimen_id': 'INTEGER',
                    'person_id': 'INTEGER',
                    'specimen_concept_id': 'INTEGER',
                    'specimen_type_concept_id': 'INTEGER',
                    'specimen_date': 'DATE',
                    'specimen_datetime': 'TIMESTAMP',
                    'quantity': 'NUMERIC',
                    'unit_concept_id': 'INTEGER',
                    'anatomic_site_concept_id': 'INTEGER',
                    'disease_status_concept_id': 'INTEGER',
                    'specimen_source_id': 'VARCHAR(50)',
                    'specimen_source_value': 'VARCHAR(50)',
                    'unit_source_value': 'VARCHAR(50)',
                    'anatomic_site_source_value': 'VARCHAR(50)',
                    'disease_status_source_value': 'VARCHAR(50)'
                },
                primary_key='specimen_id'
            ),

            "visit_detail": OMOPTableSchema(
                name="visit_detail",
                columns={
                    'visit_detail_id': 'INTEGER',
                    'person_id': 'INTEGER',
                    'visit_detail_concept_id': 'INTEGER',
                    'visit_detail_start_date': 'DATE',
                    'visit_detail_start_datetime': 'TIMESTAMP',
                    'visit_detail_end_date': 'DATE',
                    'visit_detail_end_datetime': 'TIMESTAMP',
                    'visit_detail_type_concept_id': 'INTEGER',
                    'provider_id': 'INTEGER',
                    'care_site_id': 'INTEGER',
                    'visit_detail_source_value': 'VARCHAR(50)',
                    'visit_detail_source_concept_id': 'INTEGER',
                    'admitted_from_concept_id': 'INTEGER',
                    'admitted_from_source_value': 'VARCHAR(50)',
                    'discharged_to_source_value': 'VARCHAR(50)',
                    'discharged_to_concept_id': 'INTEGER',
                    'preceding_visit_detail_id': 'INTEGER',
                    'parent_visit_detail_id': 'INTEGER',
                    'visit_occurrence_id': 'INTEGER'
                },
                primary_key='visit_detail_id'
            ),

            "vocabulary": OMOPTableSchema(
                name="vocabulary",
                columns={
                    'vocabulary_id': 'VARCHAR(20)',
                    'vocabulary_name': 'VARCHAR(255)',
                    'vocabulary_reference': 'VARCHAR(255)',
                    'vocabulary_version': 'VARCHAR(255)',
                    'vocabulary_concept_id': 'INTEGER'
                },
                primary_key='vocabulary_id'
            ),
        }
    
    def validate(self) -> bool:
        """Validate OMOP table schemas.
        
        Returns:
            True if schemas are valid
        """
        required_tables = [
            'location',
            'person', 'observation_period', 'episode', 'visit_occurrence',
            'survey_conduct', 'observation', 'measurement', 'condition_occurrence',
            'drug_exposure', 'procedure_occurrence', 'device_exposure', 'death'
        ]
        
        for table in required_tables:
            if table not in self.schemas:
                self.logger.error(f"Missing required OMOP table schema: {table}")
                return False
        
        self.logger.info("OMOP table schemas validated successfully")
        return True
    
    def get_schema(self, table_name: str) -> Optional[OMOPTableSchema]:
        """Get schema for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Table schema or None if not found
        """
        return self.schemas.get(table_name)
    
    def get_all_schemas(self) -> Dict[str, OMOPTableSchema]:
        """Get all table schemas.
        
        Returns:
            Dictionary of table name to schema
        """
        return self.schemas.copy()
    
    def get_table_names(self) -> List[str]:
        """Get list of all table names.
        
        Returns:
            List of table names
        """
        return list(self.schemas.keys())