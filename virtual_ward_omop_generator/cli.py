"""Command-line interface for Virtual Ward OMOP Generator."""

import argparse
import random
import sys
import time
from pathlib import Path
import pandas as pd
import requests
from tqdm.auto import trange

from .logging_config import setup_logging, get_logger
from .config import ConfigurationManager
from .generators import PopulationGenerator, EpisodeManager, VisitGenerator, SignalGenerator, InterventionEngine
from .validation import DataValidator, ConceptValidator
from .utils import TemporalCoordinator
from .output import DuckDBWriter
from .exceptions import ConfigurationError, DataGenerationError


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Virtual Ward OMOP Generator - Generate synthetic OMOP CDM data for virtual ward scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dataset with default settings
  %(prog)s --config config.yaml --output data.db

  # Generate with custom population size and seed
  %(prog)s --config config.yaml --output data.db --persons 1500 --seed 42

  # Generate with verbose logging
  %(prog)s --config config.yaml --output data.db --log-level DEBUG --log-file generation.log

  # Validate existing configuration
  %(prog)s --config config.yaml --validate-only

  # Show configuration template
  %(prog)s --show-template
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for DuckDB database"
    )
    
    # Optional generation parameters
    parser.add_argument(
        "--persons",
        type=int,
        help="Number of persons to generate (overrides config file)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible generation (overrides config file)"
    )
    parser.add_argument(
        "--concepts",
        type=str,
        help="Path to concept dictionary CSV file (overrides config file)"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Optional log file path"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output (equivalent to --log-level DEBUG)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output (equivalent to --log-level WARNING)"
    )
    
    # Utility options
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without generating data"
    )
    parser.add_argument(
        "--show-template",
        action="store_true",
        help="Show example configuration template and exit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without writing output"
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        default=True,
        help="Show progress indicators (default: enabled)"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress indicators"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip data validation and create database even with errors (for testing)"
    )
    
    return parser


def show_template() -> None:
    """Display a basic configuration template."""
    template = """# Virtual Ward OMOP Generator Configuration Template
# Copy this template and customize for your needs

schema_version: 0.1

vocabulary:
  id: "VH_VOCAB"
  concept_file: "./briefing/omop_virtual_ward_concepts.csv"

dataset:
  name: "virtual_ward_omop_synth"
  random_seed: 42
  persons: 1200

  # Population & episode mix
  condition_mix:
    2100000300: 0.45   # COPD
    2100000301: 0.45   # Heart Failure
    2100000302: 0.10   # Post-op

  episodes_per_person:
    min: 1
    max: 2

  episode_length_days:
    modes: [7, 14, 21]
    probs: [0.3, 0.4, 0.3]

  # Trajectory archetypes
  episode_archetype_mix:
    stable: 0.40
    flare_mild: 0.25
    flare_moderate: 0.20
    flare_severe: 0.10
    noisy_reporter: 0.05

# For complete configuration options, see the documentation
# or example files in the examples/ directory
"""
    print(template)


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if args.show_template:
        return
        
    if not args.config and not args.show_template:
        raise ValueError("--config is required unless using --show-template")
        
    if not args.output and not args.validate_only and not args.show_template:
        raise ValueError("--output is required unless using --validate-only or --show-template")
        
    if args.config and not Path(args.config).exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
        
    if args.persons is not None and (args.persons < 1 or args.persons > 20000):
        raise ValueError("--persons must be between 1 and 20000")
        
    if args.verbose and args.quiet:
        raise ValueError("Cannot specify both --verbose and --quiet")


def setup_logging_from_args(args: argparse.Namespace) -> None:
    """Set up logging based on command line arguments."""
    log_level = args.log_level
    
    if args.verbose:
        log_level = "DEBUG"
    elif args.quiet:
        log_level = "WARNING"
        
    setup_logging(level=log_level, log_file=args.log_file)


def run_generation_pipeline(config_manager: ConfigurationManager, config: dict, 
                          output_path: str, args: argparse.Namespace) -> None:
    """Run the complete data generation pipeline."""
    logger = get_logger(__name__)
    start_time = time.time()
    
    # Override config with command line arguments
    if args.persons is not None:
        config['dataset']['persons'] = args.persons
    if args.seed is not None:
        config['dataset']['random_seed'] = args.seed
        
    persons_count = config['dataset']['persons']
    logger.info(f"Starting generation for {persons_count} persons")
    
    if args.progress and not args.quiet:
        print(f"🏥 Virtual Ward OMOP Generator")
        print(f"📊 Generating data for {persons_count} persons")
        print(f"💾 Output: {output_path}")
        print()
    
    try:
        # Initialize generators (they create their own ConceptValidator and TemporalCoordinator instances)
        population_gen = PopulationGenerator(config, config_manager.get_concept_dictionary(), 
                                           config_manager.get_random_generator())
        episode_mgr = EpisodeManager(config, config_manager.get_concept_dictionary(),
                                   config_manager.get_random_generator())
        visit_gen = VisitGenerator(config, config_manager.get_concept_dictionary(),
                                 config_manager.get_random_generator())
        signal_gen = SignalGenerator(config, config_manager.get_concept_dictionary(),
                                   config_manager.get_random_generator())
        intervention_engine = InterventionEngine(config, config_manager.get_concept_dictionary(),
                                               config_manager.get_random_generator())

        # Step 0: Generate locations
        if args.progress and not args.quiet:
            print("🏠 Seeding location table")
        
        # Fetch a file of all UK postcodes (cached)
        postcodes_path = Path('./postcodes.cache')
        if postcodes_path.exists():
            logger.info("... using cached postcode file")
        else:
            logger.info("... Downloading postcode file")
            pc_resp = requests.get(
                "https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/d56fb5ac5b2e4b05a0d45a5459ead491/csv?layers=1",
                stream=True)
            pc_resp.raise_for_status()

            with open(postcodes_path, 'wb') as f:
                for data in pc_resp.iter_content(int(2e10)):
                    f.write(data)

            logger.info("Postcode file downloaded")

        all_postcodes = pd.read_csv(
            postcodes_path,
            usecols="PCDS LAT LONG".split()
        )

        locations = all_postcodes[
            all_postcodes.PCDS.str.startswith(tuple(config['dataset']['location']['districts']))
        ]
        del all_postcodes
        locations.columns = 'zip latitude longitude'.split()
        locations['location_id'] = range(1, len(locations) + 1)
        
        # Step 1: Generate population
        if args.progress and not args.quiet:
            print("👥 Generating population demographics...")
        population_data = population_gen.generate(locations)
        logger.info(f"Generated {len(population_data['person'])} persons")

        # Step 1.5: Keep only the locations that are used
        locations = locations[locations.location_id.isin(population_data['person']['location_id'])]

        # Step 2: Generate episodes
        if args.progress and not args.quiet:
            print("📅 Creating virtual ward episodes...")
        episode_data = episode_mgr.generate(
            population_data['person'],
            population_data['observation_period'],
            starting_survey_id=population_data['per_person_obs'].survey_conduct_id.max() + 1,
            starting_observation_id=population_data['per_person_obs'].observation_id.max() + 1
        )
        logger.info(f"Generated {len(episode_data['episode'])} episodes")

        # Step 2.3: Generate mortality events (before visits/signals to prevent post-death activity)
        if args.progress and not args.quiet:
            print(" Generating mortality events...")
        from .generators.mortality import MortalityGenerator
        mortality_gen = MortalityGenerator(config, config_manager.get_concept_dictionary(),
                                          config_manager.get_random_generator())
        death_data = mortality_gen.generate(
            persons_df=population_data['person'],
            episodes_df=episode_data['episode'],
            procedures_df=pd.DataFrame()  # No procedures yet, will use episode severity
        )
        logger.info(f"Generated {len(death_data)} death records")

        # Create death_lookup for preventing post-death activity
        death_lookup = {}
        if not death_data.empty:
            for _, death_row in death_data.iterrows():
                death_lookup[death_row['person_id']] = death_row['death_date']
            logger.info(f"Created death lookup with {len(death_lookup)} entries")

        # Step 2.5: Generate visits
        if args.progress and not args.quiet:
            print("🏥 Generating visit occurrences...")
        visit_data = visit_gen.generate(episode_data['episode'], death_lookup=death_lookup)
        logger.info(f"Generated {len(visit_data)} visits")
        
        # Step 3: Generate signals (PROMs and measurements)
        if args.progress and not args.quiet:
            print("📈 Generating patient signals and measurements...")
        try:
            signal_data = signal_gen.generate(
                episode_data['episode'],
                visits=visit_data,
                device_assignments=episode_data.get('device_exposure', pd.DataFrame()),
                per_person_obs=population_data.get('per_person_obs', pd.DataFrame()),
                starting_survey_id=episode_data['per_episode_obs'].survey_conduct_id.max() + 1,
                starting_observation_id=episode_data['per_episode_obs'].observation_id.max() + 1,
                death_lookup=death_lookup
            )
        except Exception as e:
            logger.error(f"Signal generation failed: {str(e)}")
            logger.debug(f"Episode data keys: {list(episode_data.keys())}")
            logger.debug(f"Episode data shape: {episode_data['episode'].shape}")
            logger.debug(f"Device exposure shape: {episode_data.get('device_exposure', pd.DataFrame()).shape}")
            raise
        logger.info(f"Generated {len(signal_data.get('observation', []))} PROM observations")
        logger.info(f"Generated {len(signal_data.get('measurement', []))} device measurements")
        

        # Step 3.5 - Merge observations from PROMs with Per-Episode and Per-Person Data before saving
        signal_data['observation'] = pd.concat([
            episode_data.get('per_episode_obs', pd.DataFrame()),
            population_data.get('per_person_obs', pd.DataFrame()),
            signal_data['observation']
        ])

        # Step 4: Generate interventions
        if args.progress and not args.quiet:
            print("🏥 Applying clinical intervention logic...")
        intervention_data = intervention_engine.generate(
            episode_data['episode'],
            signal_data.get('observation', pd.DataFrame()),
            signal_data.get('measurement', pd.DataFrame())
        )
        logger.info(f"Generated {len(intervention_data.get('procedure_occurrence', []))} procedures")
        logger.info(f"Generated {len(intervention_data.get('drug_exposure', []))} drug exposures")

        # Combine all data
        all_tables = {}
        all_tables['location'] = locations
        all_tables.update(population_data)
        all_tables.update(episode_data)
        # Ensure updated observation periods from episode generation take precedence
        if 'observation_period' in episode_data:
            all_tables['observation_period'] = episode_data['observation_period']
        all_tables['visit_occurrence'] = visit_data  # Add visit data
        all_tables.update(signal_data)
        all_tables.update(intervention_data)
        all_tables['death'] = death_data  # Add death data

        # Step 5: Validate data
        if not args.skip_validation:
            if args.progress and not args.quiet:
                print("✅ Validating data integrity...")
            validator = DataValidator(config, config_manager.get_concept_dictionary())
            validation_passed = validator.validate(all_tables)
            validation_results = validator.get_validation_report()
            
            if not validation_passed or validator.has_errors:
                logger.error("Data validation failed:")
                for error in validator.validation_errors:
                    logger.error(f"  - {error}")
                raise DataGenerationError("Data validation failed")
            
            if validator.has_warnings:
                logger.warning("Data validation warnings:")
                for warning in validator.validation_warnings:
                    logger.warning(f"  - {warning}")
        else:
            if args.progress and not args.quiet:
                print("⚠️  Skipping data validation...")
            logger.warning("Data validation skipped - database may contain invalid data")
            validation_results = {}
        
        # Step 6: Write to database
        if not args.dry_run:
            if args.progress and not args.quiet:
                print("💾 Writing to DuckDB database...")
            db_writer = DuckDBWriter(config)
            db_writer.write(all_tables, output_path)
            logger.info(f"Successfully wrote data to {output_path}")
        else:
            logger.info("Dry run completed - no data written")
            
        # Report completion
        elapsed_time = time.time() - start_time
        if args.progress and not args.quiet:
            print(f"✨ Generation completed in {elapsed_time:.1f} seconds")
            print(f"📋 Coverage report:")
            for metric, value in validation_results.items():
                if isinstance(value, (int, float)):
                    print(f"   {metric}: {value}")
        
        logger.info(f"Generation completed successfully in {elapsed_time:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        if args.progress and not args.quiet:
            print(f"❌ Generation failed: {str(e)}")
        raise


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging early
    setup_logging_from_args(args)
    logger = get_logger(__name__)
    
    try:
        # Handle special cases first
        if args.show_template:
            show_template()
            return
            
        # Validate arguments
        validate_arguments(args)
        
        logger.info("Virtual Ward OMOP Generator CLI started")
        logger.info(f"Configuration: {args.config}")
        logger.info(f"Output: {args.output}")
        
        # Load and validate configuration
        config_manager = ConfigurationManager()
        config = config_manager.load_yaml_spec(args.config)
        
        # Load concept dictionary
        concept_file = args.concepts or config.get('vocabulary', {}).get('concept_file')
        if concept_file:
            config_manager.load_concept_dictionary(concept_file)
        
        # Set up random generator
        seed = args.seed or config.get('dataset', {}).get('random_seed')
        if seed is not None:
            config_manager.get_random_generator(seed)
            
        if args.validate_only:
            logger.info("Configuration validation completed successfully")
            if args.progress and not args.quiet:
                print("✅ Configuration is valid")
            return
            
        # Run generation pipeline
        run_generation_pipeline(config_manager, config, args.output, args)
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        if not args.quiet:
            print("\n⚠️  Generation interrupted by user")
        sys.exit(1)
    except (ConfigurationError, DataGenerationError, ValueError, FileNotFoundError) as e:
        logger.error(f"Error: {str(e)}")
        if not args.quiet:
            print(f"❌ Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        if not args.quiet:
            print(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()