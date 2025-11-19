# Virtual Ward OMOP Generator

A comprehensive synthetic data generator for creating realistic OMOP CDM (Common Data Model Version 5.4) datasets specifically designed for virtual ward patient monitoring scenarios. Generate learnable synthetic data to develop and test machine learning models for predicting clinical interventions without using real patient data.

## 🏥 Overview

The Virtual Ward OMOP Generator creates synthetic datasets that simulate real-world virtual ward operations, including:

- **Patient Demographics**: Realistic population with age, gender, race, and ethnicity distributions
- **Virtual Ward Episodes**: Episodes with configurable lengths and trajectory patterns
- **Patient-Reported Outcomes (PROMs)**: Daily symptom surveys with realistic completion patterns
- **Device Measurements**: Continuous monitoring data from condition-specific devices
- **Clinical Interventions**: Evidence-based intervention logic linking symptoms to clinical actions
- **OMOP CDM Compliance**: Full compatibility with standard OMOP analysis tools

## ✨ Key Features

- 🎯 **Learnable Patterns**: Generated data achieves ROC-AUC ≥ 0.7 for intervention prediction
- 🔧 **Highly Configurable**: YAML-based configuration for all generation parameters
- 📊 **Multiple Conditions**: Support for COPD, Heart Failure, and Post-operative scenarios
- 🏥 **Realistic Clinical Logic**: Evidence-based intervention triggers and escalation patterns
- 📈 **Temporal Patterns**: Subject-specific baselines, drift, and realistic noise
- 🎲 **Reproducible**: Deterministic generation with configurable random seeds
- ⚡ **Performance Optimized**: Generate 1,000+ patients in under 10 minutes
- 🗄️ **DuckDB Output**: Analysis-ready database with custom views

## 🚀 Quick Start

### Installation

```bash
# Install uv if you haven't already
pip install uv

# Install the project and dependencies
uv pip install -e .

# For development dependencies
uv pip install -e ".[dev]"
```

> **Note**: After installation, you can use either `python -m virtual_ward_omop_generator.cli` or the shorter `virtual-ward-omop-generator` command. The module approach works in all environments, including development setups.

### Generate Your First Dataset

```bash
# Copy a working template configuration
cp examples/basic_config.yaml my_config.yaml

# Edit the concept file path in my_config.yaml
# vocabulary:
#   concept_file: "./path/to/your/omop_virtual_ward_concepts.csv"

# Generate dataset (start with small population for testing)
python -m virtual_ward_omop_generator.cli --config my_config.yaml --output my_dataset.db --persons 500

# Explore the results
duckdb my_dataset.db -c "SHOW TABLES"
```

### Command Line Options

```bash
# Basic usage
python -m virtual_ward_omop_generator.cli --config config.yaml --output dataset.db

# With custom parameters
python -m virtual_ward_omop_generator.cli \
  --config config.yaml \
  --output dataset.db \
  --persons 1500 \
  --seed 42 \
  --verbose

# Validate configuration only
python -m virtual_ward_omop_generator.cli --config config.yaml --validate-only

# Show configuration template
python -m virtual_ward_omop_generator.cli --show-template

# Dry run (validate without writing output)
python -m virtual_ward_omop_generator.cli --config config.yaml --output dataset.db --dry-run
```

## 📋 Requirements

- Python 3.8+
- Dependencies: pandas, numpy, pyyaml, duckdb, scipy
- OMOP concept dictionary CSV file

## 🏗️ Architecture

The generator follows a multi-stage pipeline:

```
Configuration → Population → Episodes → Signals → Interventions → Validation → Output
```

1. **Population Generation**: Demographics and observation periods
2. **Episode Management**: Virtual ward episodes with trajectory assignment
3. **Signal Generation**: PROMs and device measurements with temporal patterns
4. **Intervention Engine**: Clinical logic for triggering interventions
5. **Data Validation**: Integrity checks and quality assurance
6. **Database Output**: DuckDB with analysis-ready views

## 📊 Generated Data

### Main OMOP Tables
- `PERSON`: Patient demographics
- `OBSERVATION_PERIOD`: Continuous observation periods
- `EPISODE`: Virtual ward episodes with trajectory metadata
- `SURVEY_CONDUCT`: Daily survey administration records
- `OBSERVATION`: Patient-reported outcome measures (PROMs)
- `MEASUREMENT`: Device-generated measurements
- `CONDITION_OCCURRENCE`: Index conditions (COPD, HF, Post-op)
- `DRUG_EXPOSURE`: Medication prescriptions
- `PROCEDURE_OCCURRENCE`: Clinical procedures and interventions
- `DEVICE_EXPOSURE`: Monitoring device assignments
- `VISIT_OCCURRENCE`: Telehealth and home visits
- `DEATH`: Death events (rare, <0.5%)

### Analysis Views
- `vw_prom_timeseries`: Patient-reported data over time
- `vw_device_timeseries`: Device measurement timeseries
- `vw_interventions`: All interventions across domains
- `vw_episode_summary`: Episode-level statistics

## 🎛️ Configuration

The generator uses YAML configuration files with extensive customization options:

### Example Configuration Structure
```yaml
schema_version: 0.1

vocabulary:
  id: "VH_VOCAB"
  concept_file: "./omop_virtual_ward_concepts.csv"

dataset:
  name: "virtual_ward_synth"
  random_seed: 42
  persons: 1200
  
  condition_mix:
    2100000300: 0.45   # COPD
    2100000301: 0.45   # Heart Failure
    2100000302: 0.10   # Post-operative

  episode_archetype_mix:
    stable: 0.40
    flare_mild: 0.25
    flare_moderate: 0.20
    flare_severe: 0.10
    noisy_reporter: 0.05

# ... additional configuration sections
```

### Configuration Templates

Choose from pre-built templates in the `examples/` directory:

- **`basic_config.yaml`**: Minimal setup for getting started

## 🔬 Research Applications

### Machine Learning Model Development
- Intervention prediction models
- Risk stratification algorithms
- Anomaly detection systems
- Time series forecasting

### Clinical Research
- Virtual ward workflow optimization
- Device compliance studies
- PROM validation research
- Care pathway analysis

### Healthcare Informatics
- OMOP CDM tool testing
- Data pipeline validation
- Quality metric development
- Interoperability testing

## 🎯 Data Quality Features

### Realistic Patterns
- **Temporal Consistency**: All events within episode boundaries
- **Clinical Plausibility**: Evidence-based parameter ranges
- **Causal Relationships**: Symptoms trigger appropriate interventions (but recovery is not currently modelled)
- **Missing Data**: MNAR patterns with higher missingness on symptomatic days

### Validation System
- **Foreign Key Integrity**: All relationships validated
- **Physiological Bounds**: Measurements within clinical ranges
- **Temporal Logic**: Proper event sequencing
- **Coverage Metrics**: Automated quality reporting

### Learnability Guarantee
- Generated datasets achieve ROC-AUC ≥ 0.7 for urgent review prediction
- Configurable signal-to-noise ratios
- Realistic intervention adherence rates (80-90%)

## 🛠️ Development

### Project Structure
```
virtual_ward_omop_generator/
├── cli.py                 # Command-line interface
├── config/               # Configuration management
├── generators/           # Data generation components
├── models/              # Data models and schemas
├── validation/          # Data validation system
├── output/              # Database output handling
└── examples/            # Configuration templates
```

### Development Setup

```bash
# Format code
black .

# Type checking
mypy virtual_ward_omop_generator

# Run tests
pytest

# Run with coverage
pytest --cov=virtual_ward_omop_generator
```

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/test_validation_system.py

# Run with verbose output
python -m pytest -v
```

## 📈 Performance

- **Generation Speed**: 1,200 persons in <3 minutes on standard laptop
- **Memory Efficiency**: Chunked processing for large datasets
- **Scalability**: Supports 500-2,000 person populations
- **Database Size**: ~50MB for 1,200 persons with full feature set

## 🎯 Example Workflows

### Basic Dataset Generation
```bash
# Start with a simple configuration
cp examples/basic_config.yaml my_config.yaml
python -m virtual_ward_omop_generator.cli --config my_config.yaml --output basic_dataset.db
```

### Configuration Validation
```bash
# Test configuration without generating data
python -m virtual_ward_omop_generator.cli --config my_config.yaml --validate-only
```

## 🔍 Data Analysis Examples

Once you have generated a dataset, you can analyze it using SQL:

```sql
-- Episode summary statistics
SELECT 
  COUNT(*) as total_episodes,
  AVG(julianday(episode_end_datetime) - julianday(episode_start_datetime)) as avg_length_days,
  MIN(episode_start_datetime) as earliest_episode,
  MAX(episode_end_datetime) as latest_episode
FROM episode;

-- Survey completion rates by day of week
SELECT 
  strftime('%w', survey_start_datetime) as day_of_week,
  COUNT(*) as survey_count,
  COUNT(DISTINCT person_id) as unique_persons
FROM survey_conduct 
GROUP BY strftime('%w', survey_start_datetime)
ORDER BY day_of_week;

-- Intervention frequency by type
SELECT 
  procedure_concept_id,
  COUNT(*) as intervention_count,
  COUNT(DISTINCT person_id) as unique_persons,
  COUNT(DISTINCT episode_id) as unique_episodes
FROM procedure_occurrence 
GROUP BY procedure_concept_id
ORDER BY intervention_count DESC;

-- PROM score distributions
SELECT 
  observation_concept_id,
  AVG(value_as_number) as mean_score,
  MIN(value_as_number) as min_score,
  MAX(value_as_number) as max_score,
  COUNT(*) as total_responses
FROM observation 
WHERE value_as_number IS NOT NULL
GROUP BY observation_concept_id;
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Code Style**: Use Black formatting and type hints
2. **Testing**: Add tests for new features
3. **Documentation**: Update documentation for changes
4. **Pull Requests**: Use descriptive commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OMOP Collaborative for the Common Data Model
- Clinical domain experts for validation
- Open source community for foundational tools

## 📞 Support

- **Documentation**: Check the `examples/` directory
- **Issues**: Report bugs via GitHub Issues
- **CLI Help**: Run `python -m virtual_ward_omop_generator.cli --help`

## 🚨 Troubleshooting

### Common Issues

**"Configuration file not found"**
```bash
# Check file path and use absolute path if needed
python -m virtual_ward_omop_generator.cli --config /full/path/to/config.yaml --output dataset.db
```

**"Concept dictionary not found"**
```bash
# Verify the concept_file path in your configuration
# Make sure the CSV file exists and is readable
ls -la /path/to/concepts.csv
```

**"persons must be between 500 and 2000"**
```bash
# Adjust population size in config or command line
python -m virtual_ward_omop_generator.cli --config config.yaml --output dataset.db --persons 1000
```

**Memory issues with large populations**
```bash
# Use smaller population sizes or enable chunked processing
# Monitor memory usage during generation
```

### Getting Help

1. **Enable verbose logging**: Use `--verbose` flag for detailed output
2. **Validate configuration**: Use `--validate-only` to check config
3. **Test with dry run**: Use `--dry-run` to test without writing output
4. **Check examples**: Review template configurations in `examples/`


## Use of ONS Postcode Products

Locations/Postcodes are sourced from the ONS Open Geography Portal and are licenced under the Open Government Licence (OGL).  This data is not present in the repository but will be downloaded when required.

  * Contains OS data © Crown copyright and database right 2025
  * Contains Royal Mail data © Royal Mail copyright and database right 2025
  * Source: Office for National Statistics licensed under the Open Government Licence v.3.0
