# Proofly
Enterprise-grade health metrics analysis and prediction engine for healthcare applications


Proofly is a production-ready Python package that empowers healthcare applications with advanced analytics and predictive capabilities. Built with a focus on accuracy and reliability, it processes vital health metrics to deliver actionable insights for various chronic conditions including diabetes, hypertension, COPD, CKD, and CHF.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Basic Usage](#basic-usage)
  - [Advanced Features](#advanced-features)
  - [Configuration](#configuration)
  - [Error Handling](#error-handling)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

## Features

### Core Capabilities

- **Comprehensive Health Scoring**
  - Evidence-based scoring algorithms
  - Condition-specific metric weighting
  - Customizable scoring models
  - Real-time score calculation

- **Clinical Risk Assessment**
  - Multi-factor risk stratification
  - Predictive risk modeling
  - Early warning system
  - Risk trend analysis

- **Longitudinal Analysis**
  - Time-series health data processing
  - Trend detection and forecasting
  - Statistical significance testing
  - Anomaly detection

- **Smart Clinical Recommendations**
  - Personalized health insights
  - Evidence-based intervention suggestions
  - Lifestyle modification recommendations
  - Medication adherence tracking

### Technical Features

- Robust input validation with comprehensive error handling
- Thread-safe implementation for concurrent processing
- Efficient caching system for improved performance
- Extensive logging and monitoring capabilities
- HIPAA-compliant data handling
- RESTful API integration support
- Exportable reports in multiple formats
- Customizable alert thresholds

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Standard Installation

```bash
pip install proofly
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/moroii69/proofly-python-package.git
cd proofly

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Optional Dependencies

```bash
# Install with all optional features
pip install "proofly[all]"

# Install specific feature sets
pip install "proofly[visualization]"  # For plotting capabilities
pip install "proofly[export]"         # For report export features
pip install "proofly[ml]"             # For advanced ML features
```

## Quick Start

```python
from proofly import HealthAnalyzer
from proofly.models import DiabetesMetrics
from datetime import datetime

# Initialize the analyzer
analyzer = HealthAnalyzer(
    config={
        "logging_level": "INFO",
        "cache_enabled": True,
        "validation_mode": "strict"
    }
)

# Create metrics using the type-safe model
metrics = DiabetesMetrics(
    blood_glucose=120,
    hba1c=6.5,
    blood_pressure=130,
    timestamp=datetime.now()
)

# Perform analysis
result = analyzer.analyze_metrics(
    condition="diabetes",
    metrics=metrics
)

# Access results
print(f"Health Score: {result.health_score}")
print(f"Risk Level: {result.risk_level.name}")
print(f"Confidence: {result.confidence_score}%")
print("\nRecommendations:")
for rec in result.recommendations:
    print(f"- {rec.description}")
```

## Usage Guide

### Basic Usage

#### Supported Health Conditions

Proofly provides specialized analysis for multiple chronic conditions:

1. **Diabetes Management**
```python
from proofly import HealthAnalyzer
from proofly.models import DiabetesMetrics

analyzer = HealthAnalyzer()

# Single point analysis
diabetes_metrics = DiabetesMetrics(
    blood_glucose=120,  # mg/dL
    hba1c=6.5,         # %
    blood_pressure=130  # mmHg
)

result = analyzer.analyze_metrics(
    condition="diabetes",
    metrics=diabetes_metrics
)

# Access comprehensive results
print(f"Health Score: {result.health_score}")
print(f"Risk Level: {result.risk_level.name}")
print(f"Confidence: {result.confidence_score}%")

# Get detailed analysis
analysis = result.get_detailed_analysis()
print(f"Blood Glucose Status: {analysis.blood_glucose.status}")
print(f"HbA1c Trend: {analysis.hba1c.trend}")
print(f"Blood Pressure Category: {analysis.blood_pressure.category}")
```

2. **Hypertension Monitoring**
```python
from proofly.models import HypertensionMetrics

hypertension_metrics = HypertensionMetrics(
    systolic_pressure=130,   # mmHg
    diastolic_pressure=85,   # mmHg
    heart_rate=72           # bpm
)

result = analyzer.analyze_metrics(
    condition="hypertension",
    metrics=hypertension_metrics
)
```

3. **COPD Assessment**
```python
from proofly.models import COPDMetrics

copd_metrics = COPDMetrics(
    oxygen_saturation=95,    # %
    peak_flow=350,          # L/min
    respiratory_rate=18      # breaths/min
)

result = analyzer.analyze_metrics(
    condition="copd",
    metrics=copd_metrics
)
```

### Advanced Features

#### Historical Analysis

```python
from proofly.models import HistoricalData

# Create historical dataset
historical_data = HistoricalData([
    DiabetesMetrics(
        blood_glucose=118,
        hba1c=6.4,
        blood_pressure=128,
        timestamp=datetime(2024, 1, 1)
    ),
    DiabetesMetrics(
        blood_glucose=122,
        hba1c=6.5,
        blood_pressure=130,
        timestamp=datetime(2024, 1, 2)
    )
])

# Analyze with historical context
result = analyzer.analyze_metrics(
    condition="diabetes",
    metrics=diabetes_metrics,
    history=historical_data,
    analysis_options={
        "trend_analysis": True,
        "prediction_window": "30d",
        "confidence_threshold": 0.85
    }
)

# Access trend analysis
trend = result.get_trend_analysis()
print(f"Trend Direction: {trend.direction}")
print(f"Rate of Change: {trend.rate_of_change}")
print(f"Prediction (30 days): {trend.prediction.value}")
```

#### Custom Analysis Configuration

```python
from proofly.config import AnalysisConfig
from proofly.enums import ValidationMode, RiskModel

# Configure analysis parameters
config = AnalysisConfig(
    validation_mode=ValidationMode.STRICT,
    risk_model=RiskModel.CONSERVATIVE,
    confidence_threshold=0.9,
    cache_enabled=True,
    include_raw_data=True
)

# Initialize analyzer with custom configuration
analyzer = HealthAnalyzer(config=config)

# Perform analysis with custom options
result = analyzer.analyze_metrics(
    condition="diabetes",
    metrics=diabetes_metrics,
    options={
        "detailed_analysis": True,
        "include_recommendations": True,
        "prediction_enabled": True
    }
)
```

#### Export and Reporting

```python
from proofly.export import ReportGenerator
from proofly.enums import ReportFormat

# Generate detailed report
report = ReportGenerator.create_report(
    result,
    format=ReportFormat.PDF,
    include_graphs=True,
    include_recommendations=True
)

# Export report
report.save("health_analysis_report.pdf")

# Export raw data
result.export_data("analysis_data.json")
```

### Error Handling

```python
from proofly.exceptions import (
    ValidationError,
    ConfigurationError,
    AnalysisError
)

try:
    result = analyzer.analyze_metrics(
        condition="diabetes",
        metrics=DiabetesMetrics(
            blood_glucose=500,  # Invalid value
            hba1c=6.5,
            blood_pressure=130
        )
    )
except ValidationError as e:
    print(f"Validation Error: {e.message}")
    print(f"Invalid Fields: {e.invalid_fields}")
except ConfigurationError as e:
    print(f"Configuration Error: {e.message}")
except AnalysisError as e:
    print(f"Analysis Error: {e.message}")
    print(f"Error Code: {e.error_code}")
```

### Configuration

#### Metric Ranges and Thresholds

```python
from proofly.config import MetricConfig

# Access default configurations
diabetes_config = MetricConfig.get_condition_config("diabetes")
print("Normal Ranges:")
print(f"Blood Glucose: {diabetes_config.ranges.blood_glucose.normal}")
print(f"HbA1c: {diabetes_config.ranges.hba1c.normal}")
print(f"Blood Pressure: {diabetes_config.ranges.blood_pressure.normal}")

# Customize metric thresholds
custom_config = MetricConfig(
    condition="diabetes",
    ranges={
        "blood_glucose": {
            "normal": (80, 120),
            "warning": (120, 140),
            "critical": (140, 200)
        }
    }
)

analyzer.update_config(custom_config)
```

## API Reference

### Core Classes

- `HealthAnalyzer`: Main analysis engine
- `MetricConfig`: Configuration management
- `AnalysisResult`: Analysis results container
- `ReportGenerator`: Report generation utility

### Models

- `DiabetesMetrics`: Diabetes-specific metrics
- `HypertensionMetrics`: Hypertension-specific metrics
- `COPDMetrics`: COPD-specific metrics
- `CKDMetrics`: CKD-specific metrics
- `CHFMetrics`: CHF-specific metrics

### Utility Classes

- `TrendAnalyzer`: Time-series analysis tools
- `RiskCalculator`: Risk assessment utilities
- `RecommendationEngine`: Recommendation generation
- `DataValidator`: Input validation tools


## Contributing

We welcome contributions! Please follow these steps:

1. Check the [Issues](https://github.com/moroii69/proofly/issues) page for open tasks
2. Fork the repository
3. Create a feature branch (`git checkout -b feature/amazing-feature`)
4. Make your changes with appropriate tests
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to your branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
flake8 proofly
black proofly

# Generate documentation
cd docs
make html
```

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

## Support

- Issue Tracker: [GitHub Issues](https://github.com/moroii69/proofly/issues)
- Email Support: support@proofly.xyz

## Acknowledgments

- Built with input from healthcare professionals
- Implements evidence-based medical guidelines
- Uses validated statistical models
- Follows healthcare industry best practices
- Adheres to HIPAA compliance standards

---

Made with ❤️ by the Proofly Team