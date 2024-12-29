from typing import Dict, List, Any, TypedDict

class MetricOption(TypedDict):
    value: str
    label: str
    unit: str
    description: str
    normal_range: Dict[str, float]

METRIC_OPTIONS: Dict[str, List[MetricOption]] = {
    "diabetes": [
        {
            "value": "bloodGlucose",
            "label": "Blood Glucose",
            "unit": "mg/dL",
            "description": "Fasting blood glucose level",
            "normal_range": {"min": 70, "max": 100}
        },
        {
            "value": "hba1c",
            "label": "HbA1c",
            "unit": "%",
            "description": "Average blood glucose levels over 2-3 months",
            "normal_range": {"min": 4, "max": 5.7}
        },
        {
            "value": "bloodPressure",
            "label": "Blood Pressure",
            "unit": "mmHg",
            "description": "Systolic blood pressure",
            "normal_range": {"min": 90, "max": 120}
        }
    ],
    "hypertension": [
        {
            "value": "systolic",
            "label": "Systolic Pressure",
            "unit": "mmHg",
            "description": "Upper number of blood pressure reading",
            "normal_range": {"min": 90, "max": 120}
        },
        {
            "value": "diastolic",
            "label": "Diastolic Pressure",
            "unit": "mmHg",
            "description": "Lower number of blood pressure reading",
            "normal_range": {"min": 60, "max": 80}
        },
        {
            "value": "heartRate",
            "label": "Heart Rate",
            "unit": "bpm",
            "description": "Resting heart rate",
            "normal_range": {"min": 60, "max": 100}
        }
    ],
    "copd": [
        {
            "value": "oxygenSaturation",
            "label": "Oxygen Saturation",
            "unit": "%",
            "description": "Blood oxygen level",
            "normal_range": {"min": 95, "max": 100}
        },
        {
            "value": "peakFlow",
            "label": "Peak Flow",
            "unit": "L/min",
            "description": "Maximum speed of exhalation",
            "normal_range": {"min": 400, "max": 600}
        },
        {
            "value": "respiratoryRate",
            "label": "Respiratory Rate",
            "unit": "breaths/min",
            "description": "Breaths per minute at rest",
            "normal_range": {"min": 12, "max": 20}
        }
    ],
    "ckd": [
        {
            "value": "creatinine",
            "label": "Creatinine",
            "unit": "mg/dL",
            "description": "Kidney function marker",
            "normal_range": {"min": 0.7, "max": 1.3}
        },
        {
            "value": "gfr",
            "label": "GFR",
            "unit": "mL/min",
            "description": "Glomerular filtration rate",
            "normal_range": {"min": 90, "max": 120}
        },
        {
            "value": "bloodPressure",
            "label": "Blood Pressure",
            "unit": "mmHg",
            "description": "Systolic blood pressure",
            "normal_range": {"min": 90, "max": 120}
        }
    ],
    "chf": [
        {
            "value": "weight",
            "label": "Weight",
            "unit": "kg",
            "description": "Daily morning weight",
            "normal_range": {"min": 0, "max": 500}  # Varies by individual
        },
        {
            "value": "bloodPressure",
            "label": "Blood Pressure",
            "unit": "mmHg",
            "description": "Systolic blood pressure",
            "normal_range": {"min": 90, "max": 120}
        },
        {
            "value": "heartRate",
            "label": "Heart Rate",
            "unit": "bpm",
            "description": "Resting heart rate",
            "normal_range": {"min": 60, "max": 100}
        }
    ]
}

VALUE_RANGES = {
    "diabetes": {
        "bloodGlucose": {"min": 50, "max": 400},
        "hba1c": {"min": 4, "max": 15},
        "bloodPressure": {"min": 90, "max": 180}
    },
    "hypertension": {
        "systolic": {"min": 90, "max": 180},
        "diastolic": {"min": 60, "max": 120},
        "heartRate": {"min": 40, "max": 200}
    },
    "copd": {
        "oxygenSaturation": {"min": 80, "max": 100},
        "peakFlow": {"min": 50, "max": 800},
        "respiratoryRate": {"min": 10, "max": 40}
    },
    "ckd": {
        "creatinine": {"min": 0.5, "max": 2.0},
        "gfr": {"min": 30, "max": 120},
        "bloodPressure": {"min": 90, "max": 180}
    },
    "chf": {
        "weight": {"min": 30, "max": 300},
        "bloodPressure": {"min": 90, "max": 180},
        "heartRate": {"min": 40, "max": 200}
    }
}

RISK_LEVELS = {
    "LOW": {
        "message": "Your health indicators are within normal ranges. Keep up the good work!",
        "score_range": (80, 100),
        "color": "#28a745"
    },
    "MEDIUM": {
        "message": "Keep an eye on your health. Some metrics need attention.",
        "score_range": (60, 79),
        "color": "#ffc107"
    },
    "HIGH": {
        "message": "Important: Please consult with your healthcare provider.",
        "score_range": (0, 59),
        "color": "#dc3545"
    }
}

CONDITION_DESCRIPTIONS = {
    "diabetes": {
        "name": "Diabetes",
        "description": "A chronic condition affecting how your body processes blood sugar.",
        "key_metrics": ["bloodGlucose", "hba1c"],
        "monitoring_frequency": "Daily for blood glucose, every 3 months for HbA1c"
    },
    "hypertension": {
        "name": "Hypertension",
        "description": "High blood pressure that can lead to heart disease.",
        "key_metrics": ["systolic", "diastolic"],
        "monitoring_frequency": "Daily or as recommended by healthcare provider"
    },
    "copd": {
        "name": "COPD",
        "description": "Chronic obstructive pulmonary disease affecting breathing.",
        "key_metrics": ["oxygenSaturation", "peakFlow"],
        "monitoring_frequency": "Daily or as symptoms indicate"
    },
    "ckd": {
        "name": "Chronic Kidney Disease",
        "description": "Long-term condition affecting kidney function.",
        "key_metrics": ["creatinine", "gfr"],
        "monitoring_frequency": "As recommended by healthcare provider"
    },
    "chf": {
        "name": "Congestive Heart Failure",
        "description": "Condition where the heart can't pump blood effectively.",
        "key_metrics": ["weight", "bloodPressure"],
        "monitoring_frequency": "Daily weight and blood pressure monitoring"
    }
}