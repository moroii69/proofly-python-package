from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from .constants import METRIC_OPTIONS, VALUE_RANGES, CONDITION_DESCRIPTIONS

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def validate_metric_value(
    condition: str,
    metric: str,
    value: float,
    timestamp: Optional[datetime] = None
) -> Tuple[bool, List[str]]:
    """
    Validate a single metric value against defined ranges and rules.
    
    Args:
        condition: The health condition being monitored
        metric: The specific metric being validated
        value: The value to validate
        timestamp: Optional timestamp of the measurement
    
    Returns:
        Tuple of (is_valid: bool, warnings: List[str])
    """
    warnings = []
    
    # Check if metric exists for condition
    if metric not in VALUE_RANGES[condition]:
        raise ValidationError(f"Invalid metric '{metric}' for condition '{condition}'")
    
    # Get range for this metric
    value_range = VALUE_RANGES[condition][metric]
    normal_range = next(
        (m["normal_range"] for m in METRIC_OPTIONS[condition] 
         if m["value"] == metric),
        None
    )
    
    # Check if value is within acceptable range
    if not value_range["min"] <= value <= value_range["max"]:
        raise ValidationError(
            f"Value {value} for {metric} is outside acceptable range "
            f"({value_range['min']}-{value_range['max']})"
        )
    
    # Check if value is outside normal range (warning)
    if normal_range and (
        value < normal_range["min"] or value > normal_range["max"]
    ):
        warnings.append(
            f"{metric} value {value} is outside normal range "
            f"({normal_range['min']}-{normal_range['max']})"
        )
    
    # Validate timestamp if provided
    if timestamp:
        if timestamp > datetime.now():
            raise ValidationError("Future timestamps are not allowed")
        if timestamp < datetime.now() - timedelta(days=7):
            warnings.append("Measurement is more than 7 days old")
    
    return len(warnings) == 0, warnings

def validate_metrics(
    condition: str,
    metrics: Dict[str, float],
    timestamps: Optional[Dict[str, datetime]] = None
) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Validate all provided metrics for a given condition.
    
    Args:
        condition: The health condition being monitored
        metrics: Dictionary of metric names and their values
        timestamps: Optional dictionary of metric timestamps
    
    Returns:
        Tuple of (is_valid: bool, warnings: Dict[str, List[str]])
    """
    if condition not in CONDITION_DESCRIPTIONS:
        raise ValidationError(f"Unknown condition: {condition}")
    
    warnings: Dict[str, List[str]] = {}
    all_valid = True
    
    # Check required metrics
    required_metrics = set(m["value"] for m in METRIC_OPTIONS[condition])
    missing_metrics = required_metrics - set(metrics.keys())
    if missing_metrics:
        raise ValidationError(
            f"Missing required metrics for {condition}: {missing_metrics}"
        )
    
    # Validate each metric
    for metric, value in metrics.items():
        timestamp = timestamps.get(metric) if timestamps else None
        try:
            is_valid, metric_warnings = validate_metric_value(
                condition, metric, value, timestamp
            )
            if not is_valid:
                all_valid = False
            if metric_warnings:
                warnings[metric] = metric_warnings
        except ValidationError as e:
            logger.error(f"Validation error for {metric}: {str(e)}")
            raise
    
    return all_valid, warnings

def validate_history(
    history: List[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """
    Validate historical measurements for trending analysis.
    
    Args:
        history: List of historical measurements
    
    Returns:
        Tuple of (is_valid: bool, warnings: List[str])
    """
    warnings = []
    
    if not history:
        return True, warnings
    
    # Check for minimum data points
    if len(history) < 3:
        warnings.append(
            "Limited historical data available. Trending analysis may be limited."
        )
    
    # Check for gaps in data
    timestamps = [
        datetime.fromisoformat(entry["timestamp"])
        for entry in history
    ]
    timestamps.sort()
    
    for i in range(len(timestamps) - 1):
        gap = timestamps[i + 1] - timestamps[i]
        if gap > timedelta(days=2):
            warnings.append(
                f"Gap in historical data detected between "
                f"{timestamps[i].date()} and {timestamps[i + 1].date()}"
            )
    
    return len(warnings) == 0, warnings