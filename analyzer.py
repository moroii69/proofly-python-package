from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import logging
from .validators import validate_metrics, validate_history
from .constants import (
    METRIC_OPTIONS,
    VALUE_RANGES,
    RISK_LEVELS,
    CONDITION_DESCRIPTIONS
)

logger = logging.getLogger(__name__)

class HealthAnalyzer:
    def __init__(self):
        self.last_analysis = None
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the analyzer."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def calculate_health_score(
        self,
        condition: str,
        metrics: Dict[str, float],
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate comprehensive health score based on metrics and history.
        
        Args:
            condition: The health condition being monitored
            metrics: Current metric values
            history: Optional historical measurements
        
        Returns:
            Tuple of (score: float, details: Dict[str, Any])
        """
        score_components = {}
        weights = self._get_metric_weights(condition)
        
        # Calculate base score from current metrics
        for metric, value in metrics.items():
            metric_score = self._calculate_metric_score(
                condition, metric, value
            )
            score_components[metric] = {
                "score": metric_score,
                "weight": weights[metric],
                "weighted_score": metric_score * weights[metric]
            }
        
        # Adjust score based on history if available
        trend_impact = 0
        if history:
            _, trend_warnings = validate_history(history)
            if not trend_warnings:
                trend_impact = self._calculate_trend_impact(
                    condition, metrics, history
                )
        
        # Calculate final weighted score
        base_score = sum(
            comp["weighted_score"] for comp in score_components.values()
        )
        final_score = min(100, max(0, base_score + trend_impact))
        
        return final_score, {
        "components": score_components,
        "trend_impact": trend_impact,
        "final_score": final_score,
        "details": {
            "timestamp": datetime.now().isoformat(),
            "methodology": "Weighted average with trend adjustment"
        }
    }

    def _get_metric_weights(self, condition: str) -> Dict[str, float]:
        """
        Get importance weights for each metric based on condition.
        
        Args:
            condition: The health condition being monitored
        
        Returns:
            Dictionary of metric weights
        """
        key_metrics = set(CONDITION_DESCRIPTIONS[condition]["key_metrics"])
        weights = {}
        
        for metric_option in METRIC_OPTIONS[condition]:
            metric = metric_option["value"]
            # Key metrics get higher weights
            weights[metric] = 0.4 if metric in key_metrics else 0.2
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        return {k: v/total_weight for k, v in weights.items()}

    def _calculate_metric_score(
        self,
        condition: str,
        metric: str,
        value: float
    ) -> float:
        """
        Calculate normalized score for a single metric.
        
        Args:
            condition: The health condition
            metric: The metric name
            value: The metric value
        
        Returns:
            Normalized score between 0 and 100
        """
        metric_info = next(
            m for m in METRIC_OPTIONS[condition]
            if m["value"] == metric
        )
        normal_range = metric_info["normal_range"]
        value_range = VALUE_RANGES[condition][metric]
        
        # Perfect score if within normal range
        if normal_range["min"] <= value <= normal_range["max"]:
            return 100.0
        
        # Calculate how far outside normal range the value is
        if value < normal_range["min"]:
            deviation = (normal_range["min"] - value) / (normal_range["min"] - value_range["min"])
        else:
            deviation = (value - normal_range["max"]) / (value_range["max"] - normal_range["max"])
        
        # Convert to score (100 = perfect, 0 = at or beyond value range limits)
        return max(0, 100 * (1 - deviation))

    def _calculate_trend_impact(
        self,
        condition: str,
        current_metrics: Dict[str, float],
        history: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate score adjustment based on metric trends.
        
        Args:
            condition: The health condition
            current_metrics: Current metric values
            history: Historical measurements
        
        Returns:
            Score adjustment value (-10 to +10)
        """
        trend_impacts = []
        
        for metric, current_value in current_metrics.items():
            # Extract historical values for this metric
            historical_values = [
                entry["metrics"][metric]
                for entry in history
                if metric in entry["metrics"]
            ]
            
            if len(historical_values) < 3:
                continue
                
            # Calculate trend
            x = np.arange(len(historical_values))
            slope, _, r_value, _, _ = stats.linregress(x, historical_values)
            
            # Get target range for this metric
            metric_info = next(
                m for m in METRIC_OPTIONS[condition]
                if m["value"] == metric
            )
            target_range = metric_info["normal_range"]
            target_mid = (target_range["min"] + target_range["max"]) / 2
            
            # Calculate impact based on trend direction and current position
            if current_value < target_mid and slope > 0:
                # Improving from below target
                trend_impacts.append(min(5, slope * 2))
            elif current_value > target_mid and slope < 0:
                # Improving from above target
                trend_impacts.append(min(5, -slope * 2))
            else:
                # Moving away from target
                trend_impacts.append(max(-5, -abs(slope) * 2))
        
        # Return average impact
        return sum(trend_impacts) / len(trend_impacts) if trend_impacts else 0

    def determine_risk_level(self, health_score: float) -> str:
        """
        Determine risk level based on health score.
        
        Args:
            health_score: The calculated health score
            
        Returns:
            Risk level string ("LOW", "MEDIUM", or "HIGH")
        """
        for level, info in RISK_LEVELS.items():
            if info["score_range"][0] <= health_score <= info["score_range"][1]:
                return level
        return "HIGH"  # Default to high risk if no range matches

    def generate_recommendations(
        self,
        condition: str,
        metrics: Dict[str, float],
        risk_level: str,
        score_details: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Generate detailed health recommendations.
        
        Args:
            condition: The health condition
            metrics: Current metric values
            risk_level: Calculated risk level
            score_details: Details from score calculation
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # General recommendation based on risk level
        recommendations.append({
            "type": "general",
            "priority": "high" if risk_level == "HIGH" else "medium",
            "message": RISK_LEVELS[risk_level]["message"]
        })
        
        # Condition-specific recommendations
        for metric, value in metrics.items():
            metric_info = next(
                m for m in METRIC_OPTIONS[condition]
                if m["value"] == metric
            )
            normal_range = metric_info["normal_range"]
            
            if value < normal_range["min"]:
                recommendations.append({
                    "type": "metric_specific",
                    "metric": metric_info["label"],
                    "priority": "high",
                    "message": (
                        f"Your {metric_info['label']} is below the normal range. "
                        f"Consider the following steps:\n"
                        f"1. Monitor more frequently\n"
                        f"2. Consult your healthcare provider\n"
                        f"3. Keep a detailed log of activities"
                    )
                })
            elif value > normal_range["max"]:
                recommendations.append({
                    "type": "metric_specific",
                    "metric": metric_info["label"],
                    "priority": "high",
                    "message": (
                        f"Your {metric_info['label']} is above the normal range. "
                        f"Recommended actions:\n"
                        f"1. Verify measurement accuracy\n"
                        f"2. Schedule a check-up\n"
                        f"3. Review medication schedule if applicable"
                    )
                })
        
        # Trend-based recommendations
        if "trend_impact" in score_details:
            trend = score_details["trend_impact"]
            if trend < -2:
                recommendations.append({
                    "type": "trend",
                    "priority": "high",
                    "message": (
                        "Your health metrics show a concerning trend. "
                        "Please consult your healthcare provider soon."
                    )
                })
            elif trend > 2:
                recommendations.append({
                   "type": "trend",
                    "priority": "low",
                    "message": (
                        "Your health metrics are showing positive improvement. "
                        "Keep up with your current management plan."
                    )
                })
        
        return recommendations

    def analyze_metrics(
        self,
        condition: str,
        metrics: Dict[str, float],
        history: Optional[List[Dict[str, Any]]] = None,
        include_raw_data: bool = False
    ) -> Dict[str, Any]:
        """
        Perform comprehensive health analysis.
        
        Args:
            condition: The health condition being monitored
            metrics: Dictionary of current metric values
            history: Optional list of historical measurements
            include_raw_data: Whether to include raw data in results
            
        Returns:
            Dictionary containing complete analysis results
        """
        logger.info(f"Starting analysis for condition: {condition}")
        
        try:
            # Validate inputs
            is_valid, warnings = validate_metrics(condition, metrics)
            if not is_valid:
                logger.warning(f"Validation warnings: {warnings}")
            
            # Calculate health score
            health_score, score_details = self.calculate_health_score(
                condition, metrics, history
            )
            
            # Determine risk level
            risk_level = self.determine_risk_level(health_score)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(
                condition, metrics, risk_level, score_details
            )
            
            # Compile analysis results
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "condition": {
                    "name": CONDITION_DESCRIPTIONS[condition]["name"],
                    "description": CONDITION_DESCRIPTIONS[condition]["description"]
                },
                "health_score": round(health_score, 1),
                "risk_level": {
                    "level": risk_level,
                    "message": RISK_LEVELS[risk_level]["message"],
                    "color": RISK_LEVELS[risk_level]["color"]
                },
                "recommendations": recommendations,
                "metric_analysis": {
                    metric: {
                        "value": value,
                        "unit": next(
                            m["unit"] for m in METRIC_OPTIONS[condition]
                            if m["value"] == metric
                        ),
                        "status": self._get_metric_status(condition, metric, value),
                        "score": score_details["components"][metric]["score"]
                    }
                    for metric, value in metrics.items()
                },
                "confidence": self._calculate_confidence(
                    metrics, warnings, bool(history)
                )
            }
            
            if include_raw_data:
                analysis["raw_data"] = {
                    "metrics": metrics,
                    "history": history,
                    "score_details": score_details,
                    "validation_warnings": warnings
                }
            
            self.last_analysis = analysis
            logger.info("Analysis completed successfully")
            return analysis
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise

    def _get_metric_status(
        self,
        condition: str,
        metric: str,
        value: float
    ) -> str:
        """
        Determine status category for a metric value.
        
        Args:
            condition: The health condition
            metric: The metric name
            value: The metric value
            
        Returns:
            Status string ("NORMAL", "WARNING", or "CRITICAL")
        """
        metric_info = next(
            m for m in METRIC_OPTIONS[condition]
            if m["value"] == metric
        )
        normal_range = metric_info["normal_range"]
        value_range = VALUE_RANGES[condition][metric]
        
        if normal_range["min"] <= value <= normal_range["max"]:
            return "NORMAL"
        
        # Calculate how far outside normal range
        if value < normal_range["min"]:
            deviation = (normal_range["min"] - value) / (normal_range["min"] - value_range["min"])
        else:
            deviation = (value - normal_range["max"]) / (value_range["max"] - normal_range["max"])
        
        return "CRITICAL" if deviation > 0.5 else "WARNING"

    def _calculate_confidence(
        self,
        metrics: Dict[str, float],
        warnings: Dict[str, List[str]],
        has_history: bool
    ) -> float:
        """
        Calculate confidence score for the analysis.
        
        Args:
            metrics: Current metric values
            warnings: Validation warnings
            has_history: Whether historical data was provided
            
        Returns:
            Confidence percentage
        """
        base_confidence = 90.0  # Start with 90% confidence
        
        # Reduce confidence based on warnings
        warning_count = sum(len(w) for w in warnings.values())
        base_confidence -= warning_count * 5
        
        # Adjust based on data completeness
        if not has_history:
            base_confidence -= 10
        
        # Ensure confidence stays within reasonable bounds
        return max(50.0, min(95.0, base_confidence))
"""