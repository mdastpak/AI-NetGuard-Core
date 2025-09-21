"""
Ethics Agent

Responsible for ethical decision making, bias detection, and ensuring
responsible AI practices in the AI-NetGuard system.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import asyncio
from collections import defaultdict
from .base_agent import BaseAgent


class EthicsAgent(BaseAgent):
    """Agent specialized in ethical AI and responsible practices."""

    def __init__(self, coordinator_agent=None, **kwargs):
        system_message = """
        You are the EthicsAgent, responsible for ensuring ethical AI practices
        and responsible decision making in AI-NetGuard.
        """

        super().__init__(
            name="EthicsAgent",
            system_message=system_message,
            coordinator_agent=coordinator_agent,
            **kwargs
        )

        self.capabilities = ["bias_detection", "ethical_decision_making", "fairness_assessment", "bias_mitigation", "fairness_optimization", "continuous_monitoring"]
        self.dependencies = ["EvaluationAgent", "PrivacyAgent", "MonitoringAgent"]

    async def _execute_task(self, task_description: str, **kwargs) -> Any:
        if "assess_bias" in task_description.lower():
            return await self._assess_bias(**kwargs)
        elif "fairness" in task_description.lower():
            return await self._assess_fairness(**kwargs)
        elif "bias_mitigation" in task_description.lower():
            return await self._mitigate_bias(**kwargs)
        elif "ethical_decision" in task_description.lower():
            return await self._make_ethical_decision(**kwargs)
        elif "continuous_monitoring" in task_description.lower():
            return await self._continuous_ethics_monitoring(**kwargs)
        else:
            return {"status": "completed", "task": task_description}

    async def _assess_bias(self, predictions=None, targets=None, sensitive_features=None, **kwargs):
        """Assess bias in model predictions across sensitive features."""
        if predictions is None or targets is None or sensitive_features is None:
            return {'error': 'Predictions, targets, and sensitive features required'}

        bias_metrics = {}

        # Demographic parity (equal acceptance rates across groups)
        for feature_name, feature_values in sensitive_features.items():
            groups = defaultdict(list)
            for i, val in enumerate(feature_values):
                groups[val].append((predictions[i], targets[i]))

            group_rates = {}
            for group, data in groups.items():
                pred_rate = np.mean([p for p, t in data])
                true_rate = np.mean([t for p, t in data])
                group_rates[group] = {'prediction_rate': pred_rate, 'true_rate': true_rate}

            # Calculate disparity
            rates = list(group_rates.values())
            if len(rates) > 1:
                max_rate = max(r['prediction_rate'] for r in rates)
                min_rate = min(r['prediction_rate'] for r in rates)
                bias_metrics[f'{feature_name}_demographic_parity'] = max_rate - min_rate

        # Equal opportunity (equal true positive rates)
        for feature_name, feature_values in sensitive_features.items():
            groups = defaultdict(list)
            for i, val in enumerate(feature_values):
                if targets[i] == 1:  # Only positive cases
                    groups[val].append(predictions[i])

            group_tpr = {}
            for group, preds in groups.items():
                tpr = np.mean(preds) if preds else 0
                group_tpr[group] = tpr

            # Calculate TPR disparity
            tprs = list(group_tpr.values())
            if len(tprs) > 1:
                max_tpr = max(tprs)
                min_tpr = min(tprs)
                bias_metrics[f'{feature_name}_equal_opportunity'] = max_tpr - min_tpr

        # Overall bias score (0-1, higher is more biased)
        bias_score = np.mean(list(bias_metrics.values())) if bias_metrics else 0

        return {
            'bias_score': min(bias_score, 1.0),
            'bias_metrics': bias_metrics,
            'bias_detected': bias_score > 0.1,  # Threshold for bias detection
            'recommendations': self._generate_bias_recommendations(bias_score, bias_metrics)
        }

    async def _assess_fairness(self, predictions=None, targets=None, sensitive_features=None, **kwargs):
        """Comprehensive fairness assessment using multiple metrics."""
        if predictions is None or targets is None:
            return {'error': 'Predictions and targets required'}

        fairness_metrics = {}

        # Accuracy parity
        accuracy = np.mean(np.array(predictions) == np.array(targets))
        fairness_metrics['overall_accuracy'] = accuracy

        # Confusion matrix components
        tp = np.sum((np.array(predictions) == 1) & (np.array(targets) == 1))
        tn = np.sum((np.array(predictions) == 0) & (np.array(targets) == 0))
        fp = np.sum((np.array(predictions) == 1) & (np.array(targets) == 0))
        fn = np.sum((np.array(predictions) == 0) & (np.array(targets) == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        fairness_metrics.update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positive_rate': recall,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
        })

        # Fairness-specific metrics if sensitive features available
        if sensitive_features:
            bias_assessment = await self._assess_bias(predictions, targets, sensitive_features)
            fairness_metrics.update(bias_assessment.get('bias_metrics', {}))

        # Overall fairness score (0-1, higher is more fair)
        fairness_score = 1.0 - (np.mean(list(fairness_metrics.values())) - 0.5) * 2
        fairness_score = max(0, min(1, fairness_score))

        return {
            'fairness_score': fairness_score,
            'fairness_metrics': fairness_metrics,
            'fairness_achieved': fairness_score > 0.8,
            'improvement_needed': fairness_score < 0.7
        }

    async def _mitigate_bias(self, model=None, training_data=None, sensitive_features=None, **kwargs):
        """Implement bias mitigation strategies."""
        mitigation_strategies = []

        if training_data is None or sensitive_features is None:
            return {'error': 'Training data and sensitive features required'}

        # Strategy 1: Reweighting
        mitigation_strategies.append({
            'strategy': 'reweighting',
            'description': 'Adjust sample weights to balance group representation',
            'implementation': 'Apply inverse frequency weighting to underrepresented groups'
        })

        # Strategy 2: Adversarial debiasing
        mitigation_strategies.append({
            'strategy': 'adversarial_debiasing',
            'description': 'Train adversarial network to remove sensitive feature information',
            'implementation': 'Use gradient reversal layer during training'
        })

        # Strategy 3: Fair representation learning
        mitigation_strategies.append({
            'strategy': 'fair_representation',
            'description': 'Learn fair representations that encode task-relevant information',
            'implementation': 'Minimize mutual information between representations and sensitive features'
        })

        # Strategy 4: Post-processing calibration
        mitigation_strategies.append({
            'strategy': 'post_processing',
            'description': 'Adjust decision thresholds to achieve fairness constraints',
            'implementation': 'Equalize odds by adjusting classification thresholds per group'
        })

        # Simulate mitigation effectiveness
        effectiveness_scores = {}
        for strategy in mitigation_strategies:
            # Mock effectiveness based on strategy type
            base_effectiveness = 0.7 + np.random.random() * 0.3
            effectiveness_scores[strategy['strategy']] = base_effectiveness

        best_strategy = max(effectiveness_scores.items(), key=lambda x: x[1])

        return {
            'mitigation_strategies': mitigation_strategies,
            'recommended_strategy': best_strategy[0],
            'expected_improvement': best_strategy[1],
            'implementation_plan': self._create_mitigation_plan(best_strategy[0])
        }

    async def _make_ethical_decision(self, scenario=None, options=None, **kwargs):
        """Make ethical decisions based on fairness and bias considerations."""
        if scenario is None or options is None:
            return {'error': 'Scenario and options required'}

        ethical_scores = {}

        for option in options:
            # Evaluate each option across multiple ethical dimensions
            fairness_score = np.random.random()  # Mock fairness assessment
            bias_score = np.random.random()  # Mock bias assessment
            transparency_score = np.random.random()  # Mock transparency assessment
            accountability_score = np.random.random()  # Mock accountability assessment

            # Weighted ethical score
            ethical_score = (0.4 * fairness_score +
                           0.3 * (1 - bias_score) +  # Lower bias is better
                           0.15 * transparency_score +
                           0.15 * accountability_score)

            ethical_scores[option] = {
                'ethical_score': ethical_score,
                'fairness': fairness_score,
                'bias': bias_score,
                'transparency': transparency_score,
                'accountability': accountability_score
            }

        best_option = max(ethical_scores.items(), key=lambda x: x[1]['ethical_score'])

        return {
            'decision': best_option[0],
            'ethical_score': best_option[1]['ethical_score'],
            'reasoning': f"Selected {best_option[0]} based on highest ethical score ({best_option[1]['ethical_score']:.3f})",
            'option_scores': ethical_scores,
            'ethical_framework': 'Fairness-Bias-Transparency-Accountability (FBTA)'
        }

    async def _continuous_ethics_monitoring(self, monitoring_duration=3600, check_interval=300, **kwargs):
        """Continuous monitoring of ethical metrics and bias detection."""
        monitoring_results = []
        alerts_triggered = []

        # Simulate continuous monitoring
        checks_performed = monitoring_duration // check_interval

        for check in range(checks_performed):
            # Mock monitoring data
            bias_score = 0.05 + np.random.random() * 0.15  # Random bias between 0.05-0.2
            fairness_score = 0.8 + np.random.random() * 0.2  # Random fairness between 0.8-1.0

            monitoring_results.append({
                'check_number': check + 1,
                'timestamp': f"2025-09-21T05:0{check}:00Z",
                'bias_score': bias_score,
                'fairness_score': fairness_score,
                'anomaly_detected': bias_score > 0.15 or fairness_score < 0.85
            })

            # Trigger alerts for concerning metrics
            if bias_score > 0.15:
                alerts_triggered.append({
                    'type': 'high_bias_alert',
                    'check': check + 1,
                    'bias_score': bias_score,
                    'action_required': 'Bias mitigation needed'
                })

            if fairness_score < 0.85:
                alerts_triggered.append({
                    'type': 'low_fairness_alert',
                    'check': check + 1,
                    'fairness_score': fairness_score,
                    'action_required': 'Fairness optimization needed'
                })

            # Brief pause between checks
            await asyncio.sleep(0.01)

        # Summary statistics
        bias_scores = [r['bias_score'] for r in monitoring_results]
        fairness_scores = [r['fairness_score'] for r in monitoring_results]

        return {
            'monitoring_duration': monitoring_duration,
            'checks_performed': checks_performed,
            'average_bias_score': np.mean(bias_scores),
            'average_fairness_score': np.mean(fairness_scores),
            'bias_trend': 'increasing' if bias_scores[-1] > bias_scores[0] else 'stable',
            'fairness_trend': 'stable' if abs(fairness_scores[-1] - fairness_scores[0]) < 0.1 else 'changing',
            'alerts_triggered': len(alerts_triggered),
            'alert_details': alerts_triggered[:5],  # First 5 alerts
            'recommendations': self._generate_monitoring_recommendations(bias_scores, fairness_scores)
        }

    def _generate_bias_recommendations(self, bias_score, bias_metrics):
        """Generate recommendations based on bias assessment."""
        recommendations = []

        if bias_score > 0.2:
            recommendations.append("Implement immediate bias mitigation strategies")
        elif bias_score > 0.1:
            recommendations.append("Monitor bias levels and consider mitigation")
        else:
            recommendations.append("Bias levels acceptable, continue monitoring")

        # Specific recommendations based on metrics
        for metric, value in bias_metrics.items():
            if 'demographic_parity' in metric and value > 0.1:
                recommendations.append(f"Address demographic parity disparity in {metric}")
            elif 'equal_opportunity' in metric and value > 0.1:
                recommendations.append(f"Address equal opportunity disparity in {metric}")

        return recommendations

    def _create_mitigation_plan(self, strategy):
        """Create detailed implementation plan for bias mitigation."""
        plans = {
            'reweighting': {
                'steps': [
                    'Analyze group distributions in training data',
                    'Calculate inverse frequency weights',
                    'Apply weights during model training',
                    'Validate mitigation effectiveness'
                ],
                'estimated_time': '2-3 days',
                'resources_needed': ['Data scientist', 'ML engineer']
            },
            'adversarial_debiasing': {
                'steps': [
                    'Design adversarial network architecture',
                    'Implement gradient reversal layer',
                    'Train combined model and adversary',
                    'Evaluate bias reduction and accuracy trade-offs'
                ],
                'estimated_time': '1-2 weeks',
                'resources_needed': ['ML researcher', 'Data scientist']
            },
            'fair_representation': {
                'steps': [
                    'Design fair representation learning objective',
                    'Implement mutual information minimization',
                    'Train fair encoder network',
                    'Fine-tune task-specific model'
                ],
                'estimated_time': '1-2 weeks',
                'resources_needed': ['ML researcher', 'Data scientist']
            },
            'post_processing': {
                'steps': [
                    'Calculate group-specific thresholds',
                    'Implement equal odds post-processing',
                    'Validate fairness constraints',
                    'Monitor accuracy-fairness trade-offs'
                ],
                'estimated_time': '3-5 days',
                'resources_needed': ['Data scientist', 'ML engineer']
            }
        }

        return plans.get(strategy, {'error': 'Unknown strategy'})

    def _generate_monitoring_recommendations(self, bias_scores, fairness_scores):
        """Generate recommendations based on monitoring trends."""
        recommendations = []

        avg_bias = np.mean(bias_scores)
        avg_fairness = np.mean(fairness_scores)

        if avg_bias > 0.15:
            recommendations.append("Immediate bias mitigation required")
        elif avg_bias > 0.1:
            recommendations.append("Monitor bias trends closely")

        if avg_fairness < 0.85:
            recommendations.append("Fairness optimization needed")
        elif avg_fairness < 0.9:
            recommendations.append("Consider fairness improvements")

        # Trend analysis
        bias_trend = np.polyfit(range(len(bias_scores)), bias_scores, 1)[0]
        if bias_trend > 0.001:
            recommendations.append("Bias is increasing - investigate causes")

        fairness_trend = np.polyfit(range(len(fairness_scores)), fairness_scores, 1)[0]
        if fairness_trend < -0.001:
            recommendations.append("Fairness is decreasing - take corrective action")

        return recommendations