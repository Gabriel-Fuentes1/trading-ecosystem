"""
Data Drift Detection for Trading Models
Monitors feature and target drift to trigger model retraining.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import json

# Drift detection libraries
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_suite import MetricSuite
from evidently.metrics import DataDriftMetric, DataQualityMetric, ColumnDriftMetric
from alibi_detect import KSDrift, MMDDrift, ChiSquareDrift
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DriftDetector:
    """
    Advanced drift detection system for trading models.
    Detects both feature drift and concept drift.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.drift_threshold = self.config.get('drift_threshold', 0.05)
        self.min_samples = self.config.get('min_samples', 100)
        self.statistical_tests = self.config.get('statistical_tests', ['ks', 'chi2'])
        
        # Initialize drift detectors
        self.detectors = {}
        
        logging.info("Drift detector initialized")
    
    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: List[str],
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect drift between reference and current data.
        
        :param reference_data: Reference (training) data
        :param current_data: Current (production) data
        :param features: List of feature columns to monitor
        :param target_column: Target column name (optional)
        :return: Drift detection results
        """
        logging.info("Starting drift detection analysis...")
        
        try:
            # Validate inputs
            if reference_data.empty or current_data.empty:
                return {'error': 'Empty data provided'}
            
            if len(current_data) < self.min_samples:
                return {'error': f'Insufficient samples in current data: {len(current_data)} < {self.min_samples}'}
            
            # Prepare data
            ref_features = reference_data[features].fillna(0)
            cur_features = current_data[features].fillna(0)
            
            # Feature drift detection
            feature_drift_results = self._detect_feature_drift(ref_features, cur_features, features)
            
            # Target drift detection (if target column provided)
            target_drift_results = {}
            if target_column and target_column in reference_data.columns and target_column in current_data.columns:
                target_drift_results = self._detect_target_drift(
                    reference_data[target_column],
                    current_data[target_column]
                )
            
            # Evidently report
            evidently_results = self._generate_evidently_report(reference_data, current_data, features)
            
            # Aggregate results
            drift_results = self._aggregate_drift_results(
                feature_drift_results,
                target_drift_results,
                evidently_results
            )
            
            logging.info(f"Drift detection completed. Drift detected: {drift_results.get('drift_detected', False)}")
            return drift_results
            
        except Exception as e:
            logging.error(f"Error in drift detection: {e}")
            return {'error': str(e)}
    
    def _detect_feature_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: List[str]
    ) -> Dict[str, Any]:
        """Detect drift in individual features."""
        feature_drift_results = {
            'drifted_features': [],
            'feature_drift_scores': {},
            'feature_p_values': {},
            'drift_methods': {}
        }
        
        for feature in features:
            try:
                if feature not in reference_data.columns or feature not in current_data.columns:
                    continue
                
                ref_values = reference_data[feature].dropna().values
                cur_values = current_data[feature].dropna().values
                
                if len(ref_values) == 0 or len(cur_values) == 0:
                    continue
                
                # Determine if feature is numerical or categorical
                is_numerical = pd.api.types.is_numeric_dtype(reference_data[feature])
                
                if is_numerical:
                    drift_result = self._detect_numerical_drift(ref_values, cur_values, feature)
                else:
                    drift_result = self._detect_categorical_drift(ref_values, cur_values, feature)
                
                # Store results
                feature_drift_results['feature_drift_scores'][feature] = drift_result['drift_score']
                feature_drift_results['feature_p_values'][feature] = drift_result['p_value']
                feature_drift_results['drift_methods'][feature] = drift_result['method']
                
                if drift_result['drift_detected']:
                    feature_drift_results['drifted_features'].append(feature)
                
            except Exception as e:
                logging.error(f"Error detecting drift for feature {feature}: {e}")
                continue
        
        return feature_drift_results
    
    def _detect_numerical_drift(self, ref_values: np.ndarray, cur_values: np.ndarray, feature: str) -> Dict[str, Any]:
        """Detect drift in numerical features using statistical tests."""
        drift_results = []
        
        # Kolmogorov-Smirnov test
        if 'ks' in self.statistical_tests:
            try:
                ks_stat, ks_p_value = stats.ks_2samp(ref_values, cur_values)
                drift_results.append({
                    'method': 'ks_test',
                    'statistic': ks_stat,
                    'p_value': ks_p_value,
                    'drift_detected': ks_p_value < self.drift_threshold
                })
            except Exception as e:
                logging.error(f"KS test failed for {feature}: {e}")
        
        # Mann-Whitney U test
        if 'mannwhitney' in self.statistical_tests:
            try:
                mw_stat, mw_p_value = stats.mannwhitneyu(ref_values, cur_values, alternative='two-sided')
                drift_results.append({
                    'method': 'mannwhitney_test',
                    'statistic': mw_stat,
                    'p_value': mw_p_value,
                    'drift_detected': mw_p_value < self.drift_threshold
                })
            except Exception as e:
                logging.error(f"Mann-Whitney test failed for {feature}: {e}")
        
        # Population Stability Index (PSI)
        try:
            psi_score = self._calculate_psi(ref_values, cur_values)
            drift_results.append({
                'method': 'psi',
                'statistic': psi_score,
                'p_value': None,
                'drift_detected': psi_score > 0.2  # PSI > 0.2 indicates significant drift
            })
        except Exception as e:
            logging.error(f"PSI calculation failed for {feature}: {e}")
        
        # Return the most significant result
        if drift_results:
            # Prioritize by drift detection and then by p-value
            drift_results.sort(key=lambda x: (not x['drift_detected'], x['p_value'] if x['p_value'] is not None else 1))
            best_result = drift_results[0]
            
            return {
                'drift_detected': best_result['drift_detected'],
                'drift_score': best_result['statistic'],
                'p_value': best_result['p_value'],
                'method': best_result['method']
            }
        
        return {
            'drift_detected': False,
            'drift_score': 0.0,
            'p_value': 1.0,
            'method': 'none'
        }
    
    def _detect_categorical_drift(self, ref_values: np.ndarray, cur_values: np.ndarray, feature: str) -> Dict[str, Any]:
        """Detect drift in categorical features using Chi-square test."""
        try:
            # Get unique categories
            all_categories = np.unique(np.concatenate([ref_values, cur_values]))
            
            # Create frequency tables
            ref_counts = pd.Series(ref_values).value_counts().reindex(all_categories, fill_value=0)
            cur_counts = pd.Series(cur_values).value_counts().reindex(all_categories, fill_value=0)
            
            # Chi-square test
            chi2_stat, chi2_p_value = stats.chisquare(cur_counts, ref_counts)
            
            return {
                'drift_detected': chi2_p_value < self.drift_threshold,
                'drift_score': chi2_stat,
                'p_value': chi2_p_value,
                'method': 'chi2_test'
            }
            
        except Exception as e:
            logging.error(f"Chi-square test failed for {feature}: {e}")
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'p_value': 1.0,
                'method': 'chi2_test'
            }
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)."""
        try:
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference, bins=bins)
            
            # Ensure bins cover the full range of both datasets
            min_val = min(reference.min(), current.min())
            max_val = max(reference.max(), current.max())
            bin_edges[0] = min_val - 1e-6
            bin_edges[-1] = max_val + 1e-6
            
            # Calculate frequencies
            ref_freq, _ = np.histogram(reference, bins=bin_edges)
            cur_freq, _ = np.histogram(current, bins=bin_edges)
            
            # Convert to percentages
            ref_pct = ref_freq / len(reference)
            cur_pct = cur_freq / len(current)
            
            # Avoid division by zero
            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)
            
            # Calculate PSI
            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            
            return psi
            
        except Exception as e:
            logging.error(f"PSI calculation error: {e}")
            return 0.0
    
    def _detect_target_drift(self, ref_target: pd.Series, cur_target: pd.Series) -> Dict[str, Any]:
        """Detect concept drift in target variable."""
        try:
            ref_values = ref_target.dropna().values
            cur_values = cur_target.dropna().values
            
            if len(ref_values) == 0 or len(cur_values) == 0:
                return {'target_drift_detected': False, 'target_drift_score': 0.0}
            
            # Use appropriate test based on target type
            if pd.api.types.is_numeric_dtype(ref_target):
                # Numerical target - use KS test
                ks_stat, ks_p_value = stats.ks_2samp(ref_values, cur_values)
                return {
                    'target_drift_detected': ks_p_value < self.drift_threshold,
                    'target_drift_score': ks_stat,
                    'target_p_value': ks_p_value,
                    'target_drift_method': 'ks_test'
                }
            else:
                # Categorical target - use Chi-square test
                all_categories = np.unique(np.concatenate([ref_values, cur_values]))
                ref_counts = pd.Series(ref_values).value_counts().reindex(all_categories, fill_value=0)
                cur_counts = pd.Series(cur_values).value_counts().reindex(all_categories, fill_value=0)
                
                chi2_stat, chi2_p_value = stats.chisquare(cur_counts, ref_counts)
                return {
                    'target_drift_detected': chi2_p_value < self.drift_threshold,
                    'target_drift_score': chi2_stat,
                    'target_p_value': chi2_p_value,
                    'target_drift_method': 'chi2_test'
                }
                
        except Exception as e:
            logging.error(f"Target drift detection error: {e}")
            return {'target_drift_detected': False, 'target_drift_score': 0.0}
    
    def _generate_evidently_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: List[str]
    ) -> Dict[str, Any]:
        """Generate Evidently AI drift report."""
        try:
            # Create column mapping
            column_mapping = ColumnMapping()
            column_mapping.numerical_features = [f for f in features if pd.api.types.is_numeric_dtype(reference_data[f])]
            column_mapping.categorical_features = [f for f in features if not pd.api.types.is_numeric_dtype(reference_data[f])]
            
            # Create and run report
            report = Report(metrics=[
                DataDriftMetric(),
                DataQualityMetric()
            ])
            
            report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
            
            # Extract results
            report_dict = report.as_dict()
            
            return {
                'evidently_drift_detected': report_dict['metrics'][0]['result']['dataset_drift'],
                'evidently_drift_share': report_dict['metrics'][0]['result']['share_of_drifted_columns'],
                'evidently_drifted_features': report_dict['metrics'][0]['result']['drifted_features']
            }
            
        except Exception as e:
            logging.error(f"Evidently report generation error: {e}")
            return {'evidently_drift_detected': False, 'evidently_drift_share': 0.0}
    
    def _aggregate_drift_results(
        self,
        feature_drift_results: Dict[str, Any],
        target_drift_results: Dict[str, Any],
        evidently_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate all drift detection results."""
        
        # Determine overall drift status
        feature_drift_detected = len(feature_drift_results.get('drifted_features', [])) > 0
        target_drift_detected = target_drift_results.get('target_drift_detected', False)
        evidently_drift_detected = evidently_results.get('evidently_drift_detected', False)
        
        overall_drift_detected = feature_drift_detected or target_drift_detected or evidently_drift_detected
        
        # Calculate drift severity
        drift_severity = self._calculate_drift_severity(
            feature_drift_results,
            target_drift_results,
            evidently_results
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': overall_drift_detected,
            'drift_severity': drift_severity,
            'feature_drift': feature_drift_results,
            'target_drift': target_drift_results,
            'evidently_results': evidently_results,
            'recommendation': self._get_drift_recommendation(drift_severity)
        }
    
    def _calculate_drift_severity(
        self,
        feature_drift_results: Dict[str, Any],
        target_drift_results: Dict[str, Any],
        evidently_results: Dict[str, Any]
    ) -> str:
        """Calculate drift severity level."""
        
        num_drifted_features = len(feature_drift_results.get('drifted_features', []))
        total_features = len(feature_drift_results.get('feature_drift_scores', {}))
        
        if total_features == 0:
            return 'unknown'
        
        drift_ratio = num_drifted_features / total_features
        target_drift = target_drift_results.get('target_drift_detected', False)
        
        if target_drift or drift_ratio > 0.5:
            return 'high'
        elif drift_ratio > 0.2:
            return 'medium'
        elif drift_ratio > 0:
            return 'low'
        else:
            return 'none'
    
    def _get_drift_recommendation(self, drift_severity: str) -> str:
        """Get recommendation based on drift severity."""
        recommendations = {
            'high': 'Immediate model retraining recommended. Significant drift detected.',
            'medium': 'Schedule model retraining within 24 hours. Moderate drift detected.',
            'low': 'Monitor closely. Consider retraining if drift persists.',
            'none': 'No action required. Model performance is stable.',
            'unknown': 'Unable to assess drift. Check data quality.'
        }
        
        return recommendations.get(drift_severity, 'Unknown drift level.')
    
    def save_drift_report(self, drift_results: Dict[str, Any], filepath: str) -> None:
        """Save drift detection results to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(drift_results, f, indent=2, default=str)
            logging.info(f"Drift report saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving drift report: {e}")
