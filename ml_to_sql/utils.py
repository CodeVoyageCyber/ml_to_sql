"""
Utility functions for model validation and helper operations
"""

import xgboost as xgb
import numpy as np
from typing import Dict, List, Any


class ModelValidator:
    """
    Validate XGBoost models for SQL conversion compatibility
    """
    
    def validate_model(self, model: xgb.Booster) -> Dict[str, Any]:
        """
        Validate that the model can be converted to SQL
        
        Args:
            model: XGBoost Booster model
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "info": {}
        }
        
        # Check basic model properties
        try:
            results["info"]["num_features"] = model.num_features()
            results["info"]["num_trees"] = model.num_boosted_rounds()
            results["info"]["objective"] = model.attr('objective')
        except Exception as e:
            results["errors"].append(f"Could not read basic model properties: {str(e)}")
            results["is_valid"] = False
            return results
        
        # Check if model has too many trees (performance warning)
        num_trees = model.num_boosted_rounds()
        if num_trees > 100:
            results["warnings"].append(
                f"Model has {num_trees} trees. SQL query may be very large and slow. "
                "Consider using max_trees parameter to limit the number of trees."
            )
        
        # Check if model has too many features
        num_features = model.num_features()
        if num_features > 100:
            results["warnings"].append(
                f"Model has {num_features} features. Ensure your SQL database can handle "
                "complex queries with many columns."
            )
        
        # Check for unsupported objectives
        objective = model.attr('objective')
        if objective and 'multi:' in objective:
            results["errors"].append(
                f"Multi-class objectives ('{objective}') are not currently supported for SQL conversion."
            )
            results["is_valid"] = False
        
        # Validate tree structure
        try:
            tree_dump = model.get_dump(dump_format='json')
            if not tree_dump:
                results["errors"].append("Model contains no trees")
                results["is_valid"] = False
        except Exception as e:
            results["errors"].append(f"Could not extract tree structure: {str(e)}")
            results["is_valid"] = False
        
        return results
    
    def estimate_sql_complexity(self, model: xgb.Booster) -> Dict[str, Any]:
        """
        Estimate the complexity of the generated SQL query
        
        Args:
            model: XGBoost Booster model
            
        Returns:
            Dictionary with complexity estimates
        """
        num_trees = model.num_boosted_rounds()
        
        # Rough estimates based on typical tree structures
        estimated_nodes_per_tree = 15  # Average for typical models
        estimated_conditions_per_tree = estimated_nodes_per_tree // 2
        
        total_conditions = num_trees * estimated_conditions_per_tree
        
        # Estimate query length (very rough)
        chars_per_condition = 50  # Average character count per condition
        estimated_query_length = total_conditions * chars_per_condition
        
        complexity_level = "Low"
        if total_conditions > 100:
            complexity_level = "Medium"
        if total_conditions > 500:
            complexity_level = "High"
        if total_conditions > 1000:
            complexity_level = "Very High"
        
        return {
            "num_trees": num_trees,
            "estimated_total_conditions": total_conditions,
            "estimated_query_length_chars": estimated_query_length,
            "complexity_level": complexity_level,
            "recommendations": self._get_complexity_recommendations(complexity_level, num_trees)
        }
    
    def _get_complexity_recommendations(self, complexity_level: str, num_trees: int) -> List[str]:
        """Get recommendations based on complexity level"""
        recommendations = []
        
        if complexity_level in ["High", "Very High"]:
            recommendations.append("Consider using max_trees parameter to limit query size")
            recommendations.append("Test query performance on a small dataset first")
            recommendations.append("Consider creating a SQL view instead of embedding the query")
        
        if num_trees > 200:
            recommendations.append("Consider model pruning or using fewer boosting rounds")
            recommendations.append("Split predictions into batches for large datasets")
        
        if complexity_level == "Very High":
            recommendations.append("Consider using a simpler model or feature selection")
            recommendations.append("Query may exceed database query length limits")
        
        return recommendations


def compare_predictions(model: xgb.Booster, 
                       sql_predictions: np.ndarray, 
                       test_data: np.ndarray,
                       tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Compare XGBoost model predictions with SQL-generated predictions
    
    Args:
        model: XGBoost model
        sql_predictions: Predictions from SQL query
        test_data: Test dataset used for predictions
        tolerance: Tolerance for floating point comparison
        
    Returns:
        Dictionary with comparison results
    """
    # Get model predictions
    import xgboost as xgb
    dtest = xgb.DMatrix(test_data)
    model_predictions = model.predict(dtest)
    
    # Compare predictions
    diff = np.abs(model_predictions - sql_predictions)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Check if predictions match within tolerance
    matches = np.all(diff <= tolerance)
    
    return {
        "predictions_match": matches,
        "max_difference": float(max_diff),
        "mean_difference": float(mean_diff),
        "tolerance_used": tolerance,
        "num_predictions": len(model_predictions),
        "percentage_exact_match": float(np.sum(diff == 0) / len(diff) * 100)
    }


def generate_test_data(num_samples: int = 100, num_features: int = 5, random_seed: int = 42) -> np.ndarray:
    """
    Generate random test data for model validation
    
    Args:
        num_samples: Number of samples to generate
        num_features: Number of features
        random_seed: Random seed for reproducibility
        
    Returns:
        Generated test data array
    """
    np.random.seed(random_seed)
    return np.random.randn(num_samples, num_features)