"""
Main XGBoost to SQL converter module
"""

import xgboost as xgb
import numpy as np
import json
from typing import Dict, List, Optional, Union
from .sql_generator import SQLGenerator
from .utils import ModelValidator


class XGBoostToSQL:
    """
    Convert XGBoost models to SQL queries for prediction
    """
    
    def __init__(self, model: Union[str, xgb.Booster, xgb.XGBClassifier, xgb.XGBRegressor]):
        """
        Initialize the converter with an XGBoost model
        
        Args:
            model: XGBoost model (file path, Booster, or sklearn-style model)
        """
        self.model = self._load_model(model)
        self.feature_names = None
        self.sql_generator = SQLGenerator()
        self.validator = ModelValidator()
        
        # Validate the model
        self.validator.validate_model(self.model)
        
    def _load_model(self, model) -> xgb.Booster:
        """Load XGBoost model from various input types"""
        if isinstance(model, str):
            # Load from file
            booster = xgb.Booster()
            booster.load_model(model)
            return booster
        elif isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
            # Extract booster from sklearn-style model
            return model.get_booster()
        elif isinstance(model, xgb.Booster):
            return model
        else:
            raise ValueError("Model must be a file path, XGBoost Booster, XGBClassifier, or XGBRegressor")
    
    def get_feature_names(self) -> List[str]:
        """Get feature names from the model"""
        if self.feature_names is None:
            # Try to get from model
            feature_names = self.model.feature_names
            if feature_names is None:
                # Generate default names
                num_features = self.model.num_features()
                feature_names = [f"feature_{i}" for i in range(num_features)]
            self.feature_names = feature_names
        return self.feature_names
    
    def set_feature_names(self, feature_names: List[str]):
        """Set custom feature names"""
        if len(feature_names) != self.model.num_features():
            raise ValueError(f"Expected {self.model.num_features()} feature names, got {len(feature_names)}")
        self.feature_names = feature_names
    
    def _extract_tree_structure(self, tree_id: int) -> Dict:
        """Extract the structure of a single tree"""
        tree_dump = self.model.get_dump(dump_format='json')[tree_id]
        return json.loads(tree_dump)
    
    def _tree_to_sql_conditions(self, tree_dict: Dict, table_name: str = "input_table") -> str:
        """Convert a single tree to SQL CASE statement"""
        feature_names = self.get_feature_names()
        
        def build_case_statement(node, conditions_stack=[]):
            """Recursively build CASE statement from tree nodes"""
            if 'leaf' in node:
                # Leaf node - return the prediction value
                return f"{node['leaf']}"
            
            # XGBoost uses feature names like 'f0', 'f1', etc. in JSON dump
            split_feature = node['split']
            if split_feature.startswith('f') and split_feature[1:].isdigit():
                # Default feature names like 'f0', 'f1', etc.
                feature_id = int(split_feature[1:])
                feature_name = feature_names[feature_id]
            else:
                # Custom feature names - use as is
                feature_name = split_feature
            split_condition = node['split_condition']
            
            # Build condition for current split
            left_condition = f"{table_name}.{feature_name} < {split_condition}"
            right_condition = f"{table_name}.{feature_name} >= {split_condition}"
            
            # Recursively process children
            left_result = build_case_statement(node['children'][0], conditions_stack + [left_condition])
            right_result = build_case_statement(node['children'][1], conditions_stack + [right_condition])
            
            return f"CASE WHEN {left_condition} THEN {left_result} ELSE {right_result} END"
        
        return build_case_statement(tree_dict)
    
    def convert_to_sql(self, 
                      table_name: str = "input_table",
                      output_column: str = "prediction",
                      include_probability: bool = False,
                      max_trees: Optional[int] = None) -> str:
        """
        Convert the XGBoost model to a SQL query
        
        Args:
            table_name: Name of the input table in SQL
            output_column: Name of the output prediction column
            include_probability: Whether to include probability calculation (for classification)
            max_trees: Maximum number of trees to include (None for all trees)
            
        Returns:
            SQL query string
        """
        num_trees = self.model.num_boosted_rounds()
        if max_trees is not None:
            num_trees = min(num_trees, max_trees)
        
        # Generate SQL for each tree
        tree_sql_parts = []
        for tree_id in range(num_trees):
            tree_dict = self._extract_tree_structure(tree_id)
            tree_sql = self._tree_to_sql_conditions(tree_dict, table_name)
            tree_sql_parts.append(f"({tree_sql})")
        
        # Combine all trees (sum for regression, need sigmoid for classification)
        base_score = self.model.attr('base_score')
        base_prediction = float(base_score) if base_score is not None else 0.5
        
        if len(tree_sql_parts) == 0:
            combined_prediction = str(base_prediction)
        else:
            trees_sum = " + ".join(tree_sql_parts)
            combined_prediction = f"{base_prediction} + ({trees_sum})"
        
        # Build the final SQL query
        sql_parts = [
            f"SELECT *,",
            f"    {combined_prediction} AS {output_column}_raw"
        ]
        
        # Add probability calculation for classification
        if include_probability:
            sql_parts.append(f"    1.0 / (1.0 + EXP(-({combined_prediction}))) AS {output_column}_probability")
            sql_parts.append(f"    CASE WHEN 1.0 / (1.0 + EXP(-({combined_prediction}))) > 0.5 THEN 1 ELSE 0 END AS {output_column}")
        else:
            sql_parts.append(f"    {combined_prediction} AS {output_column}")
        
        sql_parts.append(f"FROM {table_name}")
        
        return "\n".join(sql_parts)
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "num_features": self.model.num_features(),
            "num_trees": self.model.num_boosted_rounds(),
            "feature_names": self.get_feature_names(),
            "objective": self.model.attr('objective'),
            "base_score": self.model.attr('base_score')
        }