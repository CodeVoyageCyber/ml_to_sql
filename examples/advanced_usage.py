"""
Advanced usage examples for XGBoost to SQL conversion
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import xgboost as xgb
from ml_to_sql import XGBoostToSQL
from ml_to_sql.sql_generator import SQLGenerator
from ml_to_sql.utils import ModelValidator, compare_predictions, generate_test_data


def demonstrate_sql_formatting():
    """Demonstrate SQL formatting and view creation"""
    print("\n" + "="*50)
    print("SQL FORMATTING AND VIEW CREATION")
    print("="*50)
    
    # Load a pre-trained model (assuming basic_usage.py was run first)
    try:
        converter = XGBoostToSQL('examples/sample_data/classification_model.json')
        feature_names = [f"feature_{i}" for i in range(converter.model.num_features())]
        converter.set_feature_names(feature_names)
    except:
        print("Please run basic_usage.py first to create sample models")
        return
    
    # Generate basic SQL
    sql_query = converter.convert_to_sql(
        table_name="customer_data",
        output_column="churn_prediction",
        include_probability=True,
        max_trees=5
    )
    
    # Use SQL generator for formatting and views
    sql_gen = SQLGenerator()
    
    # Format the SQL
    formatted_sql = sql_gen.format_sql(sql_query)
    print("Formatted SQL Query:")
    print("-" * 40)
    print(formatted_sql)
    
    # Create a view
    view_sql = sql_gen.create_view_query(sql_query, "customer_churn_predictions")
    print(f"\nView Creation SQL:")
    print("-" * 40)
    print(view_sql)
    
    # Validate SQL syntax
    validation = sql_gen.validate_sql_syntax(sql_query)
    print(f"\nSQL Validation Results:")
    for key, value in validation.items():
        print(f"  - {key}: {value}")


def demonstrate_model_validation():
    """Demonstrate model validation and complexity analysis"""
    print("\n" + "="*50)
    print("MODEL VALIDATION AND COMPLEXITY ANALYSIS")
    print("="*50)
    
    # Load model
    try:
        converter = XGBoostToSQL('examples/sample_data/regression_model.json')
    except:
        print("Please run basic_usage.py first to create sample models")
        return
    
    # Validate model
    validator = ModelValidator()
    validation_results = validator.validate_model(converter.model)
    
    print("Model Validation Results:")
    print(f"  - Valid: {validation_results['is_valid']}")
    print(f"  - Warnings: {len(validation_results['warnings'])}")
    print(f"  - Errors: {len(validation_results['errors'])}")
    
    for warning in validation_results['warnings']:
        print(f"    WARNING: {warning}")
    
    for error in validation_results['errors']:
        print(f"    ERROR: {error}")
    
    # Complexity analysis
    complexity = validator.estimate_sql_complexity(converter.model)
    print(f"\nComplexity Analysis:")
    print(f"  - Trees: {complexity['num_trees']}")
    print(f"  - Estimated Conditions: {complexity['estimated_total_conditions']}")
    print(f"  - Complexity Level: {complexity['complexity_level']}")
    print(f"  - Estimated Query Length: {complexity['estimated_query_length_chars']} chars")
    
    print(f"\nRecommendations:")
    for rec in complexity['recommendations']:
        print(f"  - {rec}")


def demonstrate_prediction_comparison():
    """Demonstrate comparing XGBoost predictions with SQL predictions"""
    print("\n" + "="*50)
    print("PREDICTION COMPARISON")
    print("="*50)
    
    # Create a simple model for testing
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
    
    # Train a simple model
    model = xgb.XGBRegressor(n_estimators=3, max_depth=2, random_state=42)
    model.fit(X, y)
    
    # Convert to SQL
    converter = XGBoostToSQL(model)
    sql_query = converter.convert_to_sql(table_name="test_data")
    
    print("Generated SQL Query for Testing:")
    print("-" * 40)
    print(sql_query[:500] + "..." if len(sql_query) > 500 else sql_query)
    
    # Generate test data
    test_data = generate_test_data(num_samples=10, num_features=3)
    
    # Get XGBoost predictions
    xgb_predictions = model.predict(test_data)
    
    print(f"\nXGBoost Predictions (first 5):")
    for i, pred in enumerate(xgb_predictions[:5]):
        print(f"  Sample {i+1}: {pred:.6f}")
    
    # Note: In a real scenario, you would execute the SQL query against a database
    # and compare the results using the compare_predictions function
    print(f"\nNote: To complete the comparison, execute the SQL query in your database")
    print(f"and use ml_to_sql.utils.compare_predictions() to validate results.")


def demonstrate_batching():
    """Demonstrate batch processing for large datasets"""
    print("\n" + "="*50)
    print("BATCH PROCESSING FOR LARGE DATASETS")
    print("="*50)
    
    try:
        converter = XGBoostToSQL('examples/sample_data/classification_model.json')
        feature_names = [f"feature_{i}" for i in range(converter.model.num_features())]
        converter.set_feature_names(feature_names)
    except:
        print("Please run basic_usage.py first to create sample models")
        return
    
    # Generate base SQL
    base_sql = converter.convert_to_sql(
        table_name="large_dataset",
        output_column="prediction",
        max_trees=10
    )
    
    # Create batched queries
    sql_gen = SQLGenerator()
    
    print("Original query would process entire table:")
    print("-" * 40)
    print(base_sql[:200] + "...")
    
    # Example of how to add filters for batching
    batch_filters = {
        "id": "BETWEEN 1 AND 1000",  # First batch
        "processed_date": "> '2024-01-01'"
    }
    
    batched_sql = sql_gen.add_filters(base_sql, batch_filters)
    print(f"\nBatched SQL with filters:")
    print("-" * 40)
    print(batched_sql[:300] + "...")


def demonstrate_different_sql_dialects():
    """Show SQL generation considerations for different databases"""
    print("\n" + "="*50)
    print("SQL DIALECT CONSIDERATIONS")
    print("="*50)
    
    try:
        converter = XGBoostToSQL('examples/sample_data/classification_model.json')
        feature_names = [f"feature_{i}" for i in range(converter.model.num_features())]
        converter.set_feature_names(feature_names)
    except:
        print("Please run basic_usage.py first to create sample models")
        return
    
    # Generate SQL with probability (uses EXP function)
    sql_with_prob = converter.convert_to_sql(
        table_name="data",
        output_column="prediction",
        include_probability=True,
        max_trees=2
    )
    
    print("SQL with probability calculation (uses EXP function):")
    print("Supported by: PostgreSQL, MySQL, SQL Server, BigQuery")
    print("-" * 40)
    print(sql_with_prob)
    
    print(f"\nNote: Different SQL databases may have variations in:")
    print("  - Mathematical functions (EXP, LOG)")
    print("  - CASE statement syntax")
    print("  - Column naming conventions")
    print("  - Query length limits")
    print("  - Performance characteristics")


if __name__ == "__main__":
    print("XGBoost to SQL Converter - Advanced Examples")
    print("===========================================")
    
    try:
        demonstrate_sql_formatting()
        demonstrate_model_validation()
        demonstrate_prediction_comparison()
        demonstrate_batching()
        demonstrate_different_sql_dialects()
        
        print("\n" + "="*50)
        print("ADVANCED EXAMPLES COMPLETED!")
        print("="*50)
        
    except Exception as e:
        print(f"\nError during advanced demonstration: {str(e)}")
        import traceback
        traceback.print_exc()