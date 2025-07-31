"""
Basic usage example for XGBoost to SQL conversion
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from ml_to_sql import XGBoostToSQL


def create_sample_classification_model():
    """Create a sample XGBoost classification model"""
    print("Creating sample classification dataset...")
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train XGBoost model
    print("Training XGBoost classifier...")
    model = xgb.XGBClassifier(
        n_estimators=10,  # Keep small for readable SQL
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save sample data
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test.to_csv('examples/sample_data/test_data_classification.csv', index=False)
    
    return model, X_test, y_test, feature_names


def create_sample_regression_model():
    """Create a sample XGBoost regression model"""
    print("Creating sample regression dataset...")
    
    # Generate sample data
    X, y = make_regression(
        n_samples=1000,
        n_features=4,
        noise=0.1,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train XGBoost model
    print("Training XGBoost regressor...")
    model = xgb.XGBRegressor(
        n_estimators=8,  # Keep small for readable SQL
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save sample data
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test.to_csv('examples/sample_data/test_data_regression.csv', index=False)
    
    return model, X_test, y_test, feature_names


def demonstrate_classification():
    """Demonstrate classification model conversion"""
    print("\n" + "="*50)
    print("CLASSIFICATION EXAMPLE")
    print("="*50)
    
    # Create and train model
    model, X_test, y_test, feature_names = create_sample_classification_model()
    
    # Save the model
    model.save_model('examples/sample_data/classification_model.json')
    
    # Convert to SQL
    print("\nConverting model to SQL...")
    converter = XGBoostToSQL(model)
    converter.set_feature_names(feature_names)
    
    # Display model info
    info = converter.get_model_info()
    print(f"Model Info:")
    print(f"  - Features: {info['num_features']}")
    print(f"  - Trees: {info['num_trees']}")
    print(f"  - Objective: {info['objective']}")
    
    # Generate SQL for classification (with probability)
    sql_query = converter.convert_to_sql(
        table_name="classification_data",
        output_column="predicted_class",
        include_probability=True
    )
    
    print(f"\nGenerated SQL Query:")
    print("-" * 40)
    print(sql_query)
    
    # Save SQL to file
    with open('examples/sample_data/classification_query.sql', 'w') as f:
        f.write(sql_query)
    print(f"\nSQL query saved to 'examples/sample_data/classification_query.sql'")


def demonstrate_regression():
    """Demonstrate regression model conversion"""
    print("\n" + "="*50)
    print("REGRESSION EXAMPLE")
    print("="*50)
    
    # Create and train model
    model, X_test, y_test, feature_names = create_sample_regression_model()
    
    # Save the model
    model.save_model('examples/sample_data/regression_model.json')
    
    # Convert to SQL
    print("\nConverting model to SQL...")
    converter = XGBoostToSQL(model)
    converter.set_feature_names(feature_names)
    
    # Display model info
    info = converter.get_model_info()
    print(f"Model Info:")
    print(f"  - Features: {info['num_features']}")
    print(f"  - Trees: {info['num_trees']}")
    print(f"  - Objective: {info['objective']}")
    
    # Estimate complexity
    from ml_to_sql.utils import ModelValidator
    validator = ModelValidator()
    complexity = validator.estimate_sql_complexity(model.get_booster())
    print(f"  - SQL Complexity: {complexity['complexity_level']}")
    print(f"  - Estimated Conditions: {complexity['estimated_total_conditions']}")
    
    # Generate SQL for regression
    sql_query = converter.convert_to_sql(
        table_name="regression_data",
        output_column="predicted_value"
    )
    
    print(f"\nGenerated SQL Query:")
    print("-" * 40)
    print(sql_query)
    
    # Save SQL to file
    with open('examples/sample_data/regression_query.sql', 'w') as f:
        f.write(sql_query)
    print(f"\nSQL query saved to 'examples/sample_data/regression_query.sql'")


def demonstrate_model_loading():
    """Demonstrate loading models from files"""
    print("\n" + "="*50)
    print("MODEL LOADING EXAMPLE")
    print("="*50)
    
    # Load previously saved classification model
    print("Loading classification model from file...")
    converter = XGBoostToSQL('examples/sample_data/classification_model.json')
    
    # Set feature names (since they're not stored in the model file)
    feature_names = [f"feature_{i}" for i in range(converter.model.num_features())]
    converter.set_feature_names(feature_names)
    
    # Generate simpler SQL (limit trees for readability)
    sql_query = converter.convert_to_sql(
        table_name="my_data",
        output_column="prediction",
        max_trees=3  # Limit to first 3 trees for demonstration
    )
    
    print("Generated SQL (limited to 3 trees):")
    print("-" * 40)
    print(sql_query)


if __name__ == "__main__":
    print("XGBoost to SQL Converter - Usage Examples")
    print("========================================")
    
    # Create output directory
    os.makedirs('examples/sample_data', exist_ok=True)
    
    try:
        # Run demonstrations
        demonstrate_classification()
        demonstrate_regression() 
        demonstrate_model_loading()
        
        print("\n" + "="*50)
        print("EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("\nGenerated files:")
        print("- examples/sample_data/classification_model.json")
        print("- examples/sample_data/regression_model.json") 
        print("- examples/sample_data/classification_query.sql")
        print("- examples/sample_data/regression_query.sql")
        print("- examples/sample_data/test_data_classification.csv")
        print("- examples/sample_data/test_data_regression.csv")
        
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()