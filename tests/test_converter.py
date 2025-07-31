"""
Unit tests for XGBoost to SQL converter
"""

import unittest
import tempfile
import os
import sys
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression

# Add the parent directory to the path to import ml_to_sql
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml_to_sql import XGBoostToSQL
from ml_to_sql.utils import ModelValidator, generate_test_data
from ml_to_sql.sql_generator import SQLGenerator


class TestXGBoostToSQL(unittest.TestCase):
    """Test cases for XGBoost to SQL conversion"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple classification model
        X, y = make_classification(
            n_samples=100, n_features=3, n_informative=2, 
            n_redundant=1, random_state=42
        )
        self.clf_model = xgb.XGBClassifier(n_estimators=3, max_depth=2, random_state=42)
        self.clf_model.fit(X, y)
        
        # Create a simple regression model
        X_reg, y_reg = make_regression(
            n_samples=100, n_features=3, noise=0.1, random_state=42
        )
        self.reg_model = xgb.XGBRegressor(n_estimators=3, max_depth=2, random_state=42)
        self.reg_model.fit(X_reg, y_reg)
        
        self.feature_names = ['feature_0', 'feature_1', 'feature_2']
    
    def test_converter_initialization(self):
        """Test converter initialization with different model types"""
        # Test with XGBClassifier
        converter = XGBoostToSQL(self.clf_model)
        self.assertIsNotNone(converter.model)
        
        # Test with XGBRegressor
        converter = XGBoostToSQL(self.reg_model)
        self.assertIsNotNone(converter.model)
        
        # Test with Booster
        booster = self.clf_model.get_booster()
        converter = XGBoostToSQL(booster)
        self.assertIsNotNone(converter.model)
    
    def test_feature_names(self):
        """Test feature name handling"""
        converter = XGBoostToSQL(self.clf_model)
        
        # Test default feature names
        default_names = converter.get_feature_names()
        self.assertEqual(len(default_names), 3)
        self.assertTrue(all(name.startswith('feature_') for name in default_names))
        
        # Test setting custom feature names
        converter.set_feature_names(self.feature_names)
        self.assertEqual(converter.get_feature_names(), self.feature_names)
        
        # Test invalid feature names length
        with self.assertRaises(ValueError):
            converter.set_feature_names(['feature_0', 'feature_1'])  # Wrong length
    
    def test_sql_generation_classification(self):
        """Test SQL generation for classification models"""
        converter = XGBoostToSQL(self.clf_model)
        converter.set_feature_names(self.feature_names)
        
        # Test basic SQL generation
        sql = converter.convert_to_sql()
        self.assertIn('SELECT', sql)
        self.assertIn('FROM', sql)
        self.assertIn('CASE WHEN', sql)
        
        # Test SQL with probability
        sql_prob = converter.convert_to_sql(include_probability=True)
        self.assertIn('EXP', sql_prob)
        self.assertIn('probability', sql_prob)
        
        # Test custom table and column names
        sql_custom = converter.convert_to_sql(
            table_name='my_table',
            output_column='my_prediction'
        )
        self.assertIn('my_table', sql_custom)
        self.assertIn('my_prediction', sql_custom)
    
    def test_sql_generation_regression(self):
        """Test SQL generation for regression models"""
        converter = XGBoostToSQL(self.reg_model)
        converter.set_feature_names(self.feature_names)
        
        sql = converter.convert_to_sql()
        self.assertIn('SELECT', sql)
        self.assertIn('FROM', sql)
        self.assertIn('CASE WHEN', sql)
    
    def test_model_info(self):
        """Test model information extraction"""
        converter = XGBoostToSQL(self.clf_model)
        info = converter.get_model_info()
        
        self.assertIn('num_features', info)
        self.assertIn('num_trees', info)
        self.assertEqual(info['num_features'], 3)
        self.assertEqual(info['num_trees'], 3)
    
    def test_max_trees_limit(self):
        """Test limiting number of trees in SQL"""
        converter = XGBoostToSQL(self.clf_model)
        converter.set_feature_names(self.feature_names)
        
        # Generate SQL with limited trees
        sql_limited = converter.convert_to_sql(max_trees=2)
        sql_full = converter.convert_to_sql()
        
        # Limited SQL should be shorter
        self.assertLess(len(sql_limited), len(sql_full))
    
    def test_model_file_loading(self):
        """Test loading model from file"""
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            self.clf_model.save_model(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Load model from file
            converter = XGBoostToSQL(tmp_path)
            self.assertIsNotNone(converter.model)
            
            # Test that it can generate SQL
            sql = converter.convert_to_sql()
            self.assertIn('SELECT', sql)
        
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestModelValidator(unittest.TestCase):
    """Test cases for model validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        X, y = make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, random_state=42)
        self.model = xgb.XGBClassifier(n_estimators=2, max_depth=2, random_state=42)
        self.model.fit(X, y)
    
    def test_model_validation(self):
        """Test basic model validation"""
        validator = ModelValidator()
        results = validator.validate_model(self.model.get_booster())
        
        self.assertIn('is_valid', results)
        self.assertIn('warnings', results)
        self.assertIn('errors', results)
        self.assertIn('info', results)
    
    def test_complexity_estimation(self):
        """Test SQL complexity estimation"""
        validator = ModelValidator()
        complexity = validator.estimate_sql_complexity(self.model.get_booster())
        
        self.assertIn('num_trees', complexity)
        self.assertIn('complexity_level', complexity)
        self.assertIn('recommendations', complexity)
        self.assertIsInstance(complexity['recommendations'], list)


class TestSQLGenerator(unittest.TestCase):
    """Test cases for SQL generation utilities"""
    
    def test_sql_validation(self):
        """Test SQL syntax validation"""
        generator = SQLGenerator()
        
        # Valid SQL
        valid_sql = "SELECT * FROM table WHERE col = 1"
        result = generator.validate_sql_syntax(valid_sql)
        self.assertTrue(result['valid'])
        
        # Invalid SQL (unbalanced parentheses)
        invalid_sql = "SELECT * FROM table WHERE (col = 1"
        result = generator.validate_sql_syntax(invalid_sql)
        self.assertFalse(result['balanced_parentheses'])
    
    def test_view_creation(self):
        """Test SQL view creation"""
        generator = SQLGenerator()
        base_query = "SELECT * FROM table"
        view_sql = generator.create_view_query(base_query, "my_view")
        
        self.assertIn("CREATE VIEW my_view", view_sql)
        self.assertIn(base_query, view_sql)
    
    def test_filter_addition(self):
        """Test adding filters to SQL queries"""
        generator = SQLGenerator()
        base_query = "SELECT * FROM table"
        filters = {"col1": "> 10", "col2": "= 'value'"}
        
        filtered_sql = generator.add_filters(base_query, filters)
        self.assertIn("WHERE", filtered_sql)
        self.assertIn("col1 > 10", filtered_sql)
        self.assertIn("col2 = 'value'", filtered_sql)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""
    
    def test_generate_test_data(self):
        """Test test data generation"""
        data = generate_test_data(num_samples=10, num_features=3)
        self.assertEqual(data.shape, (10, 3))
        self.assertIsInstance(data, np.ndarray)
    
    def test_generate_test_data_reproducible(self):
        """Test that test data generation is reproducible"""
        data1 = generate_test_data(random_seed=42)
        data2 = generate_test_data(random_seed=42)
        np.testing.assert_array_equal(data1, data2)


if __name__ == '__main__':
    print("Running XGBoost to SQL Converter Tests")
    print("=====================================")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestXGBoostToSQL))
    test_suite.addTest(unittest.makeSuite(TestModelValidator))
    test_suite.addTest(unittest.makeSuite(TestSQLGenerator))
    test_suite.addTest(unittest.makeSuite(TestUtilityFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    if result.wasSuccessful():
        print(f"\nAll tests passed! ({result.testsRun} tests)")
    else:
        print(f"\nTests failed: {len(result.failures)} failures, {len(result.errors)} errors")