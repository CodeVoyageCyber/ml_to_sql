# # XGBoost to SQL Converter

A Python library that converts trained XGBoost models into SQL queries for in-database prediction. This allows you to deploy machine learning models directly in your SQL database without external dependencies.

## 🚀 Features

- **Direct Model Conversion**: Convert XGBoost models (Booster, XGBClassifier, XGBRegressor) to SQL
- **Multiple Input Formats**: Support for model files, scikit-learn style models, and XGBoost Boosters
- **SQL Dialect Compatibility**: Generated SQL works with most modern databases (PostgreSQL, MySQL, SQL Server, BigQuery)
- **Classification & Regression**: Support for both classification (with probability) and regression models
- **Performance Optimization**: Options to limit trees and batch processing for large datasets
- **Validation Tools**: Built-in model validation and SQL syntax checking
- **Comprehensive Examples**: Detailed usage examples and test cases

## 📦 Installation

### Install from source:

```bash
git clone <repository-url>
cd ml_to_sql
pip install -r requirements.txt
```

### Required Dependencies:

- xgboost >= 1.6.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0

## 🏃‍♂️ Quick Start

### Basic Classification Example

```python
import xgboost as xgb
from sklearn.datasets import make_classification
from ml_to_sql import XGBoostToSQL

# Train a model
X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
model.fit(X, y)

# Convert to SQL
converter = XGBoostToSQL(model)
converter.set_feature_names(['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'])

# Generate SQL query
sql_query = converter.convert_to_sql(
    table_name="customer_data",
    output_column="churn_prediction",
    include_probability=True
)

print(sql_query)
```

### Basic Regression Example

```python
import xgboost as xgb
from sklearn.datasets import make_regression
from ml_to_sql import XGBoostToSQL

# Train a regression model
X, y = make_regression(n_samples=1000, n_features=4, noise=0.1, random_state=42)
model = xgb.XGBRegressor(n_estimators=8, max_depth=3, random_state=42)
model.fit(X, y)

# Convert to SQL
converter = XGBoostToSQL(model)
sql_query = converter.convert_to_sql(
    table_name="housing_data",
    output_column="price_prediction"
)

print(sql_query)
```

## 📚 Documentation

### Core Classes

#### `XGBoostToSQL`

Main converter class for transforming XGBoost models to SQL.

**Parameters:**
- `model`: XGBoost model (file path, Booster, XGBClassifier, or XGBRegressor)

**Key Methods:**
- `convert_to_sql()`: Generate SQL query from model
- `set_feature_names()`: Set custom feature names
- `get_model_info()`: Get model information

#### `SQLGenerator`

Utility class for SQL formatting and validation.

**Key Methods:**
- `format_sql()`: Format SQL for readability
- `create_view_query()`: Create SQL view from query
- `validate_sql_syntax()`: Basic SQL validation
- `add_filters()`: Add WHERE clause filters

#### `ModelValidator`

Validation and analysis tools for XGBoost models.

**Key Methods:**
- `validate_model()`: Check model compatibility
- `estimate_sql_complexity()`: Analyze query complexity

### API Reference

#### `convert_to_sql()` Parameters

```python
converter.convert_to_sql(
    table_name="input_table",        # Input table name
    output_column="prediction",      # Output column name
    include_probability=False,       # Include probability calculation (classification)
    max_trees=None                   # Limit number of trees (None = all)
)
```

## 💡 Usage Examples

### Loading Models from Files

```python
# Load from saved model file
converter = XGBoostToSQL('path/to/model.json')
converter.set_feature_names(['feature_1', 'feature_2', 'feature_3'])

sql_query = converter.convert_to_sql()
```

### Working with Large Models

```python
# Limit trees for performance
sql_query = converter.convert_to_sql(max_trees=50)

# Check model complexity
from ml_to_sql.utils import ModelValidator
validator = ModelValidator()
complexity = validator.estimate_sql_complexity(model.get_booster())
print(f"Complexity: {complexity['complexity_level']}")
```

### SQL Formatting and Views

```python
from ml_to_sql.sql_generator import SQLGenerator

# Format SQL for readability
sql_gen = SQLGenerator()
formatted_sql = sql_gen.format_sql(sql_query)

# Create a database view
view_sql = sql_gen.create_view_query(sql_query, "ml_predictions")
```

### Model Validation

```python
from ml_to_sql.utils import ModelValidator

validator = ModelValidator()
results = validator.validate_model(model.get_booster())

if results['is_valid']:
    print("Model is compatible!")
else:
    print("Issues found:", results['errors'])
```

## 🧪 Running Examples

### Basic Usage
```bash
cd examples
python basic_usage.py
```

### Advanced Features
```bash
cd examples  
python advanced_usage.py
```

### Run Tests
```bash
cd tests
python test_converter.py
```

## 🏗️ Generated SQL Structure

The generated SQL queries use nested CASE statements to represent the decision tree logic:

```sql
SELECT *,
    0.5 + (
        (CASE WHEN input_table.feature_1 < 0.5 
         THEN (CASE WHEN input_table.feature_2 < 1.2 
               THEN 0.1 
               ELSE -0.1 END) 
         ELSE 0.05 END) +
        (CASE WHEN input_table.feature_1 < 0.8 
         THEN 0.2 
         ELSE -0.2 END)
    ) AS prediction
FROM input_table
```

For classification with probability:
```sql
SELECT *,
    -- Raw prediction value
    0.5 + (...tree calculations...) AS prediction_raw,
    -- Probability using sigmoid
    1.0 / (1.0 + EXP(-(0.5 + (...tree calculations...)))) AS prediction_probability,
    -- Binary classification
    CASE WHEN 1.0 / (1.0 + EXP(-(0.5 + (...tree calculations...)))) > 0.5 
         THEN 1 ELSE 0 END AS prediction
FROM input_table
```

## ⚡ Performance Considerations

### For Large Models:
- Use `max_trees` parameter to limit query complexity
- Consider model pruning before conversion
- Test queries on small datasets first
- Create database views for reusable predictions

### For Large Datasets:
- Use batch processing with filters
- Create indexes on feature columns
- Consider database-specific optimizations

### SQL Query Size:
- Models with 100+ trees may generate very large queries
- Some databases have query length limits
- Monitor query execution performance

## 🎯 Use Cases

### Real-time Scoring
Deploy models directly in your database for real-time predictions without external API calls.

### Data Pipeline Integration
Integrate ML predictions into existing SQL-based data workflows and ETL processes.

### Edge Computing
Run predictions in edge databases without external dependencies.

### Audit and Compliance
Keep all prediction logic in the database for easier auditing and compliance.

## ⚠️ Limitations

- **Multi-class Classification**: Currently not supported
- **Complex Features**: Feature engineering must be done before SQL execution
- **Query Size**: Very large models may generate unwieldy SQL queries
- **Database Compatibility**: Some advanced SQL functions may not be available in all databases

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the examples directory for common use cases
- Review the test cases for detailed usage patterns

## 🔗 Related Projects

- [XGBoost](https://github.com/dmlc/xgboost) - The original XGBoost library
- [sklearn2sql](https://github.com/paulfitz/sklearn2sql) - Similar tool for scikit-learn models
- [m2cgen](https://github.com/BayesWitnesses/m2cgen) - Model to code generator for various formats