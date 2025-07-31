"""
SQL generation utilities and formatters
"""

from typing import List, Dict, Optional
import re


class SQLGenerator:
    """
    Utility class for generating and formatting SQL queries
    """
    
    def __init__(self):
        self.indentation = "    "
    
    def format_sql(self, sql: str) -> str:
        """Format SQL query for better readability"""
        # Basic SQL formatting
        formatted = sql.replace(",", ",\n" + self.indentation)
        formatted = re.sub(r'\bFROM\b', '\nFROM', formatted)
        formatted = re.sub(r'\bWHERE\b', '\nWHERE', formatted)
        formatted = re.sub(r'\bORDER BY\b', '\nORDER BY', formatted)
        formatted = re.sub(r'\bGROUP BY\b', '\nGROUP BY', formatted)
        
        return formatted.strip()
    
    def create_batch_prediction_query(self, 
                                    base_query: str,
                                    batch_size: int = 1000,
                                    id_column: str = "id") -> List[str]:
        """
        Split a prediction query into batches for large datasets
        
        Args:
            base_query: The base prediction SQL query
            batch_size: Number of rows per batch
            id_column: Column to use for batching
            
        Returns:
            List of SQL queries for each batch
        """
        batch_queries = []
        
        # Extract table name from the query
        table_match = re.search(r'FROM\s+(\w+)', base_query, re.IGNORECASE)
        if not table_match:
            raise ValueError("Could not extract table name from query")
        
        table_name = table_match.group(1)
        
        # Create batched queries
        batch_template = f"""
        WITH batch_data AS (
            SELECT *,
                ROW_NUMBER() OVER (ORDER BY {id_column}) as row_num
            FROM {table_name}
        )
        {base_query.replace(f'FROM {table_name}', 'FROM batch_data')}
        WHERE row_num BETWEEN {{start}} AND {{end}}
        """
        
        # This is a template - actual implementation would need dataset size
        batch_queries.append(batch_template.format(start=1, end=batch_size))
        
        return batch_queries
    
    def create_view_query(self, 
                         base_query: str, 
                         view_name: str = "xgboost_predictions") -> str:
        """
        Create a SQL view from the prediction query
        
        Args:
            base_query: The base prediction SQL query
            view_name: Name for the view
            
        Returns:
            CREATE VIEW SQL statement
        """
        return f"CREATE VIEW {view_name} AS\n{base_query}"
    
    def validate_sql_syntax(self, sql: str) -> Dict[str, bool]:
        """
        Basic SQL syntax validation
        
        Args:
            sql: SQL query to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "has_select": bool(re.search(r'\bSELECT\b', sql, re.IGNORECASE)),
            "has_from": bool(re.search(r'\bFROM\b', sql, re.IGNORECASE)),
            "balanced_parentheses": sql.count('(') == sql.count(')'),
            "no_sql_injection": not bool(re.search(r';.*?DROP|;.*?DELETE|;.*?INSERT', sql, re.IGNORECASE))
        }
        
        results["valid"] = all(results.values())
        return results
    
    def add_filters(self, base_query: str, filters: Dict[str, str]) -> str:
        """
        Add WHERE clause filters to a query
        
        Args:
            base_query: Base SQL query
            filters: Dictionary of column -> condition mappings
            
        Returns:
            Modified SQL query with filters
        """
        if not filters:
            return base_query
        
        filter_conditions = []
        for column, condition in filters.items():
            filter_conditions.append(f"{column} {condition}")
        
        where_clause = " AND ".join(filter_conditions)
        
        if "WHERE" in base_query.upper():
            return base_query.replace("WHERE", f"WHERE {where_clause} AND", 1)
        else:
            # Add WHERE clause before ORDER BY if it exists
            if "ORDER BY" in base_query.upper():
                return base_query.replace("ORDER BY", f"WHERE {where_clause}\nORDER BY", 1)
            else:
                return f"{base_query}\nWHERE {where_clause}"