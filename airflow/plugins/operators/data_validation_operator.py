"""
Custom Airflow Operator for Data Validation using Pandera
Validates incoming data against predefined schemas to ensure data quality.
"""

from typing import Dict, Any, Optional
import logging
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.exceptions import AirflowException

class DataValidationOperator(BaseOperator):
    """
    Custom operator to validate data using Pandera schemas.
    
    :param data_source: Source of the data (binance, polygon, alternative)
    :param validation_schema: Name of the validation schema to use
    :param fail_on_error: Whether to fail the task if validation fails
    """
    
    template_fields = ['data_source', 'validation_schema']
    
    @apply_defaults
    def __init__(
        self,
        data_source: str,
        validation_schema: str,
        fail_on_error: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_source = data_source
        self.validation_schema = validation_schema
        self.fail_on_error = fail_on_error
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data validation."""
        logging.info(f"Starting data validation for {self.data_source}")
        
        try:
            # Get data from previous task
            data = self._get_data_from_xcom(context)
            
            # Get validation schema
            schema = self._get_validation_schema()
            
            # Perform validation
            validation_results = self._validate_data(data, schema)
            
            # Log results
            self._log_validation_results(validation_results)
            
            # Handle validation failures
            if validation_results['has_errors'] and self.fail_on_error:
                raise AirflowException(f"Data validation failed: {validation_results['errors']}")
            
            return validation_results
            
        except Exception as e:
            logging.error(f"Error during data validation: {e}")
            if self.fail_on_error:
                raise
            return {'status': 'error', 'error': str(e)}
    
    def _get_data_from_xcom(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve data from XCom based on data source."""
        task_instance = context['task_instance']
        
        if self.data_source == 'binance':
            return task_instance.xcom_pull(task_ids='binance_ingestion.ingest_binance_data')
        elif self.data_source == 'polygon':
            return task_instance.xcom_pull(task_ids='polygon_ingestion.ingest_polygon_data')
        elif self.data_source == 'alternative':
            news_data = task_instance.xcom_pull(task_ids='alternative_data.ingest_news_data')
            onchain_data = task_instance.xcom_pull(task_ids='alternative_data.ingest_onchain_data')
            return {'news_data': news_data, 'onchain_data': onchain_data}
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")
    
    def _get_validation_schema(self) -> Dict[str, DataFrameSchema]:
        """Get validation schemas based on schema name."""
        schemas = {
            'market_data_schema': self._get_market_data_schema(),
            'alternative_data_schema': self._get_alternative_data_schema(),
        }
        
        if self.validation_schema not in schemas:
            raise ValueError(f"Unknown validation schema: {self.validation_schema}")
        
        return schemas[self.validation_schema]
    
    def _get_market_data_schema(self) -> DataFrameSchema:
        """Define market data validation schema."""
        return DataFrameSchema({
            'symbol': Column(str, checks=[
                Check.str_matches(r'^[A-Z0-9]+$'),  # Alphanumeric uppercase
                Check.str_length(min_val=3, max_val=20)
            ]),
            'timestamp': Column(pd.Timestamp, checks=[
                Check.greater_than_or_equal_to(pd.Timestamp.now() - pd.Timedelta(days=1))
            ]),
            'open': Column(float, checks=[
                Check.greater_than(0),
                Check.less_than(1e10)  # Reasonable upper bound
            ]),
            'high': Column(float, checks=[
                Check.greater_than(0),
                Check.less_than(1e10)
            ]),
            'low': Column(float, checks=[
                Check.greater_than(0),
                Check.less_than(1e10)
            ]),
            'close': Column(float, checks=[
                Check.greater_than(0),
                Check.less_than(1e10)
            ]),
            'volume': Column(float, checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than(1e15)
            ]),
            'exchange': Column(str, checks=[
                Check.isin(['binance', 'polygon', 'coinbase', 'kraken'])
            ])
        }, checks=[
            # Cross-column validation
            Check(lambda df: df['high'] >= df['low'], error='High must be >= Low'),
            Check(lambda df: df['high'] >= df['open'], error='High must be >= Open'),
            Check(lambda df: df['high'] >= df['close'], error='High must be >= Close'),
            Check(lambda df: df['low'] <= df['open'], error='Low must be <= Open'),
            Check(lambda df: df['low'] <= df['close'], error='Low must be <= Close'),
        ])
    
    def _get_alternative_data_schema(self) -> Dict[str, DataFrameSchema]:
        """Define alternative data validation schemas."""
        news_schema = DataFrameSchema({
            'title': Column(str, checks=[
                Check.str_length(min_val=10, max_val=500)
            ]),
            'content': Column(str, checks=[
                Check.str_length(min_val=50, max_val=10000)
            ], nullable=True),
            'source': Column(str, checks=[
                Check.str_length(min_val=3, max_val=100)
            ]),
            'timestamp': Column(pd.Timestamp, checks=[
                Check.greater_than_or_equal_to(pd.Timestamp.now() - pd.Timedelta(days=7))
            ]),
            'sentiment_score': Column(float, checks=[
                Check.in_range(-1.0, 1.0)
            ], nullable=True),
            'sentiment_label': Column(str, checks=[
                Check.isin(['positive', 'negative', 'neutral'])
            ], nullable=True)
        })
        
        onchain_schema = DataFrameSchema({
            'symbol': Column(str, checks=[
                Check.str_matches(r'^[A-Z0-9]+$')
            ]),
            'metric_name': Column(str, checks=[
                Check.isin(['active_addresses', 'transaction_count', 'hash_rate', 
                           'network_value', 'market_cap', 'volume_24h'])
            ]),
            'metric_value': Column(float, checks=[
                Check.greater_than_or_equal_to(0)
            ]),
            'timestamp': Column(pd.Timestamp, checks=[
                Check.greater_than_or_equal_to(pd.Timestamp.now() - pd.Timedelta(hours=24))
            ]),
            'source': Column(str, checks=[
                Check.str_length(min_val=3, max_val=50)
            ])
        })
        
        return {
            'news': news_schema,
            'onchain': onchain_schema
        }
    
    def _validate_data(self, data: Dict[str, Any], schema: Any) -> Dict[str, Any]:
        """Perform data validation using Pandera."""
        validation_results = {
            'status': 'success',
            'has_errors': False,
            'errors': [],
            'warnings': [],
            'validated_records': 0,
            'failed_records': 0
        }
        
        try:
            if self.validation_schema == 'market_data_schema':
                # Validate market data
                if 'market_data' in data and data['market_data']:
                    df = pd.DataFrame(data['market_data'])
                    validated_df = schema.validate(df, lazy=True)
                    validation_results['validated_records'] = len(validated_df)
                    
            elif self.validation_schema == 'alternative_data_schema':
                # Validate news data
                if data.get('news_data') and data['news_data'].get('news_data'):
                    news_df = pd.DataFrame(data['news_data']['news_data'])
                    validated_news = schema['news'].validate(news_df, lazy=True)
                    validation_results['validated_records'] += len(validated_news)
                
                # Validate on-chain data
                if data.get('onchain_data') and data['onchain_data'].get('onchain_data'):
                    onchain_df = pd.DataFrame(data['onchain_data']['onchain_data'])
                    validated_onchain = schema['onchain'].validate(onchain_df, lazy=True)
                    validation_results['validated_records'] += len(validated_onchain)
                    
        except pa.errors.SchemaErrors as e:
            validation_results['has_errors'] = True
            validation_results['status'] = 'failed'
            validation_results['failed_records'] = len(e.failure_cases)
            
            # Extract error details
            for error in e.failure_cases.itertuples():
                validation_results['errors'].append({
                    'column': error.column,
                    'check': error.check,
                    'error_count': error.failure_case.shape[0] if hasattr(error.failure_case, 'shape') else 1
                })
                
        except Exception as e:
            validation_results['has_errors'] = True
            validation_results['status'] = 'error'
            validation_results['errors'].append(str(e))
        
        return validation_results
    
    def _log_validation_results(self, results: Dict[str, Any]) -> None:
        """Log validation results."""
        if results['status'] == 'success':
            logging.info(f"✅ Data validation successful: {results['validated_records']} records validated")
        else:
            logging.warning(f"⚠️ Data validation issues found:")
            logging.warning(f"  - Status: {results['status']}")
            logging.warning(f"  - Validated records: {results['validated_records']}")
            logging.warning(f"  - Failed records: {results['failed_records']}")
            
            if results['errors']:
                logging.error("Validation errors:")
                for error in results['errors']:
                    logging.error(f"  - {error}")
