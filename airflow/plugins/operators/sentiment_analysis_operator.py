"""
Custom Airflow Operator for Sentiment Analysis using FinBERT
Analyzes sentiment of news articles and text data for trading signals.
"""

from typing import Dict, Any, List, Optional
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import pandas as pd
import numpy as np

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.exceptions import AirflowException

class SentimentAnalysisOperator(BaseOperator):
    """
    Custom operator for sentiment analysis using FinBERT or other transformer models.
    
    :param model_name: Name of the model to use (finbert, vader, textblob)
    :param batch_size: Batch size for processing
    :param confidence_threshold: Minimum confidence threshold for predictions
    """
    
    template_fields = ['model_name']
    
    @apply_defaults
    def __init__(
        self,
        model_name: str = 'finbert',
        batch_size: int = 32,
        confidence_threshold: float = 0.5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.tokenizer = None
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sentiment analysis."""
        logging.info(f"Starting sentiment analysis with {self.model_name}")
        
        try:
            # Get news data from previous task
            news_data = self._get_news_data_from_xcom(context)
            
            if not news_data or not news_data.get('news_data'):
                logging.warning("No news data found for sentiment analysis")
                return {'status': 'no_data', 'processed_articles': 0}
            
            # Initialize model
            self._initialize_model()
            
            # Process sentiment analysis
            sentiment_results = self._analyze_sentiment(news_data['news_data'])
            
            # Log results
            self._log_sentiment_results(sentiment_results)
            
            return sentiment_results
            
        except Exception as e:
            logging.error(f"Error during sentiment analysis: {e}")
            raise AirflowException(f"Sentiment analysis failed: {e}")
    
    def _get_news_data_from_xcom(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve news data from XCom."""
        task_instance = context['task_instance']
        return task_instance.xcom_pull(task_ids='alternative_data.ingest_news_data')
    
    def _initialize_model(self) -> None:
        """Initialize the sentiment analysis model."""
        try:
            if self.model_name == 'finbert':
                model_path = "ProsusAI/finbert"
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                
                # Create pipeline for easier inference
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if torch.cuda.is_available() else -1,
                    return_all_scores=True
                )
                
            elif self.model_name == 'vader':
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                
            elif self.model_name == 'textblob':
                from textblob import TextBlob
                # TextBlob doesn't need initialization
                pass
                
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
                
            logging.info(f"Successfully initialized {self.model_name} model")
            
        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            raise
    
    def _analyze_sentiment(self, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment of news articles."""
        results = {
            'status': 'success',
            'processed_articles': 0,
            'sentiment_data': [],
            'summary_stats': {}
        }
        
        try:
            for article in news_articles:
                # Extract text for analysis
                text = self._prepare_text(article)
                
                if not text:
                    continue
                
                # Analyze sentiment based on model
                sentiment_result = self._get_sentiment_score(text)
                
                # Add metadata
                sentiment_result.update({
                    'article_id': article.get('id'),
                    'title': article.get('title', ''),
                    'source': article.get('source', ''),
                    'timestamp': article.get('timestamp'),
                    'symbol': article.get('symbol'),
                    'original_text_length': len(text)
                })
                
                results['sentiment_data'].append(sentiment_result)
                results['processed_articles'] += 1
            
            # Calculate summary statistics
            results['summary_stats'] = self._calculate_summary_stats(results['sentiment_data'])
            
        except Exception as e:
            logging.error(f"Error during sentiment analysis: {e}")
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    def _prepare_text(self, article: Dict[str, Any]) -> str:
        """Prepare text for sentiment analysis."""
        # Combine title and content
        title = article.get('title', '')
        content = article.get('content', '')
        
        # Use title + first part of content (to avoid token limits)
        text = f"{title}. {content[:500]}" if content else title
        
        # Clean text
        text = text.strip()
        text = ' '.join(text.split())  # Remove extra whitespace
        
        return text
    
    def _get_sentiment_score(self, text: str) -> Dict[str, Any]:
        """Get sentiment score based on the selected model."""
        if self.model_name == 'finbert':
            return self._analyze_with_finbert(text)
        elif self.model_name == 'vader':
            return self._analyze_with_vader(text)
        elif self.model_name == 'textblob':
            return self._analyze_with_textblob(text)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _analyze_with_finbert(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using FinBERT."""
        try:
            # Truncate text to model's max length
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            # Get predictions
            predictions = self.sentiment_pipeline(text)
            
            # FinBERT returns: positive, negative, neutral
            scores = {pred['label'].lower(): pred['score'] for pred in predictions[0]}
            
            # Determine overall sentiment
            max_label = max(scores, key=scores.get)
            max_score = scores[max_label]
            
            # Convert to standardized format (-1 to 1)
            if max_label == 'positive':
                sentiment_score = max_score
            elif max_label == 'negative':
                sentiment_score = -max_score
            else:  # neutral
                sentiment_score = 0.0
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment_label': max_label,
                'confidence': max_score,
                'model': 'finbert',
                'raw_scores': scores
            }
            
        except Exception as e:
            logging.error(f"Error with FinBERT analysis: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'model': 'finbert',
                'error': str(e)
            }
    
    def _analyze_with_vader(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using VADER."""
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            
            # VADER returns compound score (-1 to 1)
            compound_score = scores['compound']
            
            # Determine label based on compound score
            if compound_score >= 0.05:
                label = 'positive'
            elif compound_score <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'sentiment_score': compound_score,
                'sentiment_label': label,
                'confidence': abs(compound_score),
                'model': 'vader',
                'raw_scores': scores
            }
            
        except Exception as e:
            logging.error(f"Error with VADER analysis: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'model': 'vader',
                'error': str(e)
            }
    
    def _analyze_with_textblob(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using TextBlob."""
        try:
            from textblob import TextBlob
            
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Determine label
            if polarity > 0.1:
                label = 'positive'
            elif polarity < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'sentiment_score': polarity,
                'sentiment_label': label,
                'confidence': abs(polarity),
                'model': 'textblob',
                'subjectivity': subjectivity
            }
            
        except Exception as e:
            logging.error(f"Error with TextBlob analysis: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'model': 'textblob',
                'error': str(e)
            }
    
    def _calculate_summary_stats(self, sentiment_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for sentiment analysis."""
        if not sentiment_data:
            return {}
        
        scores = [item['sentiment_score'] for item in sentiment_data if 'sentiment_score' in item]
        labels = [item['sentiment_label'] for item in sentiment_data if 'sentiment_label' in item]
        
        if not scores:
            return {}
        
        # Calculate statistics
        stats = {
            'total_articles': len(sentiment_data),
            'mean_sentiment': np.mean(scores),
            'median_sentiment': np.median(scores),
            'std_sentiment': np.std(scores),
            'min_sentiment': np.min(scores),
            'max_sentiment': np.max(scores),
            'positive_count': labels.count('positive'),
            'negative_count': labels.count('negative'),
            'neutral_count': labels.count('neutral')
        }
        
        # Calculate percentages
        total = stats['total_articles']
        stats['positive_percentage'] = (stats['positive_count'] / total) * 100
        stats['negative_percentage'] = (stats['negative_count'] / total) * 100
        stats['neutral_percentage'] = (stats['neutral_count'] / total) * 100
        
        return stats
    
    def _log_sentiment_results(self, results: Dict[str, Any]) -> None:
        """Log sentiment analysis results."""
        if results['status'] == 'success':
            stats = results.get('summary_stats', {})
            logging.info(f"✅ Sentiment analysis completed:")
            logging.info(f"  - Processed articles: {results['processed_articles']}")
            
            if stats:
                logging.info(f"  - Mean sentiment: {stats.get('mean_sentiment', 0):.3f}")
                logging.info(f"  - Positive: {stats.get('positive_percentage', 0):.1f}%")
                logging.info(f"  - Negative: {stats.get('negative_percentage', 0):.1f}%")
                logging.info(f"  - Neutral: {stats.get('neutral_percentage', 0):.1f}%")
        else:
            logging.error(f"❌ Sentiment analysis failed: {results.get('error', 'Unknown error')}")
