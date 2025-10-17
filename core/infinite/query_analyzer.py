"""Query analysis without AI for intelligent memory retrieval.

This module provides rule-based query analysis using keyword patterns,
entity extraction, temporal expression parsing, and code pattern detection.
No external AI calls are required.
"""

import re
from datetime import datetime, timedelta
from typing import Optional
from collections import Counter

from .models import QueryAnalysis, QueryIntent


class QueryAnalyzer:
    """Analyzes queries using rule-based methods without AI calls.
    
    Features:
    - Intent classification using keyword patterns
    - Entity extraction using patterns (optional spaCy NER)
    - Temporal expression parsing
    - Code pattern detection
    """
    
    # Intent classification patterns
    INTENT_PATTERNS = {
        QueryIntent.TEMPORAL: [
            r'\b(when|yesterday|today|tomorrow|last|ago|before|after|since)\b',
            r'\b(week|month|year|day|hour|minute)\b',
            r'\b\d{4}-\d{2}-\d{2}\b',  # Date format
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        ],
        QueryIntent.CODE: [
            r'\b(function|class|method|variable|import|def|async|await)\b',
            r'\b(\.py|\.js|\.ts|\.java|\.cpp|\.go|\.rs)\b',
            r'\b(code|implementation|bug|error|exception)\b',
            r'[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)',  # Function calls
        ],
        QueryIntent.PREFERENCE: [
            r'\b(like|prefer|favorite|love|hate|dislike|want|need)\b',
            r'\b(always|never|usually|often|sometimes)\b',
            r'\b(better|worse|best|worst)\b',
        ],
        QueryIntent.FACTUAL: [
            r'\b(what|who|where|which|how many|how much)\b',
            r'\b(is|are|was|were|has|have|had)\b',
            r'\b(define|explain|describe|tell me about)\b',
        ],
    }
    
    # Temporal expression patterns
    TEMPORAL_PATTERNS = {
        'relative_days': r'\b(yesterday|today|tomorrow)\b',
        'relative_time': r'\b(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago\b',
        'last_period': r'\blast\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        'date_format': r'\b(\d{4})-(\d{2})-(\d{2})\b',
        'month_year': r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b',
    }
    
    # Code pattern detection
    CODE_PATTERNS = {
        'function_call': r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',
        'class_name': r'\b([A-Z][a-zA-Z0-9_]*)\s+class\b',  # "UserAuth class"
        'class_def': r'\bclass\s+([A-Z][a-zA-Z0-9_]*)\b',  # "class UserAuth"
        'function_def': r'\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
        'file_path': r'([a-zA-Z_][a-zA-Z0-9_/\\]*\.(py|js|ts|java|cpp|go|rs|c|h))',
        'import_statement': r'\bimport\s+([a-zA-Z_][a-zA-Z0-9_.]*)\b',
    }
    
    # Entity patterns (simple rule-based)
    ENTITY_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'url': r'https?://[^\s]+',
        'file_path': r'([a-zA-Z_][a-zA-Z0-9_/\\]*\.[a-zA-Z0-9]+)',
        'number': r'\b\d+(?:\.\d+)?\b',
    }
    
    def __init__(self, use_spacy: bool = False):
        """Initialize query analyzer.
        
        Args:
            use_spacy: Whether to use spaCy for enhanced entity extraction
        """
        self.use_spacy = use_spacy
        self.nlp = None
        
        if use_spacy:
            try:
                import spacy
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    # Model not installed, fall back to pattern-based
                    self.use_spacy = False
            except ImportError:
                # spaCy not installed, fall back to pattern-based
                self.use_spacy = False
    
    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze a query and extract structured information.
        
        Args:
            query: The query string to analyze
            
        Returns:
            QueryAnalysis object with extracted information
        """
        query_lower = query.lower()
        
        # Classify intent
        intent, confidence = self._classify_intent(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Parse temporal expressions
        temporal_expressions = self._parse_temporal_expressions(query_lower)
        
        # Detect code patterns
        code_patterns = self._detect_code_patterns(query)
        
        # Extract keywords
        keywords = self._extract_keywords(query_lower)
        
        return QueryAnalysis(
            intent=intent,
            entities=entities,
            temporal_expressions=temporal_expressions,
            code_patterns=code_patterns,
            keywords=keywords,
            confidence=confidence,
        )
    
    def _classify_intent(self, query: str) -> tuple[QueryIntent, float]:
        """Classify query intent using keyword patterns.
        
        Args:
            query: Lowercase query string
            
        Returns:
            Tuple of (intent, confidence)
        """
        scores = Counter()
        
        # Score each intent based on pattern matches
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                scores[intent] += len(matches)
        
        if not scores or sum(scores.values()) == 0:
            # Default to conversational if no patterns match
            return QueryIntent.CONVERSATIONAL, 0.5
        
        # Get the highest scoring intent
        top_intent, top_score = scores.most_common(1)[0]
        total_score = sum(scores.values())
        confidence = top_score / total_score if total_score > 0 else 0.5
        
        # If multiple intents have similar scores, classify as MIXED
        if len(scores) > 1 and top_score > 0:
            second_score = scores.most_common(2)[1][1]
            if second_score / top_score > 0.7:  # Within 70% of top score
                return QueryIntent.MIXED, confidence * 0.8
        
        return top_intent, min(confidence, 1.0)
    
    def _extract_entities(self, query: str) -> list[tuple[str, str]]:
        """Extract entities from query.
        
        Args:
            query: Query string
            
        Returns:
            List of (entity_type, entity_value) tuples
        """
        entities = []
        
        if self.use_spacy and self.nlp:
            # Use spaCy NER
            doc = self.nlp(query)
            for ent in doc.ents:
                entities.append((ent.label_, ent.text))
        else:
            # Use pattern-based extraction
            for entity_type, pattern in self.ENTITY_PATTERNS.items():
                matches = re.findall(pattern, query)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    entities.append((entity_type, match))
        
        return entities
    
    def _parse_temporal_expressions(self, query: str) -> list[tuple[str, float]]:
        """Parse temporal expressions and convert to timestamps.
        
        Args:
            query: Lowercase query string
            
        Returns:
            List of (expression, timestamp) tuples
        """
        expressions = []
        now = datetime.now()
        
        # Relative days
        if match := re.search(self.TEMPORAL_PATTERNS['relative_days'], query):
            expression = match.group(0)
            if expression == 'yesterday':
                timestamp = (now - timedelta(days=1)).timestamp()
            elif expression == 'today':
                timestamp = now.timestamp()
            elif expression == 'tomorrow':
                timestamp = (now + timedelta(days=1)).timestamp()
            expressions.append((expression, timestamp))
        
        # Relative time (e.g., "2 weeks ago")
        for match in re.finditer(self.TEMPORAL_PATTERNS['relative_time'], query):
            expression = match.group(0)
            amount = int(match.group(1))
            unit = match.group(2)
            
            if unit.startswith('second'):
                delta = timedelta(seconds=amount)
            elif unit.startswith('minute'):
                delta = timedelta(minutes=amount)
            elif unit.startswith('hour'):
                delta = timedelta(hours=amount)
            elif unit.startswith('day'):
                delta = timedelta(days=amount)
            elif unit.startswith('week'):
                delta = timedelta(weeks=amount)
            elif unit.startswith('month'):
                delta = timedelta(days=amount * 30)  # Approximate
            elif unit.startswith('year'):
                delta = timedelta(days=amount * 365)  # Approximate
            else:
                continue
            
            timestamp = (now - delta).timestamp()
            expressions.append((expression, timestamp))
        
        # Last period (e.g., "last week")
        if match := re.search(self.TEMPORAL_PATTERNS['last_period'], query):
            expression = match.group(0)
            period = match.group(1)
            timestamp = None
            
            if period == 'week':
                timestamp = (now - timedelta(weeks=1)).timestamp()
            elif period == 'month':
                timestamp = (now - timedelta(days=30)).timestamp()
            elif period == 'year':
                timestamp = (now - timedelta(days=365)).timestamp()
            elif period in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
                # Find last occurrence of this weekday
                days_map = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                           'friday': 4, 'saturday': 5, 'sunday': 6}
                target_day = days_map[period]
                current_day = now.weekday()
                days_ago = (current_day - target_day) % 7
                if days_ago == 0:
                    days_ago = 7  # Last week's occurrence
                timestamp = (now - timedelta(days=days_ago)).timestamp()
            
            if timestamp is not None:
                expressions.append((expression, timestamp))
        
        # Date format (YYYY-MM-DD)
        for match in re.finditer(self.TEMPORAL_PATTERNS['date_format'], query):
            expression = match.group(0)
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            try:
                dt = datetime(year, month, day)
                expressions.append((expression, dt.timestamp()))
            except ValueError:
                # Invalid date
                continue
        
        # Month and year
        for match in re.finditer(self.TEMPORAL_PATTERNS['month_year'], query):
            expression = match.group(0)
            month_name = match.group(1)
            year = int(match.group(2))
            
            months = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12
            }
            month = months.get(month_name)
            if month:
                try:
                    dt = datetime(year, month, 1)
                    expressions.append((expression, dt.timestamp()))
                except ValueError:
                    continue
        
        return expressions
    
    def _detect_code_patterns(self, query: str) -> list[str]:
        """Detect code-related patterns in query.
        
        Args:
            query: Query string
            
        Returns:
            List of detected code patterns
        """
        patterns = []
        
        for pattern_type, pattern in self.CODE_PATTERNS.items():
            matches = re.findall(pattern, query)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if match and match not in patterns:
                    patterns.append(match)
        
        return patterns
    
    def _extract_keywords(self, query: str) -> list[str]:
        """Extract important keywords from query.
        
        Args:
            query: Lowercase query string
            
        Returns:
            List of keywords
        """
        # Remove common stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'what', 'when', 'where', 'who', 'how',
            'but', 'or', 'if', 'then', 'this', 'these', 'those', 'there'
        }
        
        # Tokenize and filter
        words = re.findall(r'\b[a-z]+\b', query)
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:10]  # Limit to top 10 keywords
