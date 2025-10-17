"""Tests for query analyzer."""

import pytest
from datetime import datetime, timedelta

from core.infinite.query_analyzer import QueryAnalyzer
from core.infinite.models import QueryIntent


class TestQueryAnalyzer:
    """Test suite for QueryAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a query analyzer instance."""
        return QueryAnalyzer(use_spacy=False)
    
    @pytest.fixture
    def analyzer_with_spacy(self):
        """Create a query analyzer with spaCy if available."""
        return QueryAnalyzer(use_spacy=True)
    
    # Intent Classification Tests
    
    def test_temporal_intent(self, analyzer):
        """Test temporal intent classification."""
        queries = [
            "What did I do yesterday?",
            "Show me conversations from last week",
            "What happened 2 months ago?",
            "Find messages from January 2024",
        ]
        
        for query in queries:
            result = analyzer.analyze(query)
            assert result.intent in [QueryIntent.TEMPORAL, QueryIntent.MIXED]
            assert result.confidence > 0.0
    
    def test_code_intent(self, analyzer):
        """Test code intent classification."""
        queries = [
            "Show me the login function",
            "Find the UserAuth class definition",
            "What changed in auth.py?",
            "Show me the implementation of authenticate()",
        ]
        
        for query in queries:
            result = analyzer.analyze(query)
            assert result.intent in [QueryIntent.CODE, QueryIntent.MIXED]
            assert result.confidence > 0.0
    
    def test_preference_intent(self, analyzer):
        """Test preference intent classification."""
        queries = [
            "I like dark mode",
            "I prefer Python over JavaScript",
            "I always use tabs instead of spaces",
            "I hate writing documentation",
        ]
        
        for query in queries:
            result = analyzer.analyze(query)
            assert result.intent in [QueryIntent.PREFERENCE, QueryIntent.MIXED]
            assert result.confidence > 0.0
    
    def test_factual_intent(self, analyzer):
        """Test factual intent classification."""
        queries = [
            "What is the capital of France?",
            "Who wrote this code?",
            "Where is the config file?",
            "How many users are there?",
        ]
        
        for query in queries:
            result = analyzer.analyze(query)
            assert result.intent in [QueryIntent.FACTUAL, QueryIntent.MIXED]
            assert result.confidence > 0.0
    
    def test_conversational_intent(self, analyzer):
        """Test conversational intent classification."""
        queries = [
            "Hello there",
            "Thanks for the help",
            "Can you assist me?",
        ]
        
        for query in queries:
            result = analyzer.analyze(query)
            # These should default to conversational
            assert result.intent == QueryIntent.CONVERSATIONAL
    
    def test_mixed_intent(self, analyzer):
        """Test mixed intent classification."""
        query = "What code changes did I make last week?"
        result = analyzer.analyze(query)
        
        # Should detect both temporal and code patterns
        assert result.intent == QueryIntent.MIXED or result.intent in [QueryIntent.TEMPORAL, QueryIntent.CODE]
        assert len(result.temporal_expressions) > 0
    
    # Entity Extraction Tests
    
    def test_email_entity_extraction(self, analyzer):
        """Test email entity extraction."""
        query = "Send this to user@example.com"
        result = analyzer.analyze(query)
        
        emails = [entity for entity_type, entity in result.entities if entity_type == 'email']
        assert len(emails) > 0
        assert 'user@example.com' in emails
    
    def test_url_entity_extraction(self, analyzer):
        """Test URL entity extraction."""
        query = "Check https://github.com/user/repo"
        result = analyzer.analyze(query)
        
        urls = [entity for entity_type, entity in result.entities if entity_type == 'url']
        assert len(urls) > 0
        assert any('github.com' in url for url in urls)
    
    def test_file_path_entity_extraction(self, analyzer):
        """Test file path entity extraction."""
        query = "Look at src/auth.py"
        result = analyzer.analyze(query)
        
        files = [entity for entity_type, entity in result.entities if entity_type == 'file_path']
        assert len(files) > 0
        assert any('auth.py' in file for file in files)
    
    def test_number_entity_extraction(self, analyzer):
        """Test number entity extraction."""
        query = "There are 42 users and 3.14 is pi"
        result = analyzer.analyze(query)
        
        numbers = [entity for entity_type, entity in result.entities if entity_type == 'number']
        assert len(numbers) >= 2
        assert '42' in numbers
        assert '3.14' in numbers
    
    # Temporal Expression Parsing Tests
    
    def test_relative_days_parsing(self, analyzer):
        """Test parsing of relative day expressions."""
        queries = ["yesterday", "today", "tomorrow"]
        
        for query in queries:
            result = analyzer.analyze(query)
            assert len(result.temporal_expressions) > 0
            expression, timestamp = result.temporal_expressions[0]
            assert expression == query
            assert isinstance(timestamp, float)
    
    def test_relative_time_parsing(self, analyzer):
        """Test parsing of relative time expressions."""
        query = "2 weeks ago"
        result = analyzer.analyze(query)
        
        assert len(result.temporal_expressions) > 0
        expression, timestamp = result.temporal_expressions[0]
        assert "2 week" in expression
        
        # Verify timestamp is approximately 2 weeks ago
        now = datetime.now().timestamp()
        expected = (datetime.now() - timedelta(weeks=2)).timestamp()
        assert abs(timestamp - expected) < 86400  # Within 1 day tolerance
    
    def test_last_period_parsing(self, analyzer):
        """Test parsing of 'last X' expressions."""
        queries = ["last week", "last month", "last year"]
        
        for query in queries:
            result = analyzer.analyze(query)
            assert len(result.temporal_expressions) > 0
            expression, timestamp = result.temporal_expressions[0]
            assert "last" in expression
            assert timestamp < datetime.now().timestamp()
    
    def test_date_format_parsing(self, analyzer):
        """Test parsing of date format (YYYY-MM-DD)."""
        query = "Show me data from 2024-01-15"
        result = analyzer.analyze(query)
        
        assert len(result.temporal_expressions) > 0
        expression, timestamp = result.temporal_expressions[0]
        assert expression == "2024-01-15"
        
        # Verify timestamp matches the date
        expected = datetime(2024, 1, 15).timestamp()
        assert abs(timestamp - expected) < 86400  # Within 1 day
    
    def test_month_year_parsing(self, analyzer):
        """Test parsing of month and year."""
        query = "What happened in january 2024?"
        result = analyzer.analyze(query)
        
        assert len(result.temporal_expressions) > 0
        expression, timestamp = result.temporal_expressions[0]
        assert "january 2024" in expression
        
        # Verify timestamp is in January 2024
        dt = datetime.fromtimestamp(timestamp)
        assert dt.year == 2024
        assert dt.month == 1
    
    def test_multiple_temporal_expressions(self, analyzer):
        """Test parsing multiple temporal expressions."""
        query = "Compare yesterday with 2024-01-15"
        result = analyzer.analyze(query)
        
        assert len(result.temporal_expressions) >= 2
    
    # Code Pattern Detection Tests
    
    def test_function_call_detection(self, analyzer):
        """Test detection of function calls."""
        query = "Show me where authenticate() is called"
        result = analyzer.analyze(query)
        
        assert len(result.code_patterns) > 0
        assert 'authenticate' in result.code_patterns
    
    def test_class_name_detection(self, analyzer):
        """Test detection of class names."""
        query = "Find the UserAuth class definition"
        result = analyzer.analyze(query)
        
        assert len(result.code_patterns) > 0
        assert 'UserAuth' in result.code_patterns
    
    def test_function_def_detection(self, analyzer):
        """Test detection of function definitions."""
        query = "Show me def login implementation"
        result = analyzer.analyze(query)
        
        assert len(result.code_patterns) > 0
        assert 'login' in result.code_patterns
    
    def test_file_path_detection(self, analyzer):
        """Test detection of file paths."""
        query = "What changed in src/auth.py?"
        result = analyzer.analyze(query)
        
        assert len(result.code_patterns) > 0
        assert any('auth.py' in pattern for pattern in result.code_patterns)
    
    def test_import_statement_detection(self, analyzer):
        """Test detection of import statements."""
        query = "Where is import datetime used?"
        result = analyzer.analyze(query)
        
        assert len(result.code_patterns) > 0
        assert 'datetime' in result.code_patterns
    
    # Keyword Extraction Tests
    
    def test_keyword_extraction(self, analyzer):
        """Test keyword extraction."""
        query = "Show me the authentication implementation for user login"
        result = analyzer.analyze(query)
        
        assert len(result.keywords) > 0
        # Should extract meaningful words, not stop words
        assert 'authentication' in result.keywords or 'login' in result.keywords
        assert 'the' not in result.keywords  # Stop word
        assert 'for' not in result.keywords  # Stop word
    
    def test_keyword_deduplication(self, analyzer):
        """Test that keywords are deduplicated."""
        query = "login login login authentication"
        result = analyzer.analyze(query)
        
        # Should only have unique keywords
        assert len(result.keywords) == len(set(result.keywords))
        assert result.keywords.count('login') == 1
    
    def test_keyword_limit(self, analyzer):
        """Test that keywords are limited to top 10."""
        query = " ".join([f"word{i}" for i in range(20)])
        result = analyzer.analyze(query)
        
        assert len(result.keywords) <= 10
    
    # Integration Tests
    
    def test_complex_query_analysis(self, analyzer):
        """Test analysis of a complex query."""
        query = "What code changes did I make to auth.py last week in the login() function?"
        result = analyzer.analyze(query)
        
        # Should detect multiple aspects
        assert result.intent in [QueryIntent.CODE, QueryIntent.MIXED, QueryIntent.TEMPORAL]
        assert len(result.temporal_expressions) > 0  # "last week"
        assert len(result.code_patterns) > 0  # "auth.py", "login"
        assert len(result.keywords) > 0
    
    def test_empty_query(self, analyzer):
        """Test analysis of empty query."""
        result = analyzer.analyze("")
        
        assert result.intent == QueryIntent.CONVERSATIONAL
        assert len(result.entities) == 0
        assert len(result.temporal_expressions) == 0
        assert len(result.code_patterns) == 0
    
    def test_query_with_special_characters(self, analyzer):
        """Test query with special characters."""
        query = "What's the @deprecated function in utils.py?"
        result = analyzer.analyze(query)
        
        # Should still extract meaningful information
        assert result.intent in [QueryIntent.CODE, QueryIntent.FACTUAL, QueryIntent.MIXED]
        assert len(result.code_patterns) > 0
    
    def test_case_insensitive_analysis(self, analyzer):
        """Test that analysis is case-insensitive for intent."""
        query1 = "show me yesterday's code"
        query2 = "SHOW ME YESTERDAY'S CODE"
        
        result1 = analyzer.analyze(query1)
        result2 = analyzer.analyze(query2)
        
        # Should have same intent
        assert result1.intent == result2.intent
    
    def test_confidence_scores(self, analyzer):
        """Test that confidence scores are reasonable."""
        queries = [
            "What did I do yesterday?",
            "Show me the login function",
            "I prefer dark mode",
        ]
        
        for query in queries:
            result = analyzer.analyze(query)
            assert 0.0 <= result.confidence <= 1.0
    
    # spaCy Integration Tests (if available)
    
    def test_spacy_entity_extraction(self, analyzer_with_spacy):
        """Test spaCy entity extraction if available."""
        if not analyzer_with_spacy.use_spacy:
            pytest.skip("spaCy not available")
        
        query = "John Smith works at Microsoft in Seattle"
        result = analyzer_with_spacy.analyze(query)
        
        # spaCy should extract person, organization, and location
        entity_types = [entity_type for entity_type, _ in result.entities]
        assert len(entity_types) > 0
    
    def test_fallback_without_spacy(self, analyzer):
        """Test that analyzer works without spaCy."""
        # This analyzer is initialized without spaCy
        assert not analyzer.use_spacy
        
        query = "Send email to user@example.com"
        result = analyzer.analyze(query)
        
        # Should still extract entities using patterns
        assert len(result.entities) > 0


class TestQueryAnalyzerEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a query analyzer instance."""
        return QueryAnalyzer(use_spacy=False)
    
    def test_very_long_query(self, analyzer):
        """Test with very long query."""
        query = "word " * 1000
        result = analyzer.analyze(query)
        
        # Should complete without error
        assert result is not None
        assert len(result.keywords) <= 10  # Should be limited
    
    def test_unicode_query(self, analyzer):
        """Test with unicode characters."""
        query = "What about café and naïve?"
        result = analyzer.analyze(query)
        
        # Should handle unicode gracefully
        assert result is not None
    
    def test_query_with_only_stop_words(self, analyzer):
        """Test query with only stop words."""
        query = "the and or but"
        result = analyzer.analyze(query)
        
        # Should return empty keywords
        assert len(result.keywords) == 0
    
    def test_invalid_date_format(self, analyzer):
        """Test with invalid date format."""
        query = "Show me data from 2024-13-45"  # Invalid month and day
        result = analyzer.analyze(query)
        
        # Should not crash, but may not extract temporal expression
        assert result is not None
    
    def test_malformed_code_patterns(self, analyzer):
        """Test with malformed code patterns."""
        query = "Show me function(((("
        result = analyzer.analyze(query)
        
        # Should handle gracefully
        assert result is not None


class TestQueryAnalyzerAccuracy:
    """Test accuracy of query analysis for rule-based routing."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a query analyzer instance."""
        return QueryAnalyzer(use_spacy=False)
    
    def test_intent_classification_accuracy_temporal(self, analyzer):
        """Test accuracy of temporal intent classification."""
        temporal_queries = [
            "What happened yesterday?",
            "Show me files from last month",
            "Find conversations from 2024-01-15",
            "What did I do 3 hours ago?",
            "Display data from january 2024",
        ]
        
        correct = 0
        for query in temporal_queries:
            result = analyzer.analyze(query)
            if result.intent in [QueryIntent.TEMPORAL, QueryIntent.MIXED]:
                correct += 1
        
        accuracy = correct / len(temporal_queries)
        assert accuracy >= 0.8, f"Temporal intent accuracy {accuracy} below 80%"
    
    def test_intent_classification_accuracy_code(self, analyzer):
        """Test accuracy of code intent classification."""
        code_queries = [
            "Show me the authenticate function",
            "Find UserAuth class",
            "What's in auth.py?",
            "Where is login() defined?",
            "Show me import statements",
        ]
        
        correct = 0
        for query in code_queries:
            result = analyzer.analyze(query)
            if result.intent in [QueryIntent.CODE, QueryIntent.MIXED]:
                correct += 1
        
        accuracy = correct / len(code_queries)
        assert accuracy >= 0.8, f"Code intent accuracy {accuracy} below 80%"
    
    def test_intent_classification_accuracy_preference(self, analyzer):
        """Test accuracy of preference intent classification."""
        preference_queries = [
            "I like dark mode",
            "I prefer Python",
            "I always use tabs",
            "I hate writing tests",
            "I love functional programming",
        ]
        
        correct = 0
        for query in preference_queries:
            result = analyzer.analyze(query)
            if result.intent in [QueryIntent.PREFERENCE, QueryIntent.MIXED]:
                correct += 1
        
        accuracy = correct / len(preference_queries)
        assert accuracy >= 0.8, f"Preference intent accuracy {accuracy} below 80%"
    
    def test_entity_extraction_accuracy(self, analyzer):
        """Test accuracy of entity extraction."""
        test_cases = [
            ("Contact user@example.com", "email", "user@example.com"),
            ("Check https://github.com/repo", "url", "github.com"),
            ("Look at src/auth.py", "file_path", "auth.py"),
            ("There are 42 users", "number", "42"),
        ]
        
        correct = 0
        for query, expected_type, expected_value in test_cases:
            result = analyzer.analyze(query)
            entities = [entity for entity_type, entity in result.entities if entity_type == expected_type]
            if any(expected_value in entity for entity in entities):
                correct += 1
        
        accuracy = correct / len(test_cases)
        assert accuracy >= 0.75, f"Entity extraction accuracy {accuracy} below 75%"
    
    def test_temporal_parsing_accuracy(self, analyzer):
        """Test accuracy of temporal expression parsing."""
        now = datetime.now()
        
        test_cases = [
            ("yesterday", 1),  # 1 day ago
            ("2 weeks ago", 14),  # 14 days ago
            ("last month", 30),  # ~30 days ago
        ]
        
        correct = 0
        for query, expected_days_ago in test_cases:
            result = analyzer.analyze(query)
            if len(result.temporal_expressions) > 0:
                _, timestamp = result.temporal_expressions[0]
                days_diff = (now.timestamp() - timestamp) / 86400
                # Allow 10% tolerance
                if abs(days_diff - expected_days_ago) / expected_days_ago < 0.1:
                    correct += 1
        
        accuracy = correct / len(test_cases)
        assert accuracy >= 0.66, f"Temporal parsing accuracy {accuracy} below 66%"
    
    def test_code_pattern_detection_accuracy(self, analyzer):
        """Test accuracy of code pattern detection."""
        test_cases = [
            ("Show me authenticate() function", "authenticate"),
            ("Find UserAuth class", "UserAuth"),
            ("What's in auth.py?", "auth.py"),
            ("Where is import datetime?", "datetime"),
        ]
        
        correct = 0
        for query, expected_pattern in test_cases:
            result = analyzer.analyze(query)
            if any(expected_pattern in pattern for pattern in result.code_patterns):
                correct += 1
        
        accuracy = correct / len(test_cases)
        assert accuracy >= 0.75, f"Code pattern detection accuracy {accuracy} below 75%"
    
    def test_keyword_extraction_relevance(self, analyzer):
        """Test that extracted keywords are relevant."""
        query = "Show me the authentication implementation for user login system"
        result = analyzer.analyze(query)
        
        # Should extract meaningful keywords
        relevant_keywords = ['authentication', 'implementation', 'user', 'login', 'system']
        extracted_relevant = [kw for kw in result.keywords if kw in relevant_keywords]
        
        # At least 60% of relevant keywords should be extracted
        assert len(extracted_relevant) >= len(relevant_keywords) * 0.6
    
    def test_mixed_intent_detection_accuracy(self, analyzer):
        """Test detection of queries with mixed intents."""
        mixed_queries = [
            "What code changes did I make yesterday?",  # CODE + TEMPORAL
            "Show me my Python preferences from last week",  # PREFERENCE + TEMPORAL
            "What functions did I write in auth.py last month?",  # CODE + TEMPORAL
        ]
        
        for query in mixed_queries:
            result = analyzer.analyze(query)
            # Should detect multiple aspects
            has_temporal = len(result.temporal_expressions) > 0
            has_code = len(result.code_patterns) > 0 or result.intent == QueryIntent.CODE
            
            # At least one of the mixed aspects should be detected
            assert has_temporal or has_code or result.intent == QueryIntent.MIXED
    
    def test_confidence_score_correlation(self, analyzer):
        """Test that confidence scores are reasonable for different query types."""
        test_cases = [
            ("What happened yesterday afternoon?", 0.3),  # Mixed intent gets penalty
            ("Show me the login function", 0.5),  # Clear code intent
            ("I prefer dark mode", 0.5),  # Clear preference intent
        ]
        
        for query, min_confidence in test_cases:
            result = analyzer.analyze(query)
            # Confidence should be above minimum threshold
            assert result.confidence >= min_confidence, \
                f"Query '{query}' has confidence {result.confidence} below {min_confidence}"
    
    def test_rule_based_routing_no_ai_calls(self, analyzer):
        """Test that analysis works without any AI calls."""
        # This is a critical requirement - no AI should be needed
        queries = [
            "What happened yesterday?",
            "Show me the login function",
            "I prefer dark mode",
            "What is Python?",
        ]
        
        for query in queries:
            # Should complete successfully without AI
            result = analyzer.analyze(query)
            assert result is not None
            assert result.intent is not None
            assert isinstance(result.confidence, float)
    
    def test_various_query_types_coverage(self, analyzer):
        """Test analysis across various query types."""
        query_types = {
            "temporal": "What happened last week?",
            "code": "Show me the UserAuth class",
            "preference": "I like Python",
            "factual": "What is the capital of France?",
            "conversational": "Hello there",
            "mixed": "What code did I write yesterday?",
        }
        
        for query_type, query in query_types.items():
            result = analyzer.analyze(query)
            
            # Should successfully analyze all types
            assert result is not None
            assert result.intent is not None
            assert 0.0 <= result.confidence <= 1.0
            
            # Should extract appropriate information
            if query_type == "temporal":
                assert len(result.temporal_expressions) > 0 or result.intent == QueryIntent.TEMPORAL
            elif query_type == "code":
                assert len(result.code_patterns) > 0 or result.intent == QueryIntent.CODE
            elif query_type == "preference":
                assert result.intent in [QueryIntent.PREFERENCE, QueryIntent.MIXED]
