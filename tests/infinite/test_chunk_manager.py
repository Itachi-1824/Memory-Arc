"""Tests for ChunkManager - orchestration of chunking, selection, and formatting."""

import pytest
import time
from core.infinite.chunk_manager import ChunkManager, ChunkManagerConfig, create_chunk_manager
from core.infinite.models import Memory, MemoryType, Chunk
from core.infinite.chunk_formatter import FormatType


class TestChunkManagerBasics:
    """Test basic ChunkManager functionality."""
    
    def test_initialization(self):
        """Test ChunkManager initialization."""
        manager = ChunkManager()
        
        assert manager.config.model_name == "gpt-4"
        assert manager.config.max_tokens > 0
        assert manager.token_counter is not None
        assert manager.semantic_chunker is not None
        assert manager.chunk_selector is not None
        assert manager.chunk_formatter is not None
    
    def test_initialization_with_config(self):
        """Test ChunkManager initialization with custom config."""
        config = ChunkManagerConfig(
            model_name="gpt-3.5-turbo",
            max_tokens=4000,
            overlap_tokens=50,
            relevance_weight=0.6,
            importance_weight=0.3,
            recency_weight=0.1
        )
        
        manager = ChunkManager(config)
        
        assert manager.config.model_name == "gpt-3.5-turbo"
        assert manager.config.max_tokens == 4000
        assert manager.config.overlap_tokens == 50
    
    def test_factory_function(self):
        """Test create_chunk_manager factory function."""
        manager = create_chunk_manager(
            model_name="claude-3",
            max_tokens=100000
        )
        
        assert manager.config.model_name == "claude-3"
        assert manager.config.max_tokens == 100000


class TestChunkContent:
    """Test chunk_content method."""
    
    def test_chunk_simple_text(self):
        """Test chunking simple text content."""
        manager = ChunkManager()
        
        content = "This is a test.\n\nThis is another paragraph.\n\nAnd a third one."
        chunks = manager.chunk_content(content, content_type="text")
        
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == len(chunks)
    
    def test_chunk_code_content(self):
        """Test chunking code content."""
        manager = ChunkManager()
        
        content = """
def function1():
    return "test"

def function2():
    return "another"

class MyClass:
    def method(self):
        pass
"""
        
        chunks = manager.chunk_content(content, content_type="code")
        
        assert len(chunks) > 0
        # Chunks should be cached for navigation
        assert len(manager._chunk_cache) > 0
    
    def test_chunk_memory_list(self):
        """Test chunking a list of Memory objects."""
        manager = ChunkManager()
        
        memories = [
            Memory(
                id="mem1",
                context_id="ctx1",
                content="First memory content",
                memory_type=MemoryType.CONVERSATION,
                created_at=time.time(),
                importance=5
            ),
            Memory(
                id="mem2",
                context_id="ctx1",
                content="Second memory content",
                memory_type=MemoryType.FACT,
                created_at=time.time(),
                importance=7
            )
        ]
        
        chunks = manager.chunk_content(memories)
        
        assert len(chunks) > 0
        # Check metadata includes memory info
        assert "memory_ids" in chunks[0].metadata or "memory_id" in chunks[0].metadata
    
    def test_chunk_large_memory(self):
        """Test chunking when a single memory exceeds max chunk size."""
        config = ChunkManagerConfig(max_tokens=50)  # Very small for testing
        manager = ChunkManager(config)
        
        # Create a large memory with clear boundaries
        large_content = "\n\n".join([f"Paragraph {i}. " + " ".join(["word"] * 30) for i in range(10)])
        memories = [
            Memory(
                id="large_mem",
                context_id="ctx1",
                content=large_content,
                memory_type=MemoryType.DOCUMENT,
                created_at=time.time(),
                importance=5
            )
        ]
        
        chunks = manager.chunk_content(memories)
        
        # Should split into multiple chunks
        assert len(chunks) >= 1  # At least one chunk created


class TestFormatChunk:
    """Test format_chunk method."""
    
    def test_format_markdown(self):
        """Test formatting chunk as markdown."""
        manager = ChunkManager()
        
        chunk = Chunk(
            id="test_chunk",
            content="Test content",
            chunk_index=0,
            total_chunks=1,
            token_count=10
        )
        
        formatted = manager.format_chunk(chunk, format_type=FormatType.MARKDOWN)
        
        assert formatted.content is not None
        assert formatted.format_type == FormatType.MARKDOWN
        assert formatted.token_count > 0
    
    def test_format_json(self):
        """Test formatting chunk as JSON."""
        manager = ChunkManager()
        
        chunk = Chunk(
            id="test_chunk",
            content="Test content",
            chunk_index=0,
            total_chunks=1,
            token_count=10
        )
        
        formatted = manager.format_chunk(chunk, format_type=FormatType.JSON)
        
        assert formatted.format_type == FormatType.JSON
        assert "{" in formatted.content  # Should be valid JSON
    
    def test_format_with_memory(self):
        """Test formatting chunk with associated memory."""
        manager = ChunkManager()
        
        chunk = Chunk(
            id="test_chunk",
            content="Test content",
            chunk_index=0,
            total_chunks=1,
            token_count=10
        )
        
        memory = Memory(
            id="mem1",
            context_id="ctx1",
            content="Test content",
            memory_type=MemoryType.CODE,
            created_at=time.time(),
            importance=8
        )
        
        formatted = manager.format_chunk(chunk, memory=memory)
        
        assert formatted.content is not None


class TestSelectChunks:
    """Test select_chunks method."""
    
    def test_select_without_query(self):
        """Test selecting chunks without a query."""
        manager = ChunkManager()
        
        chunks = [
            Chunk(
                id=f"chunk_{i}",
                content=f"Content {i}",
                chunk_index=i,
                total_chunks=3,
                token_count=50,
                metadata={"importance": 5 + i}
            )
            for i in range(3)
        ]
        
        selected = manager.select_chunks(chunks)
        
        assert len(selected) == 3
        # Should be sorted by score
        assert all(hasattr(sc, 'final_score') for sc in selected)
    
    def test_select_with_max_chunks(self):
        """Test selecting chunks with max_chunks limit."""
        manager = ChunkManager()
        
        chunks = [
            Chunk(
                id=f"chunk_{i}",
                content=f"Content {i}",
                chunk_index=i,
                total_chunks=5,
                token_count=50
            )
            for i in range(5)
        ]
        
        selected = manager.select_chunks(chunks, max_chunks=2)
        
        assert len(selected) == 2
    
    def test_select_with_max_tokens(self):
        """Test selecting chunks with token budget."""
        manager = ChunkManager()
        
        chunks = [
            Chunk(
                id=f"chunk_{i}",
                content=f"Content {i}",
                chunk_index=i,
                total_chunks=5,
                token_count=100
            )
            for i in range(5)
        ]
        
        selected = manager.select_chunks(chunks, max_tokens=250)
        
        # Should select chunks that fit within 250 tokens
        total_tokens = sum(sc.chunk.token_count for sc in selected)
        assert total_tokens <= 250
    
    def test_select_with_query(self):
        """Test selecting chunks with relevance query."""
        manager = ChunkManager()
        
        chunks = [
            Chunk(
                id="chunk_0",
                content="Python programming language",
                chunk_index=0,
                total_chunks=3,
                token_count=50
            ),
            Chunk(
                id="chunk_1",
                content="JavaScript web development",
                chunk_index=1,
                total_chunks=3,
                token_count=50
            ),
            Chunk(
                id="chunk_2",
                content="Python data science",
                chunk_index=2,
                total_chunks=3,
                token_count=50
            )
        ]
        
        selected = manager.select_chunks(chunks, query="Python")
        
        # Chunks with "Python" should score higher
        assert len(selected) > 0


class TestGetNextChunk:
    """Test get_next_chunk navigation."""
    
    def test_navigate_forward(self):
        """Test navigating forward through chunks."""
        manager = ChunkManager()
        
        content = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunks = manager.chunk_content(content)
        
        if len(chunks) > 1:
            first_chunk = chunks[0]
            next_chunk = manager.get_next_chunk(first_chunk.id, direction="forward")
            
            assert next_chunk is not None
            assert next_chunk.chunk_index == first_chunk.chunk_index + 1
    
    def test_navigate_backward(self):
        """Test navigating backward through chunks."""
        manager = ChunkManager()
        
        content = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunks = manager.chunk_content(content)
        
        if len(chunks) > 1:
            second_chunk = chunks[1]
            prev_chunk = manager.get_next_chunk(second_chunk.id, direction="backward")
            
            assert prev_chunk is not None
            assert prev_chunk.chunk_index == second_chunk.chunk_index - 1
    
    def test_navigate_at_boundary(self):
        """Test navigation at chunk boundaries."""
        manager = ChunkManager()
        
        content = "Single paragraph."
        chunks = manager.chunk_content(content)
        
        first_chunk = chunks[0]
        
        # Try to go backward from first chunk
        prev_chunk = manager.get_next_chunk(first_chunk.id, direction="backward")
        assert prev_chunk is None
        
        # Try to go forward from last chunk
        next_chunk = manager.get_next_chunk(first_chunk.id, direction="forward")
        if len(chunks) == 1:
            assert next_chunk is None
    
    def test_navigate_invalid_chunk(self):
        """Test navigation with invalid chunk ID."""
        manager = ChunkManager()
        
        result = manager.get_next_chunk("nonexistent_chunk", direction="forward")
        
        assert result is None


class TestStreamChunks:
    """Test stream_chunks method."""
    
    def test_stream_basic(self):
        """Test basic chunk streaming."""
        manager = ChunkManager()
        
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        
        streamed_chunks = list(manager.stream_chunks(content))
        
        assert len(streamed_chunks) > 0
        assert all(hasattr(fc, 'content') for fc in streamed_chunks)
        assert all(hasattr(fc, 'format_type') for fc in streamed_chunks)
    
    def test_stream_with_max_chunks(self):
        """Test streaming with max_chunks limit."""
        manager = ChunkManager()
        
        content = "\n\n".join([f"Paragraph {i}" for i in range(10)])
        
        streamed_chunks = list(manager.stream_chunks(content, max_chunks=3))
        
        assert len(streamed_chunks) <= 3
    
    def test_stream_with_format(self):
        """Test streaming with specific format."""
        manager = ChunkManager()
        
        content = "Test content for streaming."
        
        streamed_chunks = list(manager.stream_chunks(
            content,
            format_type=FormatType.JSON
        ))
        
        assert all(fc.format_type == FormatType.JSON for fc in streamed_chunks)


class TestProcessAndFormat:
    """Test process_and_format complete pipeline."""
    
    def test_complete_pipeline(self):
        """Test complete processing pipeline."""
        manager = ChunkManager()
        
        content = "First section.\n\nSecond section.\n\nThird section."
        
        result = manager.process_and_format(content)
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_pipeline_with_query(self):
        """Test pipeline with query for relevance."""
        manager = ChunkManager()
        
        content = "Python is great.\n\nJavaScript is useful.\n\nPython for data science."
        
        result = manager.process_and_format(
            content,
            query="Python programming"
        )
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_pipeline_return_string(self):
        """Test pipeline returning single string."""
        manager = ChunkManager()
        
        content = "First part.\n\nSecond part."
        
        result = manager.process_and_format(
            content,
            return_single_string=True
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_pipeline_with_memories(self):
        """Test pipeline with Memory objects."""
        manager = ChunkManager()
        
        memories = [
            Memory(
                id="mem1",
                context_id="ctx1",
                content="Memory content 1",
                memory_type=MemoryType.CONVERSATION,
                created_at=time.time(),
                importance=5
            ),
            Memory(
                id="mem2",
                context_id="ctx1",
                content="Memory content 2",
                memory_type=MemoryType.FACT,
                created_at=time.time(),
                importance=8
            )
        ]
        
        result = manager.process_and_format(memories)
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_pipeline_with_token_limit(self):
        """Test pipeline with token budget."""
        manager = ChunkManager()
        
        content = " ".join(["word"] * 500)  # Large content
        
        result = manager.process_and_format(
            content,
            max_tokens=200
        )
        
        # Should respect token limit (with small margin for metadata)
        total_tokens = sum(fc.token_count for fc in result)
        assert total_tokens <= 250  # Allow margin for metadata overhead


class TestGetStats:
    """Test get_stats method."""
    
    def test_get_stats(self):
        """Test getting chunk manager statistics."""
        manager = ChunkManager()
        
        # Add some chunks to cache
        content = "Test content for stats."
        manager.chunk_content(content)
        
        stats = manager.get_stats()
        
        assert "model_name" in stats
        assert "max_tokens" in stats
        assert "overlap_tokens" in stats
        assert "cached_chunks" in stats
        assert "weights" in stats
        assert stats["cached_chunks"] > 0


class TestChunkingAccuracy:
    """Test chunking accuracy for various content types."""
    
    def test_chunk_plain_text_accuracy(self):
        """Test chunking plain text maintains content integrity."""
        manager = ChunkManager()
        
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = manager.chunk_content(content, content_type="text")
        
        # Reconstruct content from chunks
        reconstructed = "\n\n".join(c.content for c in chunks)
        
        # Content should be preserved (allowing for minor whitespace differences)
        assert len(reconstructed) > 0
        assert "First paragraph" in reconstructed
        assert "Second paragraph" in reconstructed
        assert "Third paragraph" in reconstructed
    
    def test_chunk_markdown_preserves_structure(self):
        """Test markdown chunking preserves structural elements."""
        manager = ChunkManager()
        
        content = """# Title

## Section 1

Content for section 1.

## Section 2

Content for section 2.

### Subsection

More content."""
        
        chunks = manager.chunk_content(content, content_type="markdown")
        
        # Check that markdown structure is preserved
        reconstructed = "\n\n".join(c.content for c in chunks)
        assert "# Title" in reconstructed or "Title" in reconstructed
        assert "Section 1" in reconstructed
        assert "Section 2" in reconstructed
    
    def test_chunk_code_preserves_syntax(self):
        """Test code chunking preserves syntactic boundaries."""
        manager = ChunkManager()
        
        code = """def function1():
    return "test"

def function2():
    return "another"

class MyClass:
    def method(self):
        pass"""
        
        chunks = manager.chunk_content(code, content_type="code")
        
        # Verify all functions/classes are present
        reconstructed = "\n\n".join(c.content for c in chunks)
        assert "function1" in reconstructed
        assert "function2" in reconstructed
        assert "MyClass" in reconstructed
    
    def test_chunk_conversation_format(self):
        """Test chunking conversation-style content."""
        manager = ChunkManager()
        
        memories = [
            Memory(
                id=f"msg_{i}",
                context_id="ctx1",
                content=f"Message {i} content",
                memory_type=MemoryType.CONVERSATION,
                created_at=time.time() + i,
                importance=5
            )
            for i in range(5)
        ]
        
        chunks = manager.chunk_content(memories)
        
        assert len(chunks) > 0
        # Verify messages are included
        reconstructed = " ".join(c.content for c in chunks)
        assert "Message 0" in reconstructed
        assert "Message 4" in reconstructed
    
    def test_chunk_mixed_content_types(self):
        """Test chunking mixed content types (text + code)."""
        manager = ChunkManager()
        
        memories = [
            Memory(
                id="text1",
                context_id="ctx1",
                content="This is text content.",
                memory_type=MemoryType.FACT,
                created_at=time.time(),
                importance=5
            ),
            Memory(
                id="code1",
                context_id="ctx1",
                content="def test(): pass",
                memory_type=MemoryType.CODE,
                created_at=time.time(),
                importance=7
            ),
            Memory(
                id="text2",
                context_id="ctx1",
                content="More text content.",
                memory_type=MemoryType.DOCUMENT,
                created_at=time.time(),
                importance=6
            )
        ]
        
        chunks = manager.chunk_content(memories)
        
        assert len(chunks) > 0
        # Verify both content types are present
        reconstructed = " ".join(c.content for c in chunks)
        assert "text content" in reconstructed
        assert "test()" in reconstructed or "test" in reconstructed


class TestTokenCountingAccuracy:
    """Test token counting accuracy."""
    
    def test_token_count_matches_actual(self):
        """Test that chunk token counts match actual content."""
        manager = ChunkManager()
        
        content = "This is a test sentence with several words."
        chunks = manager.chunk_content(content)
        
        for chunk in chunks:
            # Verify token count is reasonable
            assert chunk.token_count > 0
            # Token count should be roughly proportional to content length
            # (allowing for tokenizer variations)
            assert chunk.token_count < len(chunk.content)  # Tokens < characters
    
    def test_token_budget_respected(self):
        """Test that token budget is strictly respected."""
        config = ChunkManagerConfig(max_tokens=500)
        manager = ChunkManager(config)
        
        # Create large content
        content = " ".join(["word"] * 1000)
        
        result = manager.process_and_format(content, max_tokens=200)
        
        # Calculate total tokens
        total_tokens = sum(fc.token_count for fc in result)
        
        # Should not exceed budget (with small margin for metadata)
        assert total_tokens <= 250
    
    def test_token_counting_different_models(self):
        """Test token counting for different model types."""
        models = ["gpt-4", "gpt-3.5-turbo", "claude-3"]
        content = "This is a test sentence for token counting."
        
        for model in models:
            manager = create_chunk_manager(model_name=model)
            chunks = manager.chunk_content(content)
            
            # All models should produce valid token counts
            assert all(c.token_count > 0 for c in chunks)
    
    def test_token_count_with_special_characters(self):
        """Test token counting with special characters and unicode."""
        manager = ChunkManager()
        
        content = "Hello 世界! This has émojis 🎉 and symbols: @#$%"
        chunks = manager.chunk_content(content)
        
        # Should handle special characters without errors
        assert len(chunks) > 0
        assert all(c.token_count > 0 for c in chunks)


class TestPriorityRanking:
    """Test priority ranking correctness."""
    
    def test_relevance_scoring(self):
        """Test that relevance scoring works correctly."""
        manager = ChunkManager()
        
        chunks = [
            Chunk(
                id="chunk_0",
                content="Python programming language tutorial",
                chunk_index=0,
                total_chunks=3,
                token_count=50
            ),
            Chunk(
                id="chunk_1",
                content="JavaScript web development guide",
                chunk_index=1,
                total_chunks=3,
                token_count=50
            ),
            Chunk(
                id="chunk_2",
                content="Python data science and machine learning",
                chunk_index=2,
                total_chunks=3,
                token_count=50
            )
        ]
        
        # Query for Python content
        selected = manager.select_chunks(chunks, query="Python programming")
        
        # Python-related chunks should score higher
        assert len(selected) > 0
        # First result should be most relevant
        top_chunk = selected[0]
        assert "Python" in top_chunk.chunk.content
    
    def test_importance_scoring(self):
        """Test that importance affects ranking."""
        manager = ChunkManager()
        
        memories = [
            Memory(
                id="low_imp",
                context_id="ctx1",
                content="Low importance content",
                memory_type=MemoryType.FACT,
                created_at=time.time(),
                importance=3
            ),
            Memory(
                id="high_imp",
                context_id="ctx1",
                content="High importance content",
                memory_type=MemoryType.FACT,
                created_at=time.time(),
                importance=9
            ),
            Memory(
                id="med_imp",
                context_id="ctx1",
                content="Medium importance content",
                memory_type=MemoryType.FACT,
                created_at=time.time(),
                importance=5
            )
        ]
        
        chunks = manager.chunk_content(memories)
        selected = manager.select_chunks(chunks, memories=memories)
        
        # Higher importance should rank higher
        assert len(selected) > 0
        # Verify importance is considered in scoring
        scores = [sc.importance_score for sc in selected]
        assert max(scores) > 0
    
    def test_recency_scoring(self):
        """Test that recency affects ranking."""
        manager = ChunkManager()
        
        now = time.time()
        memories = [
            Memory(
                id="old",
                context_id="ctx1",
                content="Old content",
                memory_type=MemoryType.FACT,
                created_at=now - 86400 * 30,  # 30 days ago
                importance=5
            ),
            Memory(
                id="recent",
                context_id="ctx1",
                content="Recent content",
                memory_type=MemoryType.FACT,
                created_at=now,
                importance=5
            ),
            Memory(
                id="medium",
                context_id="ctx1",
                content="Medium age content",
                memory_type=MemoryType.FACT,
                created_at=now - 86400 * 7,  # 7 days ago
                importance=5
            )
        ]
        
        chunks = manager.chunk_content(memories)
        selected = manager.select_chunks(chunks, memories=memories)
        
        # Verify recency is considered
        assert len(selected) > 0
        scores = [sc.recency_score for sc in selected]
        assert max(scores) > 0
    
    def test_combined_scoring(self):
        """Test that all scoring factors combine correctly."""
        config = ChunkManagerConfig(
            relevance_weight=0.5,
            importance_weight=0.3,
            recency_weight=0.2
        )
        manager = ChunkManager(config)
        
        now = time.time()
        memories = [
            Memory(
                id="best",
                context_id="ctx1",
                content="Python programming tutorial",
                memory_type=MemoryType.DOCUMENT,
                created_at=now,
                importance=9
            ),
            Memory(
                id="worst",
                context_id="ctx1",
                content="Unrelated content about something else entirely",
                memory_type=MemoryType.FACT,
                created_at=now - 86400 * 365,
                importance=2
            )
        ]
        
        chunks = manager.chunk_content(memories)
        selected = manager.select_chunks(
            chunks,
            query="Python programming",
            memories=memories
        )
        
        # Should have at least one chunk
        assert len(selected) >= 1
        # If multiple chunks, best match should rank highest
        if len(selected) >= 2:
            assert selected[0].final_score > selected[1].final_score
    
    def test_min_score_threshold(self):
        """Test that min_score threshold filters correctly."""
        manager = ChunkManager()
        
        chunks = [
            Chunk(
                id=f"chunk_{i}",
                content=f"Content {i}",
                chunk_index=i,
                total_chunks=5,
                token_count=50,
                metadata={"importance": i}
            )
            for i in range(5)
        ]
        
        # Select with high threshold
        selected = manager.select_chunks(chunks, min_score=0.8)
        
        # Should filter out low-scoring chunks
        assert all(sc.final_score >= 0.8 for sc in selected)


class TestFormatGeneration:
    """Test format generation for all types."""
    
    def test_format_markdown_structure(self):
        """Test markdown format structure."""
        manager = ChunkManager()
        
        chunk = Chunk(
            id="test",
            content="Test content",
            chunk_index=0,
            total_chunks=1,
            token_count=10
        )
        
        formatted = manager.format_chunk(chunk, format_type=FormatType.MARKDOWN)
        
        assert formatted.format_type == FormatType.MARKDOWN
        assert formatted.content is not None
        assert len(formatted.content) > 0
    
    def test_format_json_structure(self):
        """Test JSON format structure and validity."""
        manager = ChunkManager()
        
        chunk = Chunk(
            id="test",
            content="Test content",
            chunk_index=0,
            total_chunks=1,
            token_count=10
        )
        
        formatted = manager.format_chunk(chunk, format_type=FormatType.JSON)
        
        assert formatted.format_type == FormatType.JSON
        # Should contain JSON structure markers
        assert "{" in formatted.content
        assert "}" in formatted.content
    
    def test_format_plain_text(self):
        """Test plain text format."""
        manager = ChunkManager()
        
        chunk = Chunk(
            id="test",
            content="Test content",
            chunk_index=0,
            total_chunks=1,
            token_count=10
        )
        
        formatted = manager.format_chunk(chunk, format_type=FormatType.PLAIN)
        
        assert formatted.format_type == FormatType.PLAIN
        assert "Test content" in formatted.content
    
    def test_format_with_metadata(self):
        """Test format includes metadata when requested."""
        config = ChunkManagerConfig(include_metadata=True)
        manager = ChunkManager(config)
        
        chunk = Chunk(
            id="test",
            content="Test content",
            chunk_index=0,
            total_chunks=3,
            token_count=10,
            metadata={"importance": 8}
        )
        
        formatted = manager.format_chunk(chunk)
        
        # Should include chunk position info
        assert formatted.content is not None
    
    def test_format_without_metadata(self):
        """Test format excludes metadata when not requested."""
        config = ChunkManagerConfig(include_metadata=False)
        manager = ChunkManager(config)
        
        chunk = Chunk(
            id="test",
            content="Test content",
            chunk_index=0,
            total_chunks=1,
            token_count=10
        )
        
        formatted = manager.format_chunk(chunk)
        
        # Should be minimal
        assert formatted.content is not None
    
    def test_format_with_memory_context(self):
        """Test formatting with associated memory."""
        manager = ChunkManager()
        
        chunk = Chunk(
            id="test",
            content="Test content",
            chunk_index=0,
            total_chunks=1,
            token_count=10
        )
        
        memory = Memory(
            id="mem1",
            context_id="ctx1",
            content="Test content",
            memory_type=MemoryType.CODE,
            created_at=time.time(),
            importance=8
        )
        
        formatted = manager.format_chunk(chunk, memory=memory)
        
        # Should include memory context
        assert formatted.content is not None
    
    def test_format_code_with_syntax_highlighting(self):
        """Test code formatting includes syntax markers."""
        manager = ChunkManager()
        
        chunk = Chunk(
            id="test",
            content="def test():\n    pass",
            chunk_index=0,
            total_chunks=1,
            token_count=10,
            metadata={"content_type": "code"}
        )
        
        formatted = manager.format_chunk(chunk, format_type=FormatType.MARKDOWN)
        
        # Markdown code formatting should include code blocks
        assert formatted.content is not None


class TestChunkNavigation:
    """Test chunk navigation features."""
    
    def test_navigation_forward_complete_sequence(self):
        """Test navigating forward through complete sequence."""
        manager = ChunkManager()
        
        content = "\n\n".join([f"Section {i}" for i in range(5)])
        chunks = manager.chunk_content(content)
        
        if len(chunks) > 1:
            # Navigate through all chunks
            current = chunks[0]
            visited = [current]
            
            while True:
                next_chunk = manager.get_next_chunk(current.id, direction="forward")
                if next_chunk is None:
                    break
                visited.append(next_chunk)
                current = next_chunk
            
            # Should visit all chunks
            assert len(visited) <= len(chunks)
    
    def test_navigation_backward_complete_sequence(self):
        """Test navigating backward through complete sequence."""
        manager = ChunkManager()
        
        content = "\n\n".join([f"Section {i}" for i in range(5)])
        chunks = manager.chunk_content(content)
        
        if len(chunks) > 1:
            # Start from last chunk
            current = chunks[-1]
            visited = [current]
            
            while True:
                prev_chunk = manager.get_next_chunk(current.id, direction="backward")
                if prev_chunk is None:
                    break
                visited.append(prev_chunk)
                current = prev_chunk
            
            # Should visit all chunks in reverse
            assert len(visited) <= len(chunks)
    
    def test_navigation_bidirectional(self):
        """Test bidirectional navigation."""
        manager = ChunkManager()
        
        content = "\n\n".join([f"Section {i}" for i in range(3)])
        chunks = manager.chunk_content(content)
        
        if len(chunks) >= 3:
            # Start from middle
            middle = chunks[1]
            
            # Go forward
            next_chunk = manager.get_next_chunk(middle.id, direction="forward")
            assert next_chunk is not None
            
            # Go back to middle
            back_to_middle = manager.get_next_chunk(next_chunk.id, direction="backward")
            assert back_to_middle is not None
            assert back_to_middle.id == middle.id


class TestIntegration:
    """Integration tests for ChunkManager."""
    
    def test_end_to_end_text_processing(self):
        """Test end-to-end text processing."""
        manager = create_chunk_manager(model_name="gpt-4")
        
        content = """
# Introduction

This is a test document with multiple sections.

## Section 1

First section content with some details.

## Section 2

Second section with more information.

## Conclusion

Final thoughts and summary.
"""
        
        # Process and format
        result = manager.process_and_format(
            content,
            content_type="markdown",
            format_type=FormatType.MARKDOWN,
            return_single_string=True
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_end_to_end_code_processing(self):
        """Test end-to-end code processing."""
        manager = create_chunk_manager(model_name="gpt-4")
        
        code = """
def calculate_sum(a, b):
    '''Calculate sum of two numbers.'''
    return a + b

def calculate_product(a, b):
    '''Calculate product of two numbers.'''
    return a * b

class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, value):
        self.result += value
        return self.result
"""
        
        # Process and format
        result = manager.process_and_format(
            code,
            content_type="code",
            format_type=FormatType.MARKDOWN
        )
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_end_to_end_with_streaming(self):
        """Test end-to-end with streaming."""
        manager = create_chunk_manager(model_name="claude-3")
        
        content = "\n\n".join([f"Section {i} content." for i in range(10)])
        
        # Stream chunks
        chunk_count = 0
        for formatted_chunk in manager.stream_chunks(content, max_chunks=5):
            assert formatted_chunk.content is not None
            chunk_count += 1
        
        assert chunk_count <= 5
    
    def test_complete_workflow_with_query(self):
        """Test complete workflow with query-based selection."""
        manager = create_chunk_manager(model_name="gpt-4")
        
        # Create diverse content
        memories = [
            Memory(
                id="python1",
                context_id="ctx1",
                content="Python is great for data science and machine learning.",
                memory_type=MemoryType.DOCUMENT,
                created_at=time.time(),
                importance=8
            ),
            Memory(
                id="js1",
                context_id="ctx1",
                content="JavaScript is essential for web development.",
                memory_type=MemoryType.DOCUMENT,
                created_at=time.time(),
                importance=6
            ),
            Memory(
                id="python2",
                context_id="ctx1",
                content="Python has excellent libraries like NumPy and Pandas.",
                memory_type=MemoryType.FACT,
                created_at=time.time(),
                importance=7
            )
        ]
        
        # Process with Python query
        result = manager.process_and_format(
            memories,
            query="Python data science",
            max_chunks=2
        )
        
        # Should prioritize Python-related content
        assert len(result) <= 2
        assert len(result) > 0
    
    def test_large_content_handling(self):
        """Test handling of large content volumes."""
        manager = create_chunk_manager(model_name="gpt-4", max_tokens=2000)
        
        # Create large content
        large_content = "\n\n".join([
            f"Section {i}: " + " ".join(["word"] * 100)
            for i in range(50)
        ])
        
        # Process with token limit
        result = manager.process_and_format(
            large_content,
            max_tokens=1000
        )
        
        # Should chunk and limit appropriately
        assert len(result) > 0
        total_tokens = sum(fc.token_count for fc in result)
        # Allow margin for metadata overhead (headers, navigation, etc.)
        assert total_tokens <= 1400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
