"""Semantic versioning logic for memory evolution."""

import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)


class SemanticVersioning:
    """Handles semantic comparison and versioning of memories."""

    def __init__(self, similarity_threshold: float = 0.7):
        """Initialize semantic versioning.
        
        Args:
            similarity_threshold: Threshold for considering memories similar (0-1)
        """
        self.similarity_threshold = similarity_threshold

    def compute_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float]
    ) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        if not embedding1 or not embedding2:
            return 0.0

        if len(embedding1) != len(embedding2):
            logger.warning(f"Embedding dimension mismatch: {len(embedding1)} vs {len(embedding2)}")
            return 0.0

        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        
        # Normalize to 0-1 range (cosine similarity is -1 to 1)
        normalized = (similarity + 1) / 2
        
        return float(normalized)

    def compute_semantic_delta(
        self,
        content1: str,
        content2: str,
        embedding1: list[float] | None = None,
        embedding2: list[float] | None = None
    ) -> dict[str, Any]:
        """Compute semantic delta between two memory versions.
        
        This method analyzes the differences between two memory versions,
        computing both structural and semantic changes.
        
        Args:
            content1: Content of first memory
            content2: Content of second memory
            embedding1: Optional embedding of first memory
            embedding2: Optional embedding of second memory
            
        Returns:
            Dictionary with delta information including:
            - content_changed: Whether content differs
            - length_delta: Change in content length
            - similarity: Semantic similarity score (0-1)
            - change_type: Classification of change magnitude
            - word_overlap: Ratio of shared words
            - structural_changes: Dict of structural differences
        """
        delta = {
            "content_changed": content1 != content2,
            "length_delta": len(content2) - len(content1),
            "similarity": None,
            "change_type": None,
            "word_overlap": 0.0,
            "structural_changes": {}
        }

        # Compute word-level overlap
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if words1 or words2:
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            delta["word_overlap"] = len(intersection) / len(union) if union else 0.0

        # Analyze structural changes
        delta["structural_changes"] = {
            "words_added": len(words2 - words1),
            "words_removed": len(words1 - words2),
            "words_retained": len(intersection),
            "sentence_count_delta": content2.count('.') - content1.count('.')
        }

        # Compute similarity if embeddings provided
        if embedding1 and embedding2:
            similarity = self.compute_similarity(embedding1, embedding2)
            delta["similarity"] = similarity

            # Classify change type based on similarity and word overlap
            if similarity >= 0.95 and delta["word_overlap"] >= 0.9:
                delta["change_type"] = "typo_fix"
            elif similarity >= 0.9:
                delta["change_type"] = "minor_edit"
            elif similarity >= 0.7:
                delta["change_type"] = "moderate_change"
            elif similarity >= 0.4:
                delta["change_type"] = "major_change"
            else:
                delta["change_type"] = "complete_rewrite"
        else:
            # Fallback to word overlap if no embeddings
            if delta["word_overlap"] >= 0.9:
                delta["change_type"] = "minor_edit"
            elif delta["word_overlap"] >= 0.6:
                delta["change_type"] = "moderate_change"
            elif delta["word_overlap"] >= 0.3:
                delta["change_type"] = "major_change"
            else:
                delta["change_type"] = "complete_rewrite"

        return delta

    def detect_relationship(
        self,
        embedding1: list[float],
        embedding2: list[float],
        content1: str | None = None,
        content2: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Detect the relationship between two memory versions.
        
        This method determines how two memories relate to each other based on
        semantic similarity and optional content analysis.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            content1: Optional content of first memory for enhanced analysis
            content2: Optional content of second memory for enhanced analysis
            
        Returns:
            Tuple of (relationship_type, metadata) where relationship_type is:
            'identical', 'similar', 'related', 'unrelated', 'contradictory'
            and metadata contains additional relationship information
        """
        similarity = self.compute_similarity(embedding1, embedding2)
        
        metadata = {
            "similarity_score": similarity,
            "confidence": 0.0
        }

        # Check for contradiction if content provided
        if content1 and content2:
            is_contradiction, contra_confidence = self.detect_contradiction(
                content1, content2, embedding1, embedding2
            )
            
            if is_contradiction and contra_confidence > 0.6:
                metadata["confidence"] = contra_confidence
                return "contradictory", metadata

        # Determine relationship based on similarity
        if similarity >= 0.95:
            relationship = "identical"
            metadata["confidence"] = 0.95
        elif similarity >= self.similarity_threshold:
            relationship = "similar"
            metadata["confidence"] = similarity
        elif similarity >= 0.4:
            relationship = "related"
            metadata["confidence"] = similarity * 0.8
        else:
            relationship = "unrelated"
            metadata["confidence"] = 1.0 - similarity

        return relationship, metadata

    def detect_contradiction(
        self,
        content1: str,
        content2: str,
        embedding1: list[float],
        embedding2: list[float]
    ) -> tuple[bool, float]:
        """Detect if two memories contradict each other.
        
        This is an enhanced heuristic-based approach that looks for:
        1. Similar semantic content (high embedding similarity)
        2. Opposite sentiment or negation patterns
        3. Preference reversals
        4. Factual contradictions
        
        Args:
            content1: Content of first memory
            content2: Content of second memory
            embedding1: Embedding of first memory
            embedding2: Embedding of second memory
            
        Returns:
            Tuple of (is_contradiction, confidence_score)
        """
        # Compute semantic similarity
        similarity = self.compute_similarity(embedding1, embedding2)

        # If not semantically related, unlikely to be contradictory
        if similarity < 0.5:
            return False, 0.0

        content1_lower = content1.lower()
        content2_lower = content2.lower()

        # Enhanced negation detection
        negation_words = [
            "not", "no", "never", "don't", "doesn't", "didn't", 
            "won't", "can't", "couldn't", "shouldn't", "wouldn't",
            "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't"
        ]
        
        # Contradiction indicators
        contradiction_indicators = [
            "but", "however", "instead", "rather", "prefer",
            "actually", "contrary", "opposite", "disagree"
        ]
        
        # Preference reversal patterns
        preference_patterns = [
            ("like", "dislike"), ("love", "hate"), ("prefer", "avoid"),
            ("want", "don't want"), ("enjoy", "dislike"), ("favor", "oppose")
        ]

        # Count negation and contradiction indicators
        neg_count1 = sum(1 for word in negation_words if f" {word} " in f" {content1_lower} ")
        neg_count2 = sum(1 for word in negation_words if f" {word} " in f" {content2_lower} ")
        
        contra_count = sum(1 for word in contradiction_indicators if word in content2_lower)

        # Check for preference reversals
        preference_reversal = False
        for pos, neg in preference_patterns:
            if (pos in content1_lower and neg in content2_lower) or \
               (neg in content1_lower and pos in content2_lower):
                preference_reversal = True
                break

        # Calculate contradiction confidence
        confidence = 0.0
        is_contradiction = False

        # Negation difference (one has negation, other doesn't)
        has_negation_difference = (neg_count1 > 0) != (neg_count2 > 0)
        
        if has_negation_difference:
            # Higher similarity + negation difference = higher contradiction confidence
            confidence = similarity * 0.7
            is_contradiction = True

        # Contradiction indicators boost confidence
        if contra_count > 0:
            confidence += 0.15 * min(contra_count, 2)  # Cap at 2 indicators
            is_contradiction = True

        # Preference reversal is strong signal
        if preference_reversal:
            confidence += 0.3
            is_contradiction = True

        # High similarity with different sentiment
        if similarity >= 0.7 and (has_negation_difference or preference_reversal):
            confidence = max(confidence, 0.8)
            is_contradiction = True

        # Ensure confidence is in valid range
        confidence = min(confidence, 1.0)

        return is_contradiction, confidence

    def find_contradictions(
        self,
        memories: list[tuple[str, str, list[float]]]
    ) -> list[tuple[int, int, float]]:
        """Find contradictory memories in a list.
        
        Args:
            memories: List of (memory_id, content, embedding) tuples
            
        Returns:
            List of (index1, index2, confidence) tuples for contradictions
        """
        contradictions = []

        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                id1, content1, emb1 = memories[i]
                id2, content2, emb2 = memories[j]

                is_contradiction, confidence = self.detect_contradiction(
                    content1, content2, emb1, emb2
                )

                if is_contradiction and confidence > 0.5:
                    contradictions.append((i, j, confidence))

        return contradictions

    def compute_version_similarity_batch(
        self,
        embeddings: list[list[float]]
    ) -> list[list[float]]:
        """Compute pairwise similarity matrix for a batch of embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            2D similarity matrix where matrix[i][j] is similarity between i and j
        """
        n = len(embeddings)
        similarity_matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    sim = self.compute_similarity(embeddings[i], embeddings[j])
                    similarity_matrix[i][j] = sim
                    similarity_matrix[j][i] = sim

        return similarity_matrix

    def detect_relationships_batch(
        self,
        memories: list[tuple[str, str, list[float]]]
    ) -> dict[tuple[int, int], tuple[str, dict[str, Any]]]:
        """Detect relationships between multiple memories efficiently.
        
        Args:
            memories: List of (memory_id, content, embedding) tuples
            
        Returns:
            Dictionary mapping (index1, index2) to (relationship_type, metadata)
        """
        relationships = {}
        n = len(memories)

        for i in range(n):
            for j in range(i + 1, n):
                _, content1, emb1 = memories[i]
                _, content2, emb2 = memories[j]

                relationship, metadata = self.detect_relationship(
                    emb1, emb2, content1, content2
                )

                # Only store non-unrelated relationships
                if relationship != "unrelated":
                    relationships[(i, j)] = (relationship, metadata)

        return relationships

    def analyze_version_chain(
        self,
        versions: list[tuple[str, str, list[float]]]
    ) -> dict[str, Any]:
        """Analyze a complete version chain for patterns and insights.
        
        Args:
            versions: List of (memory_id, content, embedding) tuples in chronological order
            
        Returns:
            Dictionary with analysis results including:
            - total_versions: Number of versions
            - evolution_pattern: Overall pattern (gradual, sudden, oscillating)
            - similarity_trend: List of similarities between consecutive versions
            - major_changes: Indices of major changes
            - contradictions: List of contradictory version pairs
        """
        if len(versions) < 2:
            return {
                "total_versions": len(versions),
                "evolution_pattern": "single_version",
                "similarity_trend": [],
                "major_changes": [],
                "contradictions": []
            }

        # Compute similarities between consecutive versions
        similarity_trend = []
        for i in range(len(versions) - 1):
            _, content1, emb1 = versions[i]
            _, content2, emb2 = versions[i + 1]
            
            similarity = self.compute_similarity(emb1, emb2)
            similarity_trend.append(similarity)

        # Identify major changes (low similarity)
        major_changes = [
            i for i, sim in enumerate(similarity_trend) 
            if sim < 0.6
        ]

        # Determine evolution pattern
        if not similarity_trend:
            pattern = "single_version"
        elif all(sim >= 0.7 for sim in similarity_trend):
            pattern = "gradual"
        elif any(sim < 0.4 for sim in similarity_trend):
            pattern = "sudden"
        else:
            # Check for oscillation (high variance)
            if len(similarity_trend) >= 3:
                variance = np.var(similarity_trend)
                pattern = "oscillating" if variance > 0.1 else "moderate"
            else:
                pattern = "moderate"

        # Find contradictions in the chain
        contradictions = []
        for i in range(len(versions)):
            for j in range(i + 1, len(versions)):
                _, content1, emb1 = versions[i]
                _, content2, emb2 = versions[j]
                
                is_contra, confidence = self.detect_contradiction(
                    content1, content2, emb1, emb2
                )
                
                if is_contra and confidence > 0.6:
                    contradictions.append({
                        "version_indices": (i, j),
                        "confidence": confidence
                    })

        return {
            "total_versions": len(versions),
            "evolution_pattern": pattern,
            "similarity_trend": similarity_trend,
            "major_changes": major_changes,
            "contradictions": contradictions,
            "avg_similarity": float(np.mean(similarity_trend)) if similarity_trend else 1.0,
            "min_similarity": float(np.min(similarity_trend)) if similarity_trend else 1.0,
            "max_similarity": float(np.max(similarity_trend)) if similarity_trend else 1.0
        }
