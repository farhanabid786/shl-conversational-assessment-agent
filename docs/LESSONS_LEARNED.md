# Lessons Learned

Document

- Bugs fixed
- Retrieval improvements
- Prompt improvements
- Performance optimizations
- Evaluation observations

Update after every phase.

# Phase 1

Lessons

• Never edit the raw catalog.

• Generate search_text once during preprocessing.

• Normalize durations before embeddings.

• Normalize boolean values before retrieval.

• Build reusable preprocessing scripts.

• Validate the dataset before building embeddings.

• Freeze the knowledge base before metadata generation.

# Phase 2

Lessons Learned

- Keep canonical data immutable.
- Separate retrieval metadata from the knowledge base.
- Metadata should support retrieval rather than duplicate information.
- Normalize names before embedding.
- Generate reusable ranking tokens.
- Generate reusable filter tokens.
- Preserve entity_id across every phase.
- Validate metadata before writing.

# Phase 3

Lessons Learned

• Metadata should be embedded instead of the raw catalog.

• Normalize vectors before FAISS indexing.

• Keep entity ordering deterministic.

• Preserve embedding-to-entity mapping.

• Use BM25 only on ranking_tokens.

• Keep dense and sparse retrieval independent.

• Freeze retrieval assets after validation.

# Phase 4

Lessons Learned

• Separate retrieval from reasoning.

• Use Reciprocal Rank Fusion instead of score averaging.

• Keep retrieval deterministic.

• Metadata filtering should occur after fusion.

• Entity lookup should be O(1).

• Gemini should never retrieve directly.

• The LLM should operate only on retrieved candidates.

# Phase 5

Lessons Learned

- Keep deterministic logic outside the LLM.
- Intent detection should be rule-based before Gemini.
- Conversation state should be structured.
- Retrieval must finish before decision logic.
- Gemini should only format and reason over retrieved candidates.
- Refusal logic should not invoke retrieval.
- Prompt construction should remain isolated from business logic.

