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