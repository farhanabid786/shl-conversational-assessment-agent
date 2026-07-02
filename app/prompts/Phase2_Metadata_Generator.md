You are a Senior Python Backend Engineer and Data Engineer.

Project

SHL Conversational Assessment Recommendation System

Current Phase

Phase 2

Metadata Generation

Your task is ONLY to generate

scripts/metadata_generator.py

Do NOT generate any other file.

----------------------------------------

INPUT

data/processed/catalog_clean.json

(read only)

----------------------------------------

OUTPUT

data/processed/catalog_metadata.json

----------------------------------------

Purpose

Create a retrieval-optimized metadata layer for BM25 and FAISS.

The original catalog_clean.json must NEVER be modified.

----------------------------------------

Generate one metadata object for every assessment.

Each metadata object must contain EXACTLY these fields.

{
  "entity_id": "",
  "canonical_name": "",
  "normalized_name": "",
  "assessment_family": "",
  "keywords": [],
  "job_levels": [],
  "languages": [],
  "duration_minutes": null,
  "adaptive": false,
  "remote": false,
  "searchable_text": "",
  "filter_tokens": [],
  "ranking_tokens": [],
  "metadata_version": "1.0"
}

----------------------------------------

Rules

canonical_name

Keep original name.

----------------------------------------

normalized_name

Lowercase.

Remove punctuation.

Collapse multiple spaces.

----------------------------------------

assessment_family

Derive from the existing "keys" field.

Do not invent new categories.

----------------------------------------

keywords

Extract meaningful unique keywords from searchable_text.

Remove common English stop words.

Lowercase everything.

----------------------------------------

searchable_text

Reuse the existing search_text field.

Never regenerate it.

----------------------------------------

filter_tokens

Combine

assessment_family

job_levels

languages

adaptive

remote

Remove duplicates.

----------------------------------------

ranking_tokens

Combine

normalized_name

keywords

Remove duplicates.

----------------------------------------

Validation

Fail immediately if

- duplicate entity_id
- duplicate canonical_name
- empty searchable_text
- invalid boolean values
- duration_minutes is neither integer nor null

Raise descriptive custom exceptions.

----------------------------------------

Requirements

Python 3.10.11

Production Ready

Type Hints

Dataclasses where appropriate

Logging

Custom Exceptions

Modular Functions

PEP8

UTF-8 Output

Pretty JSON

Indent = 2

Never modify the input file.

Return ONLY the contents of

scripts/metadata_generator.py

No explanations.

No markdown.