# Architecture

                User

                  │

             POST /chat

                  │

      Conversation State Parser

                  │

         Decision Engine

      ├───────────────┐

      │               │

 Clarify         Recommend

      │               │

 Compare        Refine

      │               │

          Hybrid Retrieval

      FAISS + BM25 + Metadata

                  │

            Top Candidates

                  │

            Gemini Flash

                  │

          Structured JSON

                  │

              API Output