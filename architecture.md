# Architecture Document

## Overview

This project implements an AI-powered real estate auction system with two main user-facing capabilities:

1. **Property Search** – users can search across unstructured real estate listings using natural language.
2. **Auction Simulation** – AI buyer agents participate in a configurable auction process for selected properties.

The system combines a **data ingestion pipeline**, a **Retrieval-Augmented Generation (RAG) pipeline**, and a **multi-agent auction workflow** orchestrated with CrewAI.


## High-Level Architecture

```text
Unstructured Property Documents
        │
        ▼
   Document Loader
        │
        ▼
      Chunking
        │
        ▼
     Embeddings
        │
        ▼
   Vector Database
        │
        ▼
    RAG Retriever
        │
        ├──────────────► Property Search Chatbot
        │
        └──────────────► Buyer Agents
                               │
                               ▼
                     Auction Orchestrator
                               │
                               ▼
                        Auction Results