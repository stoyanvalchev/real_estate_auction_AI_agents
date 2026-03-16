[![Owner](https://img.shields.io/badge/Owner-stoyanvalchev-emeraldgreen)](https://github.com/stoyanvalchev)
![Python](https://img.shields.io/badge/Python-3.10–3.13-blue?logo=python&logoColor=white)
![CrewAI](https://img.shields.io/badge/Powered%20by-CrewAI-orange)
![Ollama](https://img.shields.io/badge/Model-llama3.1%3A8b-purple?logo=ollama)

# 🏠 Real Estate Auction AI Agents

An AI-powered system for **real estate property discovery and auction simulation** using autonomous agents. The dataset consists of **50 LLM-generated properties** located in Sofia, Bulgaria.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Using Local Models with Ollama](#using-local-models-with-ollama)
- [Running the Project](#running-the-project)
- [Property Search Mode](#property-search-mode)
- [Auction Simulation Mode](#auction-simulation-mode)

---

## Overview

This project combines a **RAG pipeline** for natural-language property search with an **AI agent auction simulator**, where buyer agents compete for properties based on their budgets, preferences, and bidding strategies — all running locally via Ollama, with no API costs.

---

## Installation

Ensure you have **Python >=3.10, <3.14** installed. This project uses [UV](https://docs.astral.sh/uv/) for dependency management.

**1. Install UV:**

```bash
pip install uv
```

**2. Navigate to the project directory and install dependencies:**

```bash
crewai install
```

> This will lock and install all required dependencies automatically.

---

## Using Local Models with Ollama

This project runs entirely on the **local Ollama model `llama3.1:8b`**, meaning no API keys are required and the system works fully offline.

### 1. Install Ollama

Download and install Ollama from [ollama.com](https://ollama.com), then verify the installation:

```bash
ollama --version
```

### 2. Pull the Required Model

```bash
ollama pull llama3.1:8b
```

> This may take a few minutes depending on your connection speed.

### 3. Create a `.env` File

> ⚠️ **This step is required.** The project will not run without a `.env` file in the root directory.

Create a `.env` file in the root of the project and set the following variables:

```env
MODEL=ollama/llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
```

### Hardware Requirements

| Spec | Requirement |
|------|-------------|
| RAM (minimum) | 8 GB |
| RAM (recommended) | 16 GB |
| GPU | Optional — improves performance significantly |
| CPU | Supported, but slower |

---

## Running the Project

From the root folder of your project, run:

```bash
crewai run
```

You will be prompted to choose a mode:

| Input | Mode | Description |
|-------|------|-------------|
| `s` or `search` | **Property Search** | Find properties using natural language |
| `a` or `auction` | **Auction Simulation** | AI buyer agents compete for a selected property |

---

## Property Search Mode

Describe the property you are looking for in plain English and the system will retrieve matching listings using the RAG pipeline.

**Example query:**

```
apartment in Geo Milev under 130 000 EUR
```

Results are returned with explanations and saved to `data/search_results/search_result.json`.

---

## Auction Simulation Mode

In Auction Mode, AI buyer agents evaluate a selected property and place competitive bids based on their individual budgets, preferences, and strategies.

### Configuration

Auction parameters can be customised in `src/real_estate_auction/crews/auction_crew/config/auction.yaml`:

| Parameter | Description |
|-----------|-------------|
| `min_increment` | Minimum bid increase between rounds |
| `max_rounds` | Maximum number of auction rounds |
| `property_to_auction` | ID or name of the property being auctioned |

### Results

The full auction history is saved to `data/auction_results/auction_result.json`. 