# Blockchain Exploit Detector  
### Real-time AI-powered threat detection & automated response for DeFi and Web3

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange)](https://langchain-ai.github.io/langgraph/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-purple)](https://pinecone.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Detect, classify, and respond to blockchain exploits in under 1 second** using a multi-agent AI pipeline powered by LLMs, vector search, and event-driven architecture.

---

## The Problem

Blockchain exploits happen **fast** — flash loans, governance attacks, and token dumps can drain millions in minutes.  
Traditional security tools are **slow, manual, and reactive**.

---

## The Solution: Real-Time AI Security Agent

A **multi-agent AI system** that:
1. **Detects anomalies** in real time (high-value transfers, price impact, new wallets)
2. **Classifies exploit type** using LLMs (OpenAI + Groq fallback)
3. **Matches against known attacks** via Pinecone vector similarity
4. **Recommends immediate remediation** (pause contract, freeze multisig, etc.)
5. **Generates human-readable incident reports** for security teams

---

## System Architecture

```mermaid
graph TD
    A[Blockchain Event / Webhook] --> B[FastAPI /process]
    B --> C{Anomaly Detector}
    C -->|Suspicious| D[Exploit Classifier<br/>GPT-4o → Groq]
    D --> E[Attack Matcher<br/>Pinecone + OpenAI Embeddings]
    E --> F[Remediation Advisor]
    F --> G[Explainer Agent<br/>Final Report]
    G --> H[Security Team<br/>Slack / Email / Dashboard]
