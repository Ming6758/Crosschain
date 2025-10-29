# Real-Time Blockchain Security AI: Multi-Agent Exploit Detector

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-brightgreen)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1%2B-orange)](https://langchain-ai.github.io/langgraph/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector-DB-purple)](https://www.pinecone.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-Passing-green)](https://github.com/yourusername/blockchain-security-ai/actions)

A multi-agent AI system for **real-time detection, classification, and response to blockchain exploits**. Built during my internship, it orchestrates specialized agents using **LangGraph** to monitor real-time events, classify threats with LLMs (OpenAI/Groq fallback), match against historical attacks via **Pinecone vector search**, and suggest remediations. Handles fast-moving threats like flash loans, token dumps, and governance attacks with sub-second latency.

This project demonstrates skills in **AI orchestration**, **vector databases**, **real-time systems**, and **blockchain integration**.

## 🎯 Features
- **Anomaly Detection**: Rules-based (extensible to ML) flagging of suspicious patterns.
- **Exploit Classification**: LLM-powered categorization (e.g., Flash Loan, Oracle Manipulation) with fallback reliability.
- **Historical Matching**: Pinecone vector search against a database of real exploits.
- **Remediation Advice**: Actionable recommendations like pausing contracts or alerting DAOs.
- **Human-Readable Reports**: Summarized incident reports with confidence scores and next steps.
- **Scalable Architecture**: FastAPI API + LangGraph for async processing; easy to extend to multi-chain.

## 🏗️ Architecture Overview
