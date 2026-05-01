# 🧠 AI Orchestrator (FastAPI + LangGraph)

This repository houses the asynchronous AI orchestration engine. It operates independently from the main web application, consuming events from the message broker to process heavy ML/AI tasks in the background.

> **Architectural Source of Truth:** > For system blueprints and capability maps, refer to the central [Engineering Platform Repository](https://github.com/abhishekchaturvedi07/engineering-platform).

---

## 🏗️ Tech Stack

- **Framework:** FastAPI (Python)
- **AI Orchestration:** LangChain & LangGraph (Cyclic Agentic Reasoning)
- **Vector Storage:** Pinecone / Milvus
- **Event Broker:** Kafka / RabbitMQ (Consumer & Publisher)
- **Embeddings:** OpenAI / HuggingFace

## 📂 Project Structure

- `/app/api` - FastAPI endpoints for synchronous AI health checks.
- `/app/agents` - LangGraph nodes, edges, and tool definitions.
- `/app/consumers` - Kafka/RabbitMQ background workers listening for processing jobs.
- `/app/vectorstore` - Connection logic for Pinecone/Vector databases.

ai-orchestrator/
├── memory/ # Layer 1: Rules & Context
│ ├── persona.md
│ └── architecture.rules.md
├── skills/ # Layer 2: Tools & Knowledge
│ ├── **init**.py
│ └── web_search.py
├── hooks/ # Layer 3: Guardrails
├── subagents/ # Layer 4: Delegation
├── agent.py # The Supervisor (LangGraph)
├── main.py
└── requirements.txt
