# AI Orchestrator: Project Brain

- **Role**: Backend for Agentic RAG and Intent Routing.
- **Security**: Zero-Trust. Every endpoint requires a valid JWT signed with `enterprise-secret-2026`.
- **Core Stack**: FastAPI, LangGraph, ChromaDB, RabbitMQ.
- **Routing Logic**: Supervisor node detects intent and branches to DATABASE, WEB, or NOTIFICATION.
- **Guiding Principle**: Business logic (Security/Routing) must be deterministic (Guardrails).
