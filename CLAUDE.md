# AI Orchestrator Project Rules

- **Tech Stack**: FastAPI, LangGraph, ChromaDB, RabbitMQ (pika).
- **Core Principle**: Zero-Trust. Every endpoint must use `HTTPBearer` security.
- **Routing**: Intent detection is handled by the Supervisor node. Use "Guardrails" for deterministic keyword routing.
- **Scalability**: For actions like email or notifications, use the `publish_to_queue` pattern rather than direct API calls.
- **LLM**: Primary reasoning is performed via LM Studio (local) or OpenAI-compatible endpoints.
