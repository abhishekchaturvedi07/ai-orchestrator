# Routing & Scalability

- **Semantic Router**: Uses cosine similarity via LLM to choose graph nodes.
- **Notifications**: DO NOT call email APIs directly. Use the `publish_to_queue` skill to push to RabbitMQ.
- **Guardrails**: Always implement a keyword-check before the LLM router to save tokens and ensure 100% accuracy for common queries.
