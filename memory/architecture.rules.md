# Enterprise Tech Stack Guidelines

If a user asks a question about writing code or building architecture, strictly adhere to these internal tech stack rules:

1. **Frontend:** We exclusively use Next.js (App Router), TypeScript, and TailwindCSS. Do not suggest React (Create React App) or regular CSS.
2. **Backend:** We exclusively use Node.js for Identity/Auth and Python (FastAPI) for data/AI workloads.
3. **Event Mesh:** We use RabbitMQ for asynchronous messaging. Do not suggest Kafka unless explicitly asked.
4. **AI Models:** We prioritize local, open-source models (Llama 3) via LM Studio.
