# ai-orchestrator/worker.py
import pika
import json
import os
import time

# Use the Docker environment variable, fallback to localhost for standalone testing
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")

def process_document_task(ch, method, properties, body):
    """
    This is the function that runs every time a message arrives in the queue.
    """
    try:
        # 1. Parse the incoming JSON event from the Node.js BFF
        event_data = json.loads(body)
        document_id = event_data.get("document_id")
        file_path = event_data.get("file_path")
        
        print(f"📥 [PYTHON WORKER] Received task for document: {document_id}")
        print(f"⏳ [PYTHON WORKER] Processing file at: {file_path}")

        # ---------------------------------------------------------
        # 2. YOUR AI LOGIC GOES HERE
        # This is where you would call LangChain to:
        # - Load the PDF using PyPDFLoader
        # - Split the text using RecursiveCharacterTextSplitter
        # - Embed it using OpenAI/Nomic
        # - Save it to ChromaDB
        # ---------------------------------------------------------
        
        # Simulating heavy AI processing time (e.g., 5 seconds)
        time.sleep(5)
        
        print(f"✅ [PYTHON WORKER] Successfully vectorized and stored: {document_id}")

        # 3. Acknowledge the message so RabbitMQ removes it from the queue
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print(f"❌ [PYTHON WORKER] Error processing document: {e}")
        # Negative Acknowledge: Puts the message back in the queue to try again
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)


def start_worker():
    """
    Connects to RabbitMQ and starts listening indefinitely.
    """
    print(f"🔌 [PYTHON WORKER] Connecting to RabbitMQ at {RABBITMQ_URL}...")
    
    # Standard connection logic for pika
    parameters = pika.URLParameters(RABBITMQ_URL)
    
    # Adding a retry loop in case RabbitMQ is slow to boot up in Docker
    while True:
        try:
            connection = pika.BlockingConnection(parameters)
            break
        except pika.exceptions.AMQPConnectionError:
            print("⏳ [PYTHON WORKER] Waiting for RabbitMQ to start...")
            time.sleep(3)

    channel = connection.channel()

    # Ensure the queue exists (matches the Node.js BFF exactly)
    channel.queue_declare(queue='document_ingestion_queue', durable=True)

    # Tell RabbitMQ to only give us 1 message at a time (Fair Dispatch)
    channel.basic_qos(prefetch_count=1)

    # Subscribe to the queue
    channel.basic_consume(
        queue='document_ingestion_queue', 
        on_message_callback=process_document_task
    )

    print('🎧 [PYTHON WORKER] Listening for AI ingestion tasks. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == '__main__':
    start_worker()