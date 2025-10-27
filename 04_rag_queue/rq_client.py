import redis
from rq import Queue
from dotenv import load_dotenv
import os

load_dotenv()

# Redis connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0))
)

# Create RQ queue
queue = Queue(connection=redis_client)
