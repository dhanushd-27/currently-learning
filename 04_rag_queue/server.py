from fastapi import FastAPI, Query
from queues.worker import process_query
from rq_client import queue
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/chat")
def chat(
    query: str = Query(..., description="The query to search the vector store")
):
  job = queue.enqueue(process_query, query)

  return { "status": "queued", "job_id": job.id }

@app.get("/job-status")
def dequeue(job_id: str = Query(..., description="The job_id to retrieve results for")):
    job = queue.fetch_job(job_id)
    if job is None:
        return {"status": "not_found", "result": None}
    if job.is_finished:
        return {"status": "finished", "result": job.result}
    elif job.is_failed:
        return {"status": "failed", "result": str(job.exc_info)}
    else:
        return {"status": "queued", "result": None}