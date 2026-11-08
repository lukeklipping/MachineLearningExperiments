"""

    pip install fastapi uvicorn
    uvicorn rest_experiment:app --reload
    open http://127.0.0.1:8000/docs

One resource, "experiments", with the standard REST operations:
    POST   /experiments        create
    GET    /experiments        list (optional ?status= filter)
    GET    /experiments/{id}   read one
    PATCH  /experiments/{id}   partial update
    DELETE /experiments/{id}   delete
"""
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field

app = FastAPI(title="REST Experiment")

Status = Literal["pending", "running", "done"]


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    compression_method: str = Field(examples=["quantization", "wavelet"])
    notes: Optional[str] = None


class ExperimentUpdate(BaseModel):
    # Every field optional so PATCH can change just one thing.
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    compression_method: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[Status] = None


class Experiment(ExperimentCreate):
    id: str
    status: Status
    created_at: datetime


db: dict[str, Experiment] = {}


@app.post("/experiments", response_model=Experiment, status_code=201)
def create_experiment(body: ExperimentCreate):
    exp = Experiment(
        id=uuid.uuid4().hex,
        status="pending",
        created_at=datetime.now(timezone.utc),
        **body.model_dump(),
    )
    db[exp.id] = exp
    return exp


@app.get("/experiments", response_model=list[Experiment])
def list_experiments(status: Optional[Status] = Query(default=None)):
    items = list(db.values())
    if status:
        items = [e for e in items if e.status == status]
    return items


@app.get("/experiments/{exp_id}", response_model=Experiment)
def get_experiment(exp_id: str):
    if exp_id not in db:
        raise HTTPException(404, "not found.")
    return db[exp_id]


@app.patch("/experiments/{exp_id}", response_model=Experiment)
def update_experiment(exp_id: str, body: ExperimentUpdate):
    if exp_id not in db:
        raise HTTPException(404, "not found.")
    changes = body.model_dump(exclude_unset=True)
    db[exp_id] = db[exp_id].model_copy(update=changes)
    return db[exp_id]


@app.delete("/experiments/{exp_id}", status_code=204)
def delete_experiment(exp_id: str):
    if db.pop(exp_id, None) is None:
        raise HTTPException(404, "not found.")
    return Response(status_code=204)
