from fastapi import FastAPI
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel, Field

from cuery.topics.oneshot import TopicAssigner, TopicAssignment, TopicExtractor, Topics

app = FastAPI(
    title="Cuery Topic Extraction API",
    description="API for extracting and assigning topics using cuery.topics.oneshot module.",
    version="0.1.0",
)


class ExtractTopicsRequest(BaseModel):
    texts: list[str] = Field(..., description="A list of texts to extract topics from.")
    domain: str | None = Field(
        None,
        description="The domain of the texts (e.g., 'customer support tickets', 'product reviews').",
    )
    n_topics: int = Field(10, description="Desired number of top-level topics.")
    n_subtopics: int = Field(5, description="Desired number of subtopics per top-level topic.")
    extra: str | None = Field(
        None, description="Extra instructions for the topic extraction prompt."
    )
    model: str = Field(
        ..., description="The language model to use for extraction (e.g., 'openai/gpt-3.5-turbo')."
    )
    max_dollars: float | None = Field(
        None, description="Maximum cost in dollars for processing texts for topic extraction."
    )
    max_tokens: float | None = Field(
        None, description="Maximum number of tokens to process from texts for topic extraction."
    )
    max_texts: float | None = Field(
        None, description="Maximum number of texts to process for topic extraction."
    )


class AssignTopicsRequest(BaseModel):
    texts: list[str] = Field(..., description="A list of texts to assign topics to.")
    topics: Topics | dict[str, list[str]] = Field(
        ..., description="The topic hierarchy (JSON) to use for assignment."
    )
    model: str = Field(
        ..., description="The language model to use for assignment (e.g., 'openai/gpt-3.5-turbo')."
    )


@app.post("/extract_topics", operation_id="extract_topics")
async def extract_topics(request: ExtractTopicsRequest) -> Topics:
    """Extracts a topic hierarchy from a list of texts."""
    extractor = TopicExtractor(
        domain=request.domain,
        n_topics=request.n_topics,
        n_subtopics=request.n_subtopics,
        extra=request.extra,
    )

    return await extractor(
        texts=request.texts,
        model=request.model,
        max_dollars=request.max_dollars,
        max_tokens=request.max_tokens,
        max_texts=request.max_texts,
    )


@app.post("/assign_topics", operation_id="assign_topics")
async def assign_topics(request: AssignTopicsRequest) -> list[TopicAssignment]:
    """
    Assigns topics and subtopics to a list of texts using a predefined topic hierarchy.
    """
    assigner = TopicAssigner(topics=request.topics)
    responses = await assigner(texts={"text": request.texts}, model=request.model, n_concurrent=20)
    return list(responses)  # type: ignore


mcp = FastApiMCP(app)
mcp.mount()

# To run this server (example using uvicorn):
# uvicorn cuery.server.server:app --reload
