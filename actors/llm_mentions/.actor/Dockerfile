# See Docker images from Apify at https://hub.docker.com/r/apify/.
FROM apify/actor-python:3.13

# Install uv and project (local) so cuery new actor code is available
RUN pip install uv

# First install runtime deps needed by the actor
RUN uv pip install --system "httpx~=0.27" "remove-accents~=0.1" "rapidfuzz~=3.9"

# Copy repository source (actor.json sets dockerContextDir to repo root)
COPY . ./

# Install local package to ensure cuery.actors.llm_mentions is present
RUN uv pip install --system .

RUN useradd --create-home apify && chown -R apify:apify .
USER apify

CMD ["python", "-m", "cuery.actors.llm_mentions"]


