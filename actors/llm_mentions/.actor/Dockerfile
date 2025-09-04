# See Docker images from Apify at https://hub.docker.com/r/apify/.
FROM apify/actor-python:3.13

# Install uv and project (local) so cuery new actor code is available
RUN pip install uv

# First install runtime deps needed by the actor
# Dependencies are provided by the local project install below

# Copy repository source (actor.json sets dockerContextDir to repo root)
COPY . ./

# Install local package with SEO extras and httpx used by this actor
RUN uv pip install --system ".[seo]" "httpx~=0.27"

RUN useradd --create-home apify && chown -R apify:apify .
USER apify

CMD ["python", "-m", "cuery.actors.llm_mentions"]


