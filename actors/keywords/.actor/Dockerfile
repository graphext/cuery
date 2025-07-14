# See Docker images from Apify at https://hub.docker.com/r/apify/.
FROM apify/actor-python:3.13

RUN pip install uv
RUN uv pip install --system cuery[seo]

COPY . ./

# Use compileall to ensure the runnability of the Actor Python code.
# RUN python3 -m compileall -q src/

# Create and run as a non-root user.
RUN useradd --create-home apify && chown -R apify:apify .
USER apify

CMD ["python", "-m", "cuery.actors.keywords"]
