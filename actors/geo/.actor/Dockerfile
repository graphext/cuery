# See Docker images from Apify at https://hub.docker.com/r/apify/.
FROM apify/actor-python:3.13

RUN pip install uv
RUN uv pip install --system "cuery[seo]==0.20.1"

COPY . ./

RUN useradd --create-home apify && chown -R apify:apify .
USER apify

CMD ["python", "-m", "cuery.actors.geo"]
