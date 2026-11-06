# Preserve the historical actor's Python 3.13 runtime family.
FROM apify/actor-python:3.13

USER root

RUN pip install --no-cache-dir uv==0.13.5
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
# Cuery comes from this exact source checkout, not PyPI or private Prefix.
RUN uv sync --frozen --no-dev --extra seo --no-editable --python /usr/local/bin/python

RUN (id -u apify >/dev/null 2>&1 || useradd --create-home apify) && chown -R apify:apify .
USER apify
CMD [".venv/bin/python", "-m", "cuery.actors.keywords"]
