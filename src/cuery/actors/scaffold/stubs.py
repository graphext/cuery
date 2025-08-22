ACTOR_JSON = {
    "actorSpecification": 1,
    "name": "",
    "title": "",
    "description": "",
    "version": "1.0",
    "buildTag": "latest",
    "dockerfile": "./Dockerfile",
    "dockerContextDir": "../..",
    "input": "./input_schema.json",
    "storages": {"dataset": "./dataset_schema.json"},
    "main": "",
    "readme": "./README.md",
    "minMemoryMbytes": 256,
    "maxMemoryMbytes": 4096,
    "usesStandbyMode": False,
    "environmentVariables": {
        "OPENAI_API_KEY": "@OPENAI_API_KEY",
        "ANTHROPIC_API_KEY": "@ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY": "@GOOGLE_API_KEY",
    },
}


DOCKERFILE = """
# See Docker images from Apify at https://hub.docker.com/r/apify/.
FROM apify/actor-python:3.13

RUN pip install uv
RUN uv pip install --system {pin}

COPY . ./

RUN useradd --create-home apify && chown -R apify:apify .
USER apify

CMD ["python", "-m", "cuery.actors.{module_name}"]
""".lstrip()


README = """
# {title}

{description}

## How it works

- Fetches an Apify dataset by `dataset` (either an Apify ID or a Parquet file URL)
- Runs the configured FlexTool over each record
- Pushes the processed records to the default dataset
""".lstrip()


ACTOR_MAIN = """
import asyncio

from cuery.actors import {module_name}

asyncio.run({module_name}.main())

""".lstrip()


MODULE = """
import asyncio

from apify import Actor

from {module_name} import {class_name} as ToolClass

from .utils import run_flex_tool

async def main():
    async with Actor:
        await run_flex_tool(Actor, ToolClass, max_retries=MAX_RETRIES)


if __name__ == "__main__":
    asyncio.run(main())
""".lstrip()
