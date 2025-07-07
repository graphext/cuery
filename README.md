# Cuery

[Cuery](https://cuery.readthedocs.io/en/latest/) is a Python library for LLM prompting that extends the capabilities of the Instructor library. It provides a structured approach to working with prompts, contexts, response models, and tasks for effective LLM workflow management. It's main motivation is to make it easier to iterate prompts over tabular data (DataFrames).

## Quick start

```python
from cuery import Prompt, Response, Task


# Define the desired structure of LLM responses
class Entity(Response):
    name: str
    type: str


class NamedEntities(Response):
    entities: list[Entity]


# Data to iterate prompt over (DataFrame, list[dict] or dict[str, list])
context = pd.DataFrame({
    "text": [
        "Apple is headquartered in Cupertino, California."
        "Barack Obama was the 44th President of the United States.",
        "The Eiffel Tower is located in Paris, France.",
    ]}
)

# Iterate the prompt over DataFrame rows using n concurrent async tasks
# and using the specified provider/model
prompt = Prompt.from_string("Extract named entities from the following text: {{text}}")
task = Task(prompt=prompt, response=NamedEntities)
result = await task(context, model="openai/gpt-3.5-turbo", n_concurrent=10)

# Get reuslt back as DataFrame containing both inputs and output columns
print(result.to_pandas(explode=True))
```

```
Gathering responses: 100%|██████████| 3/3 [00:01<00:00,  2.15it/s]

                                                text           name  \
0   Apple is headquartered in Cupertino, California.          Apple   
1   Apple is headquartered in Cupertino, California.      Cupertino   
2   Apple is headquartered in Cupertino, California.     California   
3  Barack Obama was the 44th President of the Uni...   Barack Obama   
4  Barack Obama was the 44th President of the Uni...           44th   
5  Barack Obama was the 44th President of the Uni...  United States   
6      The Eiffel Tower is located in Paris, France.   Eiffel Tower   
7      The Eiffel Tower is located in Paris, France.          Paris   
8      The Eiffel Tower is located in Paris, France.         France   

           type  
0  Organization  
1      Location  
2      Location  
3        Person  
4       Ordinal  
5      Location  
6      Location  
7      Location  
8      Location  
```

## Key Concepts

### Prompts

In Cuery, a `Prompt` is a class encapsulating a series of messages (user, system, etc.). Prompt messages support:

- **Jinja templating**  
    Dynamically generate content using template variables
- **Template variable validation**  
    Detects and validates that contexts used to render the final prompt contain the required variables
- **YAML configuration**  
    Load prompts from YAML files for better organization, using [glom](https://glom.readthedocs.io/en/latest/) for path-based access to nested objects
- **Pretty print**  
    Uses Rich to create pretty representations of prompts

```python
from cuery import Prompt, pprint

# Load prompts from nested YAML configuration
prompt = Prompt.from_config("work/config.yaml:prompts[0]")

# Create prompt from string
prompt = Prompt.from_string("Explain {{ topic }} in simple terms.")
pprint(prompt)

# Create a prompt manually
prompt = Prompt(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain {{ topic }} in simple terms."}
    ],
    required=["topic"]
)
```

```
╭───────────────────────── Prompt ─────────────────────────╮
│                                                          │
│  Required: ['topic']                                     │
│                                                          │
│ ╭──────────────────────── USER ────────────────────────╮ │
│ │ Explain {{ topic }} in simple terms.                 │ │
│ ╰──────────────────────────────────────────────────────╯ │
╰──────────────────────────────────────────────────────────╯
```

### Contexts

`Contexts` are collections of named variables used to render Jinja templates in prompts. There is not specific class for contexts, but where they are expected, they can be provided in various forms:

- **Pandas DataFrames**  
    Each column will be associated with a prompt variable, and each row becomes a separate context. Since prompts know which variables are required, extra columns will be
    ignored automatically. Prompts will be iterated over the rows, and will return one output for each input row.
- **Dictionaries of iterables**  
    Each key corresponds to a prompt variable, and the prompt will be iterated over the values (all iterables need to be of same length)
- **Lists of dictionaries**  
    Each dictionary in the list represents a separate context. The dictionary keys will
    be mapped to prompt variables, and the prompt will return one output for each input
    dict.

```python
import pandas as pd
from cuery.context import iter_context

df = pd.DataFrame({
    "topic": ["Machine Learning", "Natural Language Processing", "Computer Vision"],
    "audience": ["beginners", "developers", "researchers"]
})

contexts, count = iter_context(df, required=["topic", "audience"])
next(contexts)
```

```
>> {'topic': 'Machine Learning', 'audience': 'beginners'}
```

Cuery validates contexts against the required variables specified in the prompt, ensuring all necessary data is available before execution.

### Responses

A `Response` is Pydantic model that defines the structure of a desired LLM output, providing:

- **Structured parsing and validation**  
    Converts LLM text responses to strongly typed objects, ensuring outputs meet expected formats and constraints
- **Fallback handling**  
    Retries N times while validation fails providing the LLM with corresponding error messages. If _all_ retries fail, allows specification of a fallback (a `Response`
    will all values set to `None` by default.) to return instead of raising an exception. This allows iterating over hundreds or thousands of inputs without risk of losing all
    responses if only one or a few fail.
- **YAML configuration**  
    Load response models from configuration files (though that excludes the ability to
    write custom python validators).
- **Caching of _raw_ response**  
    `Cuery` automatically saves the _raw_ response from the LLM as an attribute of the (structured) `Response`. This means one can later inspect the number of tokens used e.g., and calculate it's cost in dollars. 
- **Automatic unwrapping of multivalued responses**  
  We can inspect if a response is defined as having a single field that is a list (i.e. we're asking for a multivalued response). In this case cuery can automatically handle things like unwrapping the list into separate output rows.

A `ResponseSet` further encapsulates a number of individual `Response` objects. This can be used e.g. to automatically convert a list of reponses to a DataFrame, to calculate the overall cost of having iterated a prompt over N inputs etc.

```python
from cuery import Field, Prompt, Response, Task

# Simple response model
class MovieRecommendation(Response):
    title: str
    year: int = Field(gt=1900, lt=2030)
    genre: list[str]
    rating: float = Field(ge=0, le=10)


# Multi-output response model
class MovieRecommendations(Response):
    recommendations: list[MovieRecommendation]

prompt = Prompt.from_string("Recommend a movie for {{ audience }} interested in {{ topic }}.")

context = [
    {"topic": "Machine Learning", "audience": "beginners"},
    {"topic": "Computer Vision", "audience": "researchers"},
]

task = Task(prompt=prompt, response=MovieRecommendations)
result = await task(context)
print(result)
print(result.to_pandas())
```

```
[
    MovieRecommendations(recommendations=[MovieRecommendation(title='The Matrix', year=1999, genre=['Action', 'Sci-Fi'], rating=8.7), MovieRecommendation(title='Ex Machina', year=2014, genre=['Drama', 'Sci-Fi'], rating=7.7), MovieRecommendation(title='Her', year=2013, genre=['Drama', 'Romance', 'Sci-Fi'], rating=8.0)]),
    MovieRecommendations(recommendations=[MovieRecommendation(title='Blade Runner 2049', year=2017, genre=['Sci-Fi', 'Thriller'], rating=8.0), MovieRecommendation(title='Ex Machina', year=2014, genre=['Drama', 'Sci-Fi'], rating=7.7), MovieRecommendation(title='Her', year=2013, genre=['Drama', 'Romance', 'Sci-Fi'], rating=8.0)])
]


              topic     audience              title  year  \
0  Machine Learning    beginners         The Matrix  1999   
1  Machine Learning    beginners         Ex Machina  2014   
2  Machine Learning    beginners                Her  2013   
3   Computer Vision  researchers  Blade Runner 2049  2017   
4   Computer Vision  researchers         Ex Machina  2014   
5   Computer Vision  researchers                Her  2013   

                      genre  rating  
0          [Action, Sci-Fi]     8.7  
1           [Drama, Sci-Fi]     7.7  
2  [Drama, Romance, Sci-Fi]     8.0  
3        [Sci-Fi, Thriller]     8.0  
4           [Drama, Sci-Fi]     7.7  
5  [Drama, Romance, Sci-Fi]     8.0 
```

Note how the input variables that have resulted in each response (`topic`, `audience`) are automatically included in the DataFrame representation. This makes it easy to see what the LLM extracted for each input, and can be useful to join the responses back to an original DataFrame that had more columns then were necessary for the prompt. Also, by default, multivalued responses are "exploded" into separate rows, but this can be controlled via `result.to_pandas(explode=False)`.

``` python
print(result.usage())
```

This returns a DataFrame with the number of tokens used by the prompt and the completion, and if per-token costs are known by `cuery`, the responding amount in dollars:

```
   prompt  completion      cost
0     131          31  0.000112
1     131          26  0.000104
```

We can inspect the _raw_ responses like this:

```
print(result[0]._raw_response.model)
>> gpt-3.5-turbo-0125

print(result[0]._raw_response.service_tier)
>> default
```

And the _raw_ "query" like this:

```
print(movie_task.query_log.queries[0]["messages"][0]["content"])
>> Recommend a movie for beginners interested in Machine Learning.
```

Finally we can inspect the structure of responses with built-in pretty printing:

```
from cuery import pprint

pprint(result[0])
```

```
╭───────────── RESPONSE: MovieRecommendations ─────────────╮
│                                                          │
│ ╭─ recommendations: list[__main__.MovieRecommendation]─╮ │
│ │                                                      │ │
│ │  {'required': True}                                  │ │
│ │                                                      │ │
│ ╰──────────────────────────────────────────────────────╯ │
│                                                          │
│  ╭───────── RESPONSE: MovieRecommendation ──────────╮    │
│  │                                                  │    │
│  │ ╭─ title: str ─────────────────────────────────╮ │    │
│  │ │                                              │ │    │
│  │ │  {'required': True}                          │ │    │
│  │ │                                              │ │    │
│  │ ╰──────────────────────────────────────────────╯ │    │
│  │ ╭─ year: int ──────────────────────────────────╮ │    │
│  │ │                                              │ │    │
│  │ │  {                                           │ │    │
│  │ │      'required': True,                       │ │    │
│  │ │      'metadata': [                           │ │    │
│  │ │          Gt(gt=1900),                        │ │    │
│  │ │          Lt(lt=2030)                         │ │    │
│  │ │      ]                                       │ │    │
│  │ │  }                                           │ │    │
│  │ │                                              │ │    │
│  │ ╰──────────────────────────────────────────────╯ │    │
│  │ ╭─ genre: list[str] ───────────────────────────╮ │    │
│  │ │                                              │ │    │
│  │ │  {'required': True}                          │ │    │
│  │ │                                              │ │    │
│  │ ╰──────────────────────────────────────────────╯ │    │
│  │ ╭─ rating: float ──────────────────────────────╮ │    │
│  │ │                                              │ │    │
│  │ │  {                                           │ │    │
│  │ │      'required': True,                       │ │    │
│  │ │      'metadata': [Ge(ge=0), Le(le=10)]       │ │    │
│  │ │  }                                           │ │    │
│  │ │                                              │ │    │
│  │ ╰──────────────────────────────────────────────╯ │    │
│  │                                                  │    │
│  ╰──────────────────────────────────────────────────╯    │
│                                                          │
│                                                          │
╰──────────────────────────────────────────────────────────╯
```

### Tasks and Chains

A `Task` combines a prompt and a response model into reusable units of work, simplifying:

- **Execution across LLM providers and models**: Run the same task on different LLM backends
- **Concurrency control**: Process requests in parallel with customizable limits
- **Task chaining**: Link multiple tasks together to create workflows

E.g. given the movie task above:

```python
from typing import Literal
from cuery import Task, Chain

# Reuse example from above
movie_task = Task(prompt=movie_prompt, response=MovieRecommendations)

# Add PG rating
class Rating(Response):
    pg_category: Literal["G", "PG", "PG-13", "R", "NC-17"] = Field(..., description="PG rating of the movie.")
    pg_reason: str = Field(..., description="Reason for the rating.")


rating_prompt = Prompt.from_string("What is the PG rating for {{ title }}?")
rating_task = Task(prompt=rating_prompt, response=Rating)

# Create a chain of tasks, execute with "provider/modelname"
chain = Chain(movie_task, rating_task)
result = await chain(context, model="openai/gpt-3.5-turbo", n_concurrent=20)
print(result)
```

The return value of the chain is the result of successively joining each task's output DataFrame with the previous one, using the corresponding prompt's variables as the keys:

```
              topic     audience         title  year  \
0  Machine Learning    beginners    The Matrix  1999   
1  Machine Learning    beginners    Ex Machina  2014   
2  Machine Learning    beginners    Ex Machina  2014   
3  Machine Learning    beginners           Her  2013   
4  Machine Learning    beginners           Her  2013   
5   Computer Vision  researchers  Blade Runner  1982   
6   Computer Vision  researchers           Her  2013   
7   Computer Vision  researchers           Her  2013   
8   Computer Vision  researchers    Ex Machina  2014   
9   Computer Vision  researchers    Ex Machina  2014   

                       genre  rating pg_category  \
0           [Action, Sci-Fi]     8.7           R   
1            [Drama, Sci-Fi]     7.7           R   
2            [Drama, Sci-Fi]     7.7           R   
3   [Drama, Romance, Sci-Fi]     8.0           R   
4   [Drama, Romance, Sci-Fi]     8.0           R   
5         [Sci-Fi, Thriller]     8.1           R   
6   [Drama, Romance, Sci-Fi]     8.0           R   
7   [Drama, Romance, Sci-Fi]     8.0           R   
8  [Drama, Sci-Fi, Thriller]     7.7           R   
9  [Drama, Sci-Fi, Thriller]     7.7           R   

                                           pg_reason  
0                              Violence and language  
1  Strong language, graphic nudity, and sexual co...  
2  Rated R for graphic nudity, language, sexual r...  
3  Brief graphic nudity, Sexual content and language  
4  Language, sexual content and brief graphic nudity  
5          Violence, some language, and brief nudity  
6  Brief graphic nudity, Sexual content and language  
7  Language, sexual content and brief graphic nudity  
8  Strong language, graphic nudity, and sexual co...  
9  Rated R for graphic nudity, language, sexual r... 
```

# Building on Instructor

Cuery extends the Instructor library with higher-level abstractions for managing prompts and responses in a structured way, with particular emphasis on:

- Batch processing (of DataFrames) and concurrency management
- Context validation and transformation
- Multi-output response handling and normalization
- Configuration-based workflow setup

By providing these abstractions, Cuery aims to simplify the development of complex LLM workflows while maintaining the type safety and structured outputs that Instructor provides.

# Provider-model lists

- OpenAI: https://platform.openai.com/docs/models
- Google: https://ai.google.dev/gemini-api/docs/models
- Perplexity: https://docs.perplexity.ai/models/model-cards
- Anthropic: https://docs.anthropic.com/en/docs/about-claude/models/overview

# Development


Assuming you have [uv](https://docs.astral.sh/uv/) installed already:

```bash
# Clone the repository
git clone https://github.com/graphext/cuery.git
cd cuery

# Install dependencies
uv sync --all-groups --all-extras

# Set up pre-commit hooks
pre-commit install
```

# Publish

```bash
uv build
uv publish --token `cat /path/to/token.txt`
```

# Docs

Cuery uses [Sphinx](https://sphinx-autoapi.readthedocs.io/en/latest/) with the [AutoApi extension](https://sphinx-autoapi.readthedocs.io/en/latest/index.html) and the [PyData theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html).

To build and render:

``` bash
(cd docs && uv run make clean html)
(cd docs/build/html && uv run python -m http.server)
```


# To Do
- Integrate web search API:
  - Depends on Instructor integration of OpenAI Responses API
  - PR: https://github.com/567-labs/instructor/pull/1520
- Seperate retry logic for rate limit errors and structured output validation errors
  - Issue: https://github.com/567-labs/instructor/issues/1503
