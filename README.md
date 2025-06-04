# Cuery

Cuery is a Python library for LLM prompting that extends the capabilities of the Instructor library. It provides a structured approach to working with prompts, contexts, response models, and tasks for effective LLM workflow management. It's main motivation is to make it easier to iterate prompts over tabular data (DataFrames).

## To Do
- Integrate web search API:
  - Depends on Instructor integration of OpenAI Responses API
  - PR: https://github.com/567-labs/instructor/pull/1520
- Seperate retry logic for rate limit errors and structured output validation errors
  - Issue: https://github.com/567-labs/instructor/issues/1503
- Prompt registry


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
    Cuery automatically saves the raw response from the LLM as an attribute of the (structured) Response. This means one can later inspect the number of tokens used e.g.and calculate it's cost in dollars. 
- **Automatic unwrapping of multivalued responses**  
  We can inspect if a response is defined as having a single field that is a list (i.e. we're asking for a multivalued response). In this case cuery can automatically handle things like unwrapping the list into separate output rows.

A `ResponseSet` further encapsulates a number of individual `Response` objects. This can be used e.g. to automatically convert a list of reponses to a DataFrame, to calculate the overall cost of having iterated a prompt over N inputs etc.

```python
from cuery import Field, Prompt, Response, Task


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

task = Task(prompt=prompt, response=MovieRecommendation)
result = await task(context)
print(result)
print(result.to_pandas())
```

```
[
    MovieRecommendation(title='The Imitation Game', year=2014, genre=['Biography', 'Drama', 'Thriller'], rating=8.0),
    MovieRecommendation(title='Blade Runner', year=1982, genre=['Sci-Fi', 'Thriller'], rating=8.2)
]


              topic     audience               title  year  \
0  Machine Learning    beginners  The Imitation Game  2014   
1   Computer Vision  researchers        Blade Runner  1982   

                          genre  rating  
0  [Biography, Drama, Thriller]     8.0  
1            [Sci-Fi, Thriller]     8.2  
```

Note how the input variables that have results in each response (`topic`, `audience`) are automatically included in the DataFrame representation. This can be useful to join the responses back to an original DataFrame that had more columns then were necessary for the prompt.

### 4. Tasks

Tasks combine prompts and response models into reusable units of work, simplifying:

- **Execution across LLM providers**: Run the same task on different LLM backends
- **Batch processing**: Handle multiple contexts efficiently
- **Concurrency control**: Process requests in parallel with customizable limits
- **Response post-processing**: Transform and normalize structured outputs
- **Task chaining**: Link multiple tasks together to create workflows

```python
from cuery.task import Task, Chain
from openai import AsyncOpenAI
import instructor

# Create a task
client = instructor.from_openai(AsyncOpenAI())
movie_task = Task(
    prompt=prompt,
    response=MovieRecommendations,
    client=client,
    model="gpt-4-turbo"
)

# Execute with different modes based on context type
# Single context
result = await movie_task({"genre": "sci-fi", "mood": "thoughtful"})

# Multiple contexts (automatic mode selection)
results = await movie_task(
    context=df,  # DataFrame or list of contexts
    n_concurrent=5  # Process 5 requests concurrently
)

# Post-process multi-output responses (1:N relationships)
flattened_df = movie_task.explode_responses(results, df)

# Chain multiple tasks together
movie_chain = Chain(movie_task, other_task)
final_results = await movie_chain(df, model="gpt-4-turbo", n_concurrent=5)
```

Tasks automatically select the appropriate execution mode (single, sequential, or parallel) based on the context type, and can be chained together to create multi-step workflows.

## Getting Started

```python
import instructor
from openai import AsyncOpenAI
from cuery.task import Task
from cuery.prompt import Prompt
from cuery.response import ResponseModel
from pydantic import Field

# Set up client
client = instructor.from_openai(AsyncOpenAI())

# Define response model
class Entity(ResponseModel):
    name: str
    type: str
    
class NamedEntities(ResponseModel):
    entities: list[Entity]

# Create prompt
prompt = Prompt(
    messages=[
        {"role": "system", "content": "Extract named entities from the text."},
        {"role": "user", "content": "Text: {{ text }}"}
    ],
    required=["text"]
)

# Create task
entity_extractor = Task(prompt=prompt, response=NamedEntities, client=client)

# Execute task
result = await entity_extractor({"text": "Apple is headquartered in Cupertino, California."})
```

## Building on Instructor

Cuery extends the Instructor library with higher-level abstractions for managing prompts and responses in a structured way, with particular emphasis on:

- Batch processing and concurrency management
- Context validation and transformation
- Multi-output response handling and normalization
- Configuration-based workflow setup

By providing these abstractions, Cuery aims to simplify the development of complex LLM workflows while maintaining the type safety and structured outputs that Instructor provides.

# Docs

Cuery uses [Sphinx](https://sphinx-autoapi.readthedocs.io/en/latest/) with the [AutoApi extension](https://sphinx-autoapi.readthedocs.io/en/latest/index.html) and the [PyData theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html).

To build and render:

``` bash
(cd docs && uv run make clean html)
(cd docs/build/html && uv run python -m http.server)
```