# Cuery

Cuery is a Python library for LLM prompt management that extends the capabilities of the Instructor library. It provides a structured approach to working with prompts, contexts, response models, and tasks for effective LLM workflow management.

## Key Concepts

### 1. Prompts

In Cuery, prompts are represented as a series of messages with built-in support for:

- **Jinja templating**: Dynamically generate content using template variables
- **YAML configuration**: Load prompts from YAML files for better organization
- **Iteration over contexts**: Process multiple contexts asynchronously or synchronously

```python
from cuery.prompt import Prompt, Message

# Create a prompt manually
prompt = Prompt(
    messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Explain {{ topic }} in simple terms.")
    ],
    required=["topic"]
)

# Load prompts from YAML configuration
from cuery.prompt import Prompt
prompt = Prompt.from_config("work/prompts.yaml")
```

### 2. Contexts

Contexts are collections of named variables used to fill in Jinja templates in prompts. Contexts can be created from various data sources:

- **Pandas DataFrames**: Each row becomes a separate context
- **Dictionaries of iterables**: Values are aligned to create multiple contexts
- **Lists of dictionaries**: Each dictionary represents a separate context

```python
import pandas as pd
from cuery.context import contexts_from_dataframe

# Create contexts from a DataFrame
df = pd.DataFrame({
    "topic": ["Machine Learning", "Natural Language Processing", "Computer Vision"],
    "audience": ["beginners", "developers", "researchers"]
})
contexts, count = contexts_from_dataframe(df, required=["topic", "audience"])

# Each context will be used to fill template variables in your prompt
```

Cuery validates contexts against the required variables specified in the prompt, ensuring all necessary data is available before execution.

### 3. ResponseModels

ResponseModels are Pydantic models that define the structure of LLM outputs, providing:

- **Structured parsing**: Convert LLM text responses to strongly typed objects
- **Validation**: Ensure outputs meet expected formats and constraints
- **Fallback handling**: Gracefully handle parsing errors
- **YAML configuration**: Load response models from configuration files

```python
from cuery.response import ResponseModel
from pydantic import Field

class MovieRecommendation(ResponseModel):
    title: str
    year: int = Field(gt=1900, lt=2030)
    genre: list[str]
    rating: float = Field(ge=0, le=10)
    
# Multi-output response model
class MovieRecommendations(ResponseModel):
    recommendations: list[MovieRecommendation]
    
# Create from configuration
from cuery.response import ResponseModel
response_model = ResponseModel.from_config("work/models.yaml", "movie_recommendations")
```

Cuery extends Instructor's response parsing with additional utilities for response handling, retries, and prompt refinement.

### 4. Tasks

Tasks combine prompts and response models into reusable units of work, simplifying:

- **Execution across LLM providers**: Run the same task on different LLM backends
- **Batch processing**: Handle multiple contexts efficiently
- **Concurrency control**: Process requests in parallel with customizable limits
- **Response post-processing**: Transform and normalize structured outputs

```python
from cuery.task import Task
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
```

Tasks automatically select the appropriate execution mode (single, sequential, or parallel) based on the context type.

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