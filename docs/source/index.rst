Cuery: A Python Library for LLM Prompt Management
=================================================

Cuery is a Python library for LLM prompt management that extends the capabilities of the Instructor library. It provides a structured approach to working with prompts, contexts, response models, and tasks for effective LLM workflow management.

Key Concepts
------------

1. Prompts
~~~~~~~~~~

In Cuery, prompts are represented as a series of messages with built-in support for:

- **Jinja templating**: Dynamically generate content using template variables
- **YAML configuration**: Load prompts from YAML files for better organization
- **Iteration over contexts**: Process multiple contexts (see below) asynchronously or synchronously
- **Validation**: Ensure that required variables are provided before execution
- **Pretty printing**: Display prompts in a human-readable format

.. code-block:: python

    # Create a prompt manually
    prompt = Prompt(
        messages=[
            Message(role="system", content="You are an analyst identifying AI-automatable jobs."),
            Message(role="user", content="Analyze jobs in {{sector}} focused on {{main_activity}}.")
        ],
        required=["sector", "main_activity"]
    )

    # Load prompts from YAML configuration with direct, deep config access
    from cuery.prompt import Prompt
    prompt = Prompt.from_config("work/prompts.yaml:dirce_jobs")

2. Contexts
~~~~~~~~~~~

Contexts are collections of named variables (mappings/dicts) used to fill in Jinja templates in prompts. Contexts can be created from various data sources:

- **Pandas DataFrames**: Each row becomes a separate context
- **Dictionaries of iterables**: Values are aligned to create multiple contexts
- **Lists of dictionaries**: Each dictionary represents a separate context

.. code-block:: python

    # Create contexts from a DataFrame with Spanish industry sectors
    df = pd.DataFrame({
        "Division": ["Actividades informáticas", "Comunicación y marketing", "Administración"],
        "Actividad_principal": ["Programación y desarrollo", "Relaciones públicas", "Gestión documental"]
    })
    contexts, count = contexts_from_dataframe(df, required=["Division", "Actividad_principal"])

    # Each context will be used to analyze AI-automatable jobs in those sectors

3. ResponseModels
~~~~~~~~~~~~~~~~~

ResponseModels are Pydantic models that define the structure of LLM outputs, providing:

- **Structured parsing**: Convert LLM text responses to strongly typed objects
- **Validation**: Ensure outputs meet expected formats and constraints
- **Fallback handling**: Gracefully handle parsing errors
- **YAML configuration**: Load response models from configuration files
- **Pretty printing**: Display response models in a human-readable format

.. code-block:: python

    class Job(ResponseModel):
        name: str = Field(
            description="Name of the job role (job title, less than 50 characters)",
            min_length=5,
            max_length=50,
        )
        description: str = Field(
            description="A short description of the job role (less than 200 characters)",
            min_length=20,
            max_length=200,
        )
        automation_potential: int = Field(
            description="A score from 1 to 10 indicating the job's potential for automation",
            ge=0,
            le=10,
        )
        
    # Multi-output response model (1 context -> N outputs)
    class Jobs(ResponseModel):
        jobs: list[Job]
        
    # Create from configuration
    from cuery.response import ResponseModel
    response_model = ResponseModel.from_config("work/models.yaml:Job")

4. Tasks
~~~~~~~~

Tasks combine prompts and response models into reusable units of work, simplifying:

- **Execution across LLM providers**: Run the same task on different LLM backends
- **Batch processing**: Handle multiple contexts efficiently
- **Concurrency control**: Process requests in parallel with customizable limits
- **Response post-processing**: Transform and normalize structured outputs
- **Error handling**: Manage exceptions and fallback scenarios
- **Pretty printing**: Display task details in a human-readable format

.. code-block:: python

    # Create a task for finding automatable jobs in Spanish industry sectors
    client = instructor.from_openai(AsyncOpenAI())
    task = Task(
        prompt=Prompt.from_config("work/prompts.yaml:dirce_jobs"),
        response=Jobs,
        client=client,
        model="gpt-4-turbo"
    )

    # Execute with single context
    result = await task({
        "Division": "Actividades informáticas", 
        "Actividad_principal": "Programación y desarrollo"
    })

    # Execute with multiple contexts (automatic mode selection)
    results = await task(
        context=df,  # DataFrame with industry sectors
        n_concurrent=5  # Process 5 requests concurrently
    )

    # Post-process multi-output responses
    flattened_df = dirce_jobs_task.explode_responses(results, df)

Getting Started
---------------

.. code-block:: python

    import instructor
    from openai import AsyncOpenAI
    from cuery.task import Task
    from cuery.prompt import Prompt
    from pydantic import Field
    from cuery.work.tasks import Jobs, Job

    client = instructor.from_openai(AsyncOpenAI())

    prompt = Prompt(
        messages=[
            {"role": "system", "content": "Identify jobs automatable by AI in this sector."},
            {"role": "user", "content": "Sector: {{ sector }}\nActivity: {{ activity }}"}
        ],
        required=["sector", "activity"]
    )

    job_analyzer = Task(prompt=prompt, response=Jobs, client=client)

    result = await job_analyzer({
        "Division": "Actividades financieras", 
        "Actividad_principal": "Contabilidad y auditoría"
    })

Building on Instructor
----------------------

Cuery extends the Instructor library with higher-level abstractions for managing prompts and responses in a structured way, with particular emphasis on:

- Batch processing and concurrency management
- Context validation and transformation
- Multi-output response handling and normalization
- Configuration-based workflow setup

By providing these abstractions, Cuery aims to simplify the development of complex LLM workflows while maintaining the type safety and structured outputs that Instructor provides.

Documentation
-------------

Cuery uses `Sphinx <https://sphinx-autoapi.readthedocs.io/en/latest/>`_ with the `AutoApi extension <https://sphinx-autoapi.readthedocs.io/en/latest/index.html>`_ and the `PyData theme <https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html>`_.

To build and render:

.. code-block:: bash

    (cd docs && uv run make clean html)
    (cd docs/build/html && uv run python -m http.server)

.. toctree::
   :maxdepth: 2
   :caption: Contents:
