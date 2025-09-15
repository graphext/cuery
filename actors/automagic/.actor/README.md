# Automagic Actor

The Automagic Actor is a fully automated tool that can extract and structure any type of information from data records using AI-generated schemas and custom instructions.

## How it works

1. **Automatic Schema Generation**: Based on your instructions, the actor first generates a JSON schema that defines the structure of the information you want to extract.

2. **Data Processing**: Using the generated schema and your instructions, the actor processes each record in your dataset to extract the specified information.

3. **Structured Output**: The results are returned as structured data following the automatically generated schema.

## Key Features

- **Fully Automatic**: No need to manually define schemas or response models
- **Flexible Instructions**: Describe what you want to extract in natural language
- **Custom Schema Control**: Optionally provide specific schema generation instructions
- **Multiple Record Formats**: Support for JSON, Markdown, and text record formats
- **Scalable**: Processes large datasets efficiently with concurrent execution

## Input Parameters

Use the following JSON keys when providing `input` to the actor (e.g. via API or local run).

| Key | Title (UI) | Required | Type | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `dataset` | Dataset | Yes | string | – | Apify dataset ID or URL to a parquet file containing records to process. |
| `instructions` | Processing Instructions | Yes | string | – | What to extract and how to process each record. Drives both schema generation and extraction. |
| `model` | LLM Model | No | string | `openai/gpt-3.5-turbo` | LLM provider/model for data processing (format `provider/model`). |
| `schema_model` | Schema Generation Model | No | string | `openai/gpt-4.1` | LLM provider/model used specifically to generate the JSON schema. |
| `response_schema` | Response Schema Instructions | No | string | – | Extra instructions constraining/overriding automatic schema generation. |
| `attrs` | Record Attributes | No | array[string] | – | Subset of record attribute names to include in prompts. All are used if omitted. |
| `record_format` | Record Format | No | enum(`text`,`json`,`md`) | `text` | Formatting of record data in the LLM prompt. |

### Minimal Required Example

```json
{
	"dataset": "apifyDatasetIdOrParquetUrl",
	"instructions": "Extract company name, founded year, and HQ location from each profile."
}
```

### Full Example

```json
{
	"dataset": "apifyDatasetIdOrParquetUrl",
	"instructions": "Extract company name, founded year, HQ location and classify the industry. Output industry as one of: SaaS, FinTech, Health, Other.",
	"model": "openai/gpt-4o-mini",
	"schema_model": "openai/gpt-4.1",
	"response_schema": "Ensure founded_year is an integer and industry is one of the allowed enum values.",
	"attrs": ["company_profile", "about", "summary"],
	"record_format": "json"
}
```

## Example Use Cases

- **Customer Review Analysis**: Extract sentiment, product aspects, and issues from reviews
- **Document Processing**: Extract key entities, dates, and metadata from documents  
- **Survey Analysis**: Structure open-ended survey responses into analyzable data
- **Email Processing**: Extract contact information, topics, and action items from emails
- **Social Media Analysis**: Extract hashtags, mentions, sentiment, and topics from posts

## Output

The actor returns a dataset where each record contains:
- All original record fields
- Additional fields containing the extracted information as defined by the auto-generated schema

The exact output structure depends on your instructions and the automatically generated schema.
