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

### Required

- **dataset**: The ID of the Apify dataset containing your data records or the URL of a parquet file
- **instructions**: Detailed description of what information to extract and how to process the data

### Optional

- **Model**: LLM model for data processing (default: `openai/gpt-3.5-turbo`)
- **Schema Model**: LLM model for schema generation (default: `openai/gpt-4.1`)
- **Response Schema Instructions**: Specific instructions for schema generation
- **Record Attributes**: Specific columns to focus on (if not provided, all columns are used)
- **Record Format**: How records are formatted in prompts (`text`, `json`, or `md`)

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
