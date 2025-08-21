# Entity Extractor

This Apify actor extracts entities from data records/rows using AI. It can identify and extract various types of entities (brands, products, locations, people, etc.) from arbitrary data attributes based on user-defined entity categories.

## Features

- **Flexible Entity Extraction**: Extract entities from data records with arbitrary attributes
- **AI-Powered Analysis**: Leverages advanced language models for intelligent entity recognition
- **Batch Processing**: Efficiently processes large datasets of records
- **Multiple Models**: Choose from various AI models for entity extraction
- **Custom Entity Categories**: Define your own entity types and descriptions
- **Multi-Attribute Analysis**: Uses some or all available record attributes for accurate extraction
- **Normalized Output**: Entities are returned in lowercase, singular form for consistency

## Input

### Required Input Fields

- `dataset`: The ID of the Apify dataset containing data records from which to extract entities, or the URL of a Parquet file
- `entities`: Dictionary of entity category names and their descriptions

### Optional Input Fields

- `attrs`: List of record attribute names to use for entity extraction (if None, all attributes are used)
- `record_format`: Format of the record in the prompt ("text", "json", or "md", default: "text")
- `model`: AI model for entity extraction (default: "openai/gpt-4.1-mini")

### Input Data Format

The input dataset can contain any type of data records with arbitrary attributes. The extractor will analyze all selected attributes to identify and extract entities.

## Output

The actor outputs a dataset with the original record data plus extracted entities organized by category:

- All original columns from the input dataset
- `entities_<category>`: For each entity category, a column containing a list of extracted entities of that type

For example, if you define entity categories "brand" and "product", the output will include:
- `entities_brand`: List of brand names found in the record
- `entities_product`: List of product names found in the record

## Configuration

### Entity Categories

Define your own entity categories by providing a dictionary of category names and descriptions:

```json
{
  "brand": "Company or brand names mentioned in the content",
  "product": "Specific products or services mentioned", 
  "location": "Geographic locations, cities, countries, or regions",
  "person": "Names of people, individuals, or public figures"
}
```

### Example Input

```json
{
  "dataset": "your-dataset-ref-here",
  "model": "openai/gpt-3.5-turbo",
  "entities": {
    "brand": "Company or brand names mentioned in the content",
    "product": "Specific products or services mentioned",
    "location": "Geographic locations, cities, countries, or regions"
  },
  "attrs": ["title", "description"],
  "instructions": "Focus on extracting entities relevant for SEO analysis."
}
```

## Use Cases

- **SEO Analysis**: Extract brands, products, and locations from web content
- **Content Analysis**: Identify key entities in articles, reviews, or descriptions
- **Market Research**: Extract competitor names, product mentions from datasets
- **Data Enrichment**: Add structured entity information to unstructured text data
- **Information Extraction**: Pull specific entity types from large text datasets

## Model Support

The actor supports various AI models:
- **OpenAI**: gpt-4, gpt-3.5-turbo, gpt-4-turbo
- **Anthropic**: claude-3-opus, claude-3-sonnet, claude-3-haiku
- **Google**: gemini-pro, gemini-1.5-pro

Choose the model that best fits your accuracy requirements and budget constraints.

## Entity Normalization

Extracted entities are automatically normalized:
- Converted to lowercase for consistency
- Singular form used to avoid duplicates

This ensures consistent output regardless of how entities appear in the source data.
