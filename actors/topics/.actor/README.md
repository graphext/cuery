# General Topic Extractor

This Apify actor extracts hierarchical topic structures from data records with arbitrary attributes using AI. It serves as a wrapper around `cuery.tools.flex.topics.TopicExtractor` and is designed for general topic extraction from any type of structured data.

## Features

- **Hierarchical Topic Extraction**: Creates a two-level nested topic structure with main topics and subtopics
- **Flexible Data Input**: Works with data records/DataFrames containing arbitrary attributes/columns
- **AI-Powered Analysis**: Uses advanced language models to understand semantic patterns in your data
- **Scalable Processing**: Efficiently handles large datasets with intelligent sampling
- **Configurable Topic Structure**: Control the number of top-level topics and subtopics
- **Domain-Agnostic**: Works with any type of data - not limited to SEO or specific domains
- **Context-Aware**: Supports custom instructions to provide domain-specific context

## Purpose

**This actor has one main purpose: to extract a hierarchical topic structure from your data.**

The actor extracts topics using generic AI instructions that don't assume any specific context about your data. For best results, you should provide additional context via the `instructions` parameter to help the AI understand your specific domain and requirements.

## Input

The actor requires an Apify dataset or a Parquet file containing records with arbitrary attributes. You can specify which attributes to use for topic extraction, or use all available attributes.

### Example Input

```json
{
  "dataset": "your-dataset-id-or-url",
  "model": "openai/gpt-4.1",
  "attrs": ["title", "description", "category"],
  "n_topics": 8,
  "n_subtopics": 4,
  "instructions": "Extract topics from e-commerce product data focusing on product categories and features",
  "max_samples": 300
}
```

### SEO Use Case Example

For SEO keyword topic extraction, you might use instructions similar to this:

```json
{
  "dataset": "your-dataset-id-or-url",
  "instructions": "Data records represent Google search keywords and associated SERP data. Make sure to create topics relevant in the context of SEO keywords research, focusing on the semantic meaning of keywords and SERPS, commercial intent etc.",
  "attrs": ["keyword", "titles", "domains"]
}
```

## Output

The actor outputs a hierarchical topic structure as a JSON object with key-value store entry:

```json
{
  "Marketing & Advertising": [
    "Digital Marketing Strategy",
    "Social Media Marketing", 
    "Content Marketing",
    "Email Marketing"
  ],
  "E-commerce": [
    "Product Management",
    "Shopping Experience",
    "Payment Systems"
  ],
  "Technology": [
    "Web Development",
    "Mobile Apps",
    "Analytics Tools"
  ]
}
```

The output is stored in the actor's key-value store with the key format: `topics-{dataset_ref}`

## Configuration

### Core Parameters

- **dataset**: Apify dataset ID containing your input data or URL to Parquet file (required)
- **model**: AI model for topic extraction (default: openai/gpt-3.5-turbo)
- **attrs**: List of record attributes to use for topic extraction (optional - uses all if not specified)
- **instructions**: Additional context and instructions for the AI model (highly recommended)

### Topic Structure

- **n_topics**: Maximum top-level topics (default: 10, range: 1-20)
- **n_subtopics**: Maximum subtopics per topic (default: 5, range: 1-10)
- **min_ldist**: Minimum Levenshtein distance between topic labels to avoid similar names (default: 2)

### Processing

- **max_samples**: Maximum records to sample for topic extraction (default: 500)
- **record_format**: Format for records in AI prompt - "attr_wise" or "rec_wise" (default: attr_wise)

## Supported AI Models

The actor supports any AI model by OpenAI, Google or Anthropic. The format should follow these examples:

- `openai/gpt-3.5-turbo` (default)
- `openai/gpt-4.1`
- `google/gemini-2.5-flash`
- `anthropic/claude-3-5-sonnet`
- Any other model following the `provider/model` format

## Use Cases

- **Content Analysis**: Understand topic distribution in articles, blog posts, or documents
- **SEO Research**: Organize keywords into meaningful topic hierarchies (with appropriate instructions)
- **Product Categorization**: Extract topics from product descriptions and features
- **Customer Feedback Analysis**: Identify themes in reviews, support tickets, or surveys
- **Research Organization**: Structure academic papers, reports, or research data
- **Social Media Analysis**: Categorize posts, comments, or discussions by topic
- **Market Research**: Analyze survey responses or interview transcripts

## Tips for Best Results

1. **Provide Context**: Use the `instructions` parameter to give the AI context about your data domain
2. **Select Relevant Attributes**: Choose the most relevant columns/attributes for topic extraction
3. **Adjust Topic Numbers**: Start with default values and adjust based on your data complexity
4. **Sample Size**: Use appropriate `max_samples` - more samples give better coverage but cost more
5. **Domain-Specific Instructions**: For specialized domains like SEO, provide more specific instructions
