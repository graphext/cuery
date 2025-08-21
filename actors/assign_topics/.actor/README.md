# SERP Topic Classifier

This Apify actor assigns predefined topics and subtopics to keywords based on their Search Engine Results Page (SERP) data using AI classification. It takes a dataset of keywords with SERP data and a predefined topic hierarchy to classify each keyword into appropriate topic categories.

## Features

- **Topic Assignment**: Assigns keywords to predefined topic categories and subtopics
- **SERP Context Analysis**: Uses SERP data (titles, domains, breadcrumbs) for accurate classification
- **AI-Powered Classification**: Leverages advanced language models for intelligent topic assignment
- **Flexible Topic Hierarchy**: Supports custom topic structures with multiple levels
- **Batch Processing**: Efficiently processes large datasets of keywords
- **Configurable Models**: Choose from various AI models for classification

## Input

The actor expects:

1. **Dataset**: ID of an Apify dataset containing keyword SERP data, or a URL of a Parquet file
2. **Topic Hierarchy**: A JSON object defining topics and subtopics for classification
3. **Configuration**: Options for text columns, extra context columns, and AI model selection

### Required Input Fields

- `dataset`: ID of an Apify dataset containing keyword SERP data, or a URL of a Parquet file
- `topics`: JSON object with topic hierarchy (main topics as keys, subtopics as arrays)

### Optional Input Fields

- `text_column`: Column name containing the keywords (default: "term")
- `extra_columns`: Additional SERP data columns for context (default: ["titles", "domains", "breadcrumbs"])
- `model`: AI model for classification (default: "openai/gpt-4.1-mini")

### Input Data Format

The input dataset should contain keyword SERP data with columns like:
- `term`: The keyword/search term
- `titles`: Array of SERP result titles
- `domains`: Array of SERP result domains
- `breadcrumbs`: Array of SERP result breadcrumbs

## Output

The actor outputs a dataset with the original keyword data plus assigned topic classifications:

- All original columns from the input dataset
- `topic`: The assigned main topic category
- `subtopic`: The assigned subtopic within the main category

## Configuration

### Topic Hierarchy Structure

Define your topic hierarchy as a JSON object where each key is a main topic and the value is an array of subtopics:

```json
{
  "Technology": ["AI/ML", "Web Development", "Mobile Apps", "Cloud Computing"],
  "Marketing": ["SEO", "Social Media", "Content Marketing", "Email Marketing"],
  "Business": ["Finance", "Operations", "Strategy", "Human Resources"]
}
```

### Supported AI Models

- **OpenAI**: openai/gpt-4.1-mini, openai/gpt-4.0-preview, openai/gpt-3.5-turbo
- **Google**: google/gemini-2.5-flash-preview-05-20, google/gemini-1.5-pro
- **Anthropic**: anthropic/claude-3-sonnet, anthropic/claude-3-haiku

## Use Cases

- **Content Categorization**: Organize large keyword lists by topic
- **SEO Strategy**: Group keywords for targeted content creation
- **Market Research**: Understand topic distribution in search queries
- **Content Planning**: Identify content gaps across topic categories
- **Competitive Analysis**: Categorize competitor keywords by topic

## Performance Tips

- Use concise, clear topic and subtopic names
- Include relevant SERP context columns for better accuracy
- Start with 3-10 main topics for optimal classification
- Limit subtopics to 2-8 per main topic for best results
- Use descriptive topic hierarchies that match your content strategy

## Requirements

- Input dataset with keyword SERP data
- Predefined topic hierarchy in JSON format
- Sufficient memory allocation (minimum 256MB, recommended 1GB+ for large datasets)
