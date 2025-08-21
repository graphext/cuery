# Data Record Scorer

This Apify actor performs general scoring of data records/rows using AI. It can assign custom scores to any type of data records with arbitrary attributes based on user-defined scoring criteria.

## Features

- **Flexible Scoring**: Score data records with arbitrary attributes using custom scoring criteria
- **AI-Powered Analysis**: Leverages advanced language models for intelligent scoring
- **Batch Processing**: Efficiently processes large datasets of records
- **Multiple Models**: Choose from various AI models for scoring
- **Custom Score Definitions**: Define your own scoring criteria, ranges, and descriptions
- **Multi-Attribute Analysis**: Uses some or all available record attributes for accurate scoring
- **Configurable Score Types**: Support for both integer and float scores

## Input

### Required Input Fields

- `dataset`: The ID of the dataset containing data records to score, or the URL of a Parquet file
- `name`: Name of the score to assign (becomes the column name in output)
- `min`: Minimum value of the score range
- `max`: Maximum value of the score range
- `description`: Detailed description of what the score represents and scoring criteria

### Optional Input Fields

- `attrs`: List of record attribute names to use for scoring (if None, all attributes are used)
- `type`: Score type - "integer" or "float" (default: "float")
- `record_format`: Format of the record in the prompt ("text", "json", or "md", default: "text")
- `model`: AI model for scoring (default: "openai/gpt-3.5-turbo")

### Input Data Format

The input dataset can contain any type of data records with arbitrary attributes. The scorer will use all selected attributes to make scoring decisions.

## Output

The actor outputs a dataset with the original record data plus assigned scores:

- All original columns from the input dataset
- `{score_name}`: The assigned score based on the defined criteria (column name matches the `name` parameter)

## Configuration

### Custom Scoring

Define your own scoring criteria by providing:

1. **Score Name**: The name of the score (e.g., "relevance", "quality", "purchase_probability")
2. **Score Range**: Minimum and maximum values for the score
3. **Score Type**: Whether to use integers or floats
4. **Description**: Detailed explanation of what the score measures and how it should be assigned

### Example Scoring Scenarios

#### Purchase Probability Scoring
```json
{
  "name": "purchase_probability",
  "type": "integer",
  "min": 0,
  "max": 10,
  "description": "Estimate the probability of a purchase action based on the keyword and its associated attributes. The score should be between 0 and 10, where 0 means no purchase probability and 10 means very high purchase probability."
}
```

#### Content Quality Scoring
```json
{
  "name": "content_quality",
  "type": "float",
  "min": 0.0,
  "max": 1.0,
  "description": "Evaluate the overall quality of the content based on factors like relevance, accuracy, clarity, and usefulness. Score from 0.0 (poor quality) to 1.0 (excellent quality)."
}
```

#### Relevance Scoring
```json
{
  "name": "relevance",
  "type": "integer",
  "min": 1,
  "max": 5,
  "description": "Rate how relevant this record is to the given topic or search query. Use a scale from 1 (not relevant) to 5 (highly relevant)."
}
```

## Use Cases

- **SEO Keyword Analysis**: Score keywords for purchase intent, competition difficulty, or relevance
- **Content Evaluation**: Score articles, posts, or documents for quality, relevance, or engagement potential
- **Lead Scoring**: Evaluate potential customers or leads based on various attributes
- **Product Rating**: Score products based on features, reviews, or market potential
- **Risk Assessment**: Assign risk scores to financial records, applications, or transactions
- **Sentiment Analysis**: Score text data for sentiment, emotion, or tone

## Model Support

The actor supports AI models from different providers using the "provider/model" syntax:

- **OpenAI**: e.g. "openai/gpt-4.1-mini", "openai/gpt-3.5-turbo", ...
- **Anthropic**: "anthropic/claude-4-sonnet", ...
- **Google**: "google/gemini-2.5-flash", ...


## Error Handling

The actor includes comprehensive error handling:

- Automatic retries for transient failures
- Validation of input parameters
- Graceful handling of malformed data
- Detailed error logging for debugging

