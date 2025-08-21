# Data Record Classifier

This Apify actor performs general classification of data records/rows using AI. It can classify any type of data records with arbitrary attributes into user-defined categories.

## Features

- **Flexible Classification**: Classify data records with arbitrary attributes into custom categories
- **AI-Powered Analysis**: Leverages advanced language models for intelligent classification
- **Batch Processing**: Efficiently processes large datasets of records
- **Multiple Models**: Choose from various AI models for classification
- **Custom Categories**: Define your own classification categories and descriptions
- **Multi-Attribute Analysis**: Uses some or all available record attributes for accurate classification

## Input

### Required Input Fields

- `dataset`: The ID of the Apify dataset containing data records to classify, or the URL of a Parquet file
- `categories`: Dictionary of category labels and their descriptions

### Optional Input Fields

- `attrs`: List of record attribute names to use for classification (if None, all attributes are used)
- `instructions`: Additional instructions/context for the classification task
- `record_format`: Format of the record in the prompt ("text", "json", or "md", default: "text")
- `model`: AI model for classification (default: "openai/gpt-4.1-mini")

### Input Data Format

The input dataset can contain any type of data records with arbitrary attributes. The classifier will use all selected attributes to make classification decisions.

## Output

The actor outputs a dataset with the original record data plus assigned classifications:

- All original columns from the input dataset
- `label`: The classified category label based on the defined categories

## Configuration

### Custom Categories

Define your own classification categories by providing a dictionary of category labels and descriptions:

```json
{
  "category1": "Description of category 1",
  "category2": "Description of category 2",
  "category3": "Description of category 3"
}
```

### AI Model Selection

Choose from various AI models based on your accuracy and speed requirements. For example:

- **OpenAI**: openai/gpt-4.1-mini (recommended), openai/gpt-3.5-turbo, ...
- **Google**: google/gemini-2.5-flash-preview-05-20, google/gemini-1.5-pro, ...
- **Anthropic**: anthropic/claude-3-sonnet, anthropic/claude-3-haiku, ...

### Record Attributes

Specify which attributes of your data records to use for classification:

- Include relevant attributes that contain information useful for classification
- If not specified, all attributes will be used
- Consider data quality and relevance when selecting attributes

## Use Cases

The flexible classification system can be applied to various domains, such as:

- **Search intent classification": Categorize keyword searches into navigational, commercial, transactional
- **Content Classification**: Categorize articles, documents, or posts by topic or theme
- **Customer Feedback Analysis**: Classify reviews, support tickets, or survey responses
- **Product Classification**: Categorize products by type, features, or target audience
- **Lead Scoring**: Classify leads by quality, potential, or priority
- **Sentiment Analysis**: Classify text by emotional tone or opinion
- **Risk Assessment**: Classify transactions, applications, or behaviors by risk level

## Performance Tips

- Define clear and distinct category descriptions for better accuracy
- Include multiple relevant attributes for more informed classification
- Use recent and high-quality data for optimal results
- Choose faster models (like gpt-4.1-mini) for large datasets
- Provide clear instructions/context for domain-specific classifications
- Review and validate results for critical business applications

## Requirements

- Input dataset with data records containing relevant attributes
- Sufficient memory allocation (minimum 256MB, recommended 1GB+ for large datasets)
- Valid API keys for the selected AI model provider
- Well-defined category labels and descriptions

## Example: Search Intent Classification

For search intent classification specifically, you would use:

**Input Categories:**
```json
{
  "informational": "User seeks information, answers, or knowledge (how-to, what is, tutorials)",
  "navigational": "User wants to find a specific website or page (brand names, specific sites)",
  "transactional": "User intends to make a purchase or complete an action (buy, download, sign up)",
  "commercial": "User is researching products/services before purchasing (reviews, comparisons, best)"
}
```

**Input Data Attributes:**
- `keyword`: The keyword/search term
- `titles`: Array of SERP result titles  
- `domains`: Array of SERP result domains
- ...

**Example Classifications:**
- **Informational**: "how to optimize website speed", "what is SEO", "digital marketing guide"
- **Navigational**: "Facebook login", "Gmail", "Amazon customer service"
- **Transactional**: "buy running shoes", "download software", "book hotel room"
- **Commercial**: "best laptops 2024", "iPhone vs Samsung", "web hosting reviews"
