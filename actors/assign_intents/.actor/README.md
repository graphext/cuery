# SERP Intent Classifier

This Apify actor classifies search intent for keywords based on their Search Engine Results Page (SERP) data using AI. It analyzes SERP content to categorize keywords into four primary search intent categories: informational, navigational, transactional, or commercial.

## Features

- **Intent Classification**: Automatically categorizes keywords into 4 search intent types
- **SERP Context Analysis**: Uses SERP data (titles, domains, breadcrumbs) for accurate classification
- **AI-Powered Analysis**: Leverages advanced language models for intelligent intent detection
- **Batch Processing**: Efficiently processes large datasets of keywords
- **Multiple Models**: Choose from various AI models for classification
- **Detailed Context**: Incorporates rich SERP data for better accuracy

## Search Intent Categories

- **Informational**: User seeks information, answers, or knowledge (how-to, what is, tutorials)
- **Navigational**: User wants to find a specific website or page (brand names, specific sites)
- **Transactional**: User intends to make a purchase or complete an action (buy, download, sign up)
- **Commercial**: User is researching products/services before purchasing (reviews, comparisons, "best")

## Input

The actor expects:

1. **Dataset ID**: An Apify dataset containing keyword SERP data
2. **Configuration**: Options for text columns, extra context columns, and AI model selection

### Required Input Fields

- `dataset_id`: The ID of the dataset containing keyword SERP data

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

The actor outputs a dataset with the original keyword data plus assigned search intent classifications:

- All original columns from the input dataset
- `intent`: The classified search intent category (informational, navigational, transactional, commercial)

## Configuration

### AI Model Selection

Choose from various AI models based on your accuracy and speed requirements:

- **OpenAI**: openai/gpt-4.1-mini (recommended), openai/gpt-4.0-preview, openai/gpt-3.5-turbo
- **Google**: google/gemini-2.5-flash-preview-05-20, google/gemini-1.5-pro
- **Anthropic**: anthropic/claude-3-sonnet, anthropic/claude-3-haiku

### Context Columns

Include relevant SERP data columns to improve classification accuracy:
- `titles`: Page titles from search results
- `domains`: Domain names from search results
- `breadcrumbs`: Navigation breadcrumbs from search results

## Use Cases

- **SEO Strategy**: Group keywords by intent for targeted content creation
- **Content Planning**: Create content that matches user search intent
- **PPC Campaigns**: Optimize ad copy and landing pages based on search intent
- **User Experience**: Design website flow based on visitor intent patterns
- **Market Research**: Understand customer journey and intent distribution
- **Competitive Analysis**: Analyze competitor keyword intent strategies

## Performance Tips

- Include multiple SERP context columns for better accuracy
- Use recent SERP data for more accurate intent classification
- Process keywords in batches for optimal performance
- Choose faster models (like gpt-4.1-mini) for large datasets
- Review and validate results for critical business keywords

## Requirements

- Input dataset with keyword SERP data
- Sufficient memory allocation (minimum 256MB, recommended 1GB+ for large datasets)
- Valid API keys for the selected AI model provider

## Intent Classification Examples

- **Informational**: "how to optimize website speed", "what is SEO", "digital marketing guide"
- **Navigational**: "Facebook login", "Gmail", "Amazon customer service"
- **Transactional**: "buy running shoes", "download software", "book hotel room"
- **Commercial**: "best laptops 2024", "iPhone vs Samsung", "web hosting reviews"
