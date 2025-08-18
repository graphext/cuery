# SEO Keyword Topic & Intent Classifier

This Apify actor classifies keywords and their Search Engine Results Page (SERP) data into topics and search intent categories using AI. It analyzes organic search results to extract hierarchical topic structures and categorizes search intent as informational, navigational, transactional, or commercial.

## Features

- **Topic Extraction**: Analyzes SERP data to create a hierarchical topic structure with main topics and subtopics
- **Intent Classification**: Categorizes keywords into four search intent types:
  - **Informational**: Users seeking information, answers, or knowledge
  - **Navigational**: Users looking for specific websites or pages  
  - **Transactional**: Users intending to complete an action or purchase
  - **Commercial**: Users researching products/services before purchasing
- **AI-Powered Analysis**: Uses advanced language models to understand semantic meaning and commercial context
- **Scalable Processing**: Efficiently handles large keyword datasets with intelligent sampling
- **Flexible Model Selection**: Choose from multiple AI models based on your quality and cost requirements

## Input

The actor requires aggregated SERP data for keywords, typically obtained from the SEO SERP Scraper actor. Each keyword should include:

- `term`: The keyword/search term
- `titles`: Array of page titles from top organic results
- `domains`: Array of unique domains from organic results  
- `breadcrumbs`: Array of breadcrumb navigation elements

### Example Input

```json
{
  "keywords_data": [
    {
      "term": "digital marketing",
      "titles": ["Digital Marketing Guide", "Best Digital Marketing Tools"],
      "domains": ["hubspot.com", "moz.com", "searchengineland.com"],
      "breadcrumbs": ["marketing", "tools", "guides"]
    }
  ],
  "max_samples": 500,
  "topic_model": "google/gemini-2.5-flash-preview-05-20", 
  "assignment_model": "openai/gpt-4.1-mini",
  "n_topics": 10,
  "n_subtopics": 5
}
```

## Output

The actor outputs a dataset with topic and intent classifications for each keyword:

```json
[
  {
    "term": "digital marketing",
    "topic": "Marketing & Advertising",
    "subtopic": "Digital Marketing Strategy", 
    "intent": "informational"
  }
]
```

## Configuration

### Core Parameters

- **keywords_data**: Array of keyword SERP data objects (required)
- **max_samples**: Maximum keywords to sample for topic extraction (default: 500)
- **topic_model**: AI model for topic hierarchy extraction (default: google/gemini-2.5-flash-preview-05-20)
- **assignment_model**: AI model for keyword classification (default: openai/gpt-4.1-mini)

### Topic Structure

- **n_topics**: Maximum top-level topics (default: 10, range: 3-20)
- **n_subtopics**: Maximum subtopics per topic (default: 5, range: 2-10)

### Processing

- **max_retries**: Retry attempts for failed API calls (default: 5)

## Supported AI Models

### Topic Extraction Models
- `google/gemini-2.5-flash-preview-05-20` (default) - Fast and cost-effective
- `openai/gpt-4.1-mini` - Good balance of quality and speed
- `openai/gpt-4.1-pro` - Highest quality but more expensive
- `anthropic/claude-3-5-sonnet` - Excellent for complex domains

### Assignment Models  
- `openai/gpt-4.1-mini` (default) - Excellent for classification
- `google/gemini-2.5-flash-preview-05-20` - Faster alternative
- `openai/gpt-4.1-pro` - Best accuracy for difficult cases
- `anthropic/claude-3-5-sonnet` - Good for nuanced intent classification

## Use Cases

- **SEO Content Strategy**: Understand topic coverage and identify content gaps
- **Search Intent Optimization**: Align content with user search intent
- **Keyword Grouping**: Organize large keyword lists into meaningful categories
- **Competitive Analysis**: Analyze topic focus of competitor keywords
- **Content Planning**: Guide content creation based on search patterns

## Performance Tips

- Use 100-500 samples for topic extraction to balance quality and speed
- Choose faster models for large datasets (1000+ keywords)
- Group similar keywords before processing to improve topic coherence
- Consider domain-specific context when interpreting results

## Requirements

The actor requires API keys for the selected AI models:
- OpenAI API key for GPT models
- Google API key for Gemini models  
- Anthropic API key for Claude models

Set these as environment variables in your Apify account settings.
