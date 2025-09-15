# YourName_AI_Agent_Challenge

## News Summarizer (Streamlit)
A Streamlit app that fetches recent news about a topic and generates concise AI summaries.

### Features
- Search by topic or choose category
- Fetch N latest articles (NewsAPI)
- Short summaries (2-3 sentence) with Deep Dive option
- Download digest as CSV

### Tech
- Streamlit (UI)
- NewsAPI (news source)
- OpenAI (summarization)
- Optional: newspaper3k for full article text

### Setup
1. Clone repo
2. Create `.env` with NEWSAPI_KEY and OPENAI_API_KEY
3. `pip install -r requirements.txt`
4. `streamlit run app.py`

### Limitations
- Uses NewsAPI (rate limits and source coverage)
- Article content may be truncated without full-text scraping
- Summaries depend on LLM access and tokens

### Deployment
- Streamlit Cloud or HuggingFace Spaces. Set secrets in app settings.

