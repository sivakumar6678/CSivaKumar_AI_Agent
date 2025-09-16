
# CSivaKumar_AI_Agent_Challenge

## News Summarizer AI Agent

### What does this agent do?
This AI agent fetches the latest news articles from multiple sources (NewsAPI, Google News via SerpAPI), deduplicates them, and uses Google Gemini (Generative AI) to generate concise summaries and key points for each article. Users can customize the topic, date range, summary style, and more. The agent presents the results in a modern Streamlit web app, with options to download the digest and view insights.

### Key Features
- Multi-source news fetching (NewsAPI, Google News)
- AI-powered summarization using Gemini
- Multiple summary styles (Simple, Professional, Technical, Detailed, Bulleted)
- Deduplication and sorting of articles
- Download summaries as CSV or Markdown
- Insights: source breakdown, timeline, top domains
- Caching for efficient repeated queries
- User-friendly Streamlit UI

### Limitations
- Dependent on API quotas (Gemini, NewsAPI, SerpAPI)
- Summaries may be unavailable if Gemini quota is exceeded
- Only English news supported
- Requires API keys for all services

### Tools and APIs Used
- [Streamlit](https://streamlit.io/) (UI)
- [Google Gemini](https://ai.google.dev/) (AI summarization)
- [NewsAPI](https://newsapi.org/) (news source)
- [SerpAPI](https://serpapi.com/) (Google News)
- [pandas](https://pandas.pydata.org/) (data handling)
- [dotenv](https://pypi.org/project/python-dotenv/) (config)

### Setup Instructions
1. **Clone the repo and install dependencies:**
	```bash
	pip install -r requirements.txt
	```
2. **Set up your `.env` file with the following keys:**
	```env
	NEWSAPI_KEY=your_newsapi_key
	GEMINI_API_KEY=your_gemini_api_key
	SERPAPI_KEY=your_serpapi_key
	```
3. **Run the app:**
	```bash
	streamlit run app.py
	```
4. **Open the Streamlit URL in your browser.**

### Architecture Diagram
See `architecture.png` for a high-level overview.

---

## Author
Sivakumar (CSivaKumar_AI_Agent_Challenge)

