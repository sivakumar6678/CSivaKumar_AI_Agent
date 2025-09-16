# CSivaKumar_AI_Agent_Challenge

## News Summarizer AI Agent

### Demo
- [Live Demo on Streamlit](https://sivakumar6678-csivakumar-ai-agent-app-3reonz.streamlit.app/)

---

### What does this agent do?
This AI agent fetches the latest news articles from multiple sources (NewsAPI, Google News via SerpAPI), deduplicates them, and uses Google Gemini (Generative AI) to generate concise summaries and key points for each article. Users can customize the topic, date range, summary style, and more. The agent presents the results in a modern Streamlit web app, with options to download the digest and view insights.

---

### Key Features
- Multi-source news fetching (NewsAPI, Google News)
- AI-powered summarization using Gemini
- Multiple summary styles (Simple, Professional, Technical, Detailed, Bulleted)
- Deduplication and sorting of articles
- Download summaries as CSV or Markdown
- Insights: source breakdown, timeline, top domains
- Caching for efficient repeated queries
- User-friendly Streamlit UI

---

### Limitations
- Dependent on API quotas (Gemini, NewsAPI, SerpAPI)
- Summaries may be unavailable if Gemini quota is exceeded
- Only English news supported
- Requires API keys for all services

---

### Tools and APIs Used
- [Streamlit](https://streamlit.io/) (UI)
- [Google Gemini](https://ai.google.dev/) (AI summarization)
- [NewsAPI](https://newsapi.org/) (news source)
- [SerpAPI](https://serpapi.com/) (Google News)
- [pandas](https://pandas.pydata.org/) (data handling)
- [dotenv](https://pypi.org/project/python-dotenv/) (config)

---

### Setup Instructions

#### 1. Python Environment
1. Install Python 3.11 or higher.  
   ```bash
   python --version
   ```
2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:

   * **Windows:**

     ```bash
     venv\Scripts\activate
     ```
   * **Linux / MacOS:**

     ```bash
     source venv/bin/activate
     ```
4. Upgrade pip (optional but recommended):

   ```bash
   pip install --upgrade pip
   ```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Set Up API Keys

Create a `.env` file in the project root and add your API keys:

```env
NEWSAPI_KEY=your_newsapi_key
GEMINI_API_KEY=your_gemini_api_key
SERPAPI_KEY=your_serpapi_key
```

**How to get the API keys:**

1. **NewsAPI:**

   * Sign up at [NewsAPI](https://newsapi.org/register)
   * Copy your API key and paste into `.env`.
2. **Google Gemini:**

   * Sign up via [Google Cloud AI](https://cloud.google.com/ai)
   * Enable Generative AI API and create an API key.
   * Note: API key is tied to phone number verification.
3. **SerpAPI (Google News):**

   * Sign up at [SerpAPI](https://serpapi.com/users/sign_up)
   * Generate your API key and add it to `.env`.

#### 4. Run the App

```bash
streamlit run app.py
```

* Open the Streamlit URL in your browser (usually `http://localhost:8501`).

---

### Architecture Diagram

See `architecture.png` for a high-level overview of the News Summarizer AI Agent.

```
+-------------------+         +-------------------+         +-------------------+
|                   |         |                   |         |                   |
|    User (UI)      +-------->+   Streamlit App   +-------->+   News Sources    |
|                   |         |                   |         | (NewsAPI, SerpAPI)|
+-------------------+         |                   |         +-------------------+
                              |                   |
                              |                   |         +-------------------+
                              |                   +-------->+  Gemini AI Model  |
                              |                   |         +-------------------+
                              |                   |
                              +--------+----------+
                                       |
                                       v
                              +-------------------+
                              |   Summaries,      |
                              |   Key Points,     |
                              |   Insights,       |
                              |   Downloads       |
                              +-------------------+
```

**Flow:**

* User interacts with Streamlit UI (topic, style, etc.)
* App fetches news from NewsAPI and/or Google News (SerpAPI)
* Articles are deduplicated and sorted
* Each article is summarized using Gemini AI
* Results (summaries, key points, insights) are shown in the UI and available for download

---

## Author

Sivakumar (CSivaKumar_AI_Agent_Challenge)

---

✅ Submission for AI Agent Development Challenge 2025
