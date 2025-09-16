# Architecture Diagram: News Summarizer AI Agent

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

**Legend**  
- Blue boxes = Core components  
- Green boxes = External APIs  
- Orange box = Output to user


**Flow:**
- User interacts with Streamlit UI (topic, style, etc.)
- App fetches news from NewsAPI and/or Google News (SerpAPI)
- Articles are deduplicated and sorted
- Each article is summarized using Gemini AI
- Results (summaries, key points, insights) are shown in the UI and available for download

---

Feel free to replace this with a PNG or draw.io diagram for your submission.
