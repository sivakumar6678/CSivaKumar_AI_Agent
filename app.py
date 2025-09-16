# app.py
import os
import time
import requests
from typing import List, Dict
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from serpapi import GoogleSearch
from datetime import datetime, timedelta

# Load env vars
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

MODEL_NAME = "gemini-2.0-flash"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------- Summarizer ----------
def summarize_with_gemini(prompt: str, max_output_tokens: int = 400) -> str:
    if not GEMINI_API_KEY:
        return f"""**Demo Summary:**  
• This is a mock summary showing how the News Summarizer works  
• Recent reports indicate trends and updates on this topic  
• Configure your Gemini API key for real-time AI summaries"""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=max_output_tokens
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"[Gemini Error: {e}]"

def chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    if not text:
        return []
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def summarize_article_content(content: str, title: str = "") -> str:
    if not content:
        return "No article text available."
    chunks = chunk_text(content, max_chars=3000)
    summaries = []
    for i, c in enumerate(chunks):
        prompt = f"Summarize this article (chunk {i+1}/{len(chunks)}) in 2–3 sentences:\n\n{c}"
        s = summarize_with_gemini(prompt, max_output_tokens=220)
        summaries.append(s)
        time.sleep(0.2)
    if len(summaries) == 1:
        return summaries[0]
    combined_prompt = "Combine these into one concise 2–3 sentence summary:\n\n" + "\n\n".join(summaries)
    return summarize_with_gemini(combined_prompt, max_output_tokens=220)

# ---------- Fetch News ----------
@st.cache_data(ttl=300)
def fetch_newsapi(topic: str, num_articles: int = 5) -> List[Dict]:
    if not NEWSAPI_KEY:
        # Demo articles if no key
        return [
            {"title": f"Sample: {topic} News 1", "content": f"Demo content for {topic} article 1", 
             "url": "https://example.com/1", "image": None, "source": "DemoSource", "publishedAt": "2024-01-15"},
            {"title": f"Sample: {topic} News 2", "content": f"Demo content for {topic} article 2", 
             "url": "https://example.com/2", "image": None, "source": "DemoSource", "publishedAt": "2024-01-15"},
        ]
    to_date = datetime.now()
    from_date = to_date - timedelta(days=7)
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "apiKey": NEWSAPI_KEY,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": num_articles,
        "from": from_date.strftime("%Y-%m-%d"),
        "to": to_date.strftime("%Y-%m-%d")
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    return [
        {
            "title": a.get("title"),
            "content": a.get("description") or a.get("content"),
            "url": a.get("url"),
            "image": a.get("urlToImage"),
            "source": a.get("source", {}).get("name"),
            "publishedAt": a.get("publishedAt"),
        }
        for a in data.get("articles", [])
    ]

@st.cache_data(ttl=300)
def fetch_google_news(query: str, num_results: int = 5) -> List[Dict]:
    if not SERPAPI_KEY:
        return []
    params = {"q": query, "tbm": "nws", "num": num_results, "api_key": SERPAPI_KEY}
    search = GoogleSearch(params)
    results = search.get_dict()
    return [
        {
            "title": a.get("title"),
            "content": a.get("snippet"),
            "url": a.get("link"),
            "image": a.get("thumbnail"),
            "source": a.get("source"),
            "publishedAt": a.get("date"),
        }
        for a in results.get("news_results", [])[:num_results]
    ]

def generate_summaries(articles: List[Dict]) -> List[Dict]:
    out = []
    for art in articles:
        text = art.get("content") or ""
        summary = summarize_article_content(text, art.get("title", "")) if text else "No summary available."
        out.append({**art, "summary": summary})
    return out

# ---------- Streamlit UI ----------
st.set_page_config(page_title="News Summarizer Agent", page_icon="📰", layout="wide")
st.title("📰 News Summarizer Agent")
st.caption("Get AI-powered summaries of the latest news (Gemini 2.0 Flash)")

if "history" not in st.session_state:
    st.session_state.history = []
if "summary_result" not in st.session_state:
    st.session_state.summary_result = None
if "articles_data" not in st.session_state:
    st.session_state.articles_data = None

with st.sidebar:
    st.header("⚙️ Options")
    topic = st.text_input("Enter topic", value="AI in healthcare")
    num_articles = st.slider("Articles", 1, 10, 5)
    source_choice = st.radio("Source", ["NewsAPI", "Google News (SerpAPI)"])
    if st.button("Daily Digest (Top 5 Headlines)"):
        topic = "Top News"
        num_articles = 5
    st.subheader("Recent Searches")
    for past in st.session_state.history[-5:][::-1]:
        if st.button(past, key=past):
            topic = past

tab1, tab2, tab3 = st.tabs(["📑 Summaries", "⬇️ Download", "🕒 History"])

with tab1:
    if st.button("🔍 Fetch & Summarize", type="primary", use_container_width=True):
        with st.spinner(f"Fetching {num_articles} articles from {source_choice}..."):
            try:
                if source_choice == "NewsAPI":
                    articles = fetch_newsapi(topic, num_articles)
                else:
                    articles = fetch_google_news(topic, num_articles)
            except Exception as e:
                st.error(f"❌ Error fetching news: {e}")
                st.stop()

        if not articles:
            st.warning("No articles found.")
            st.stop()

        st.session_state.history.append(topic)
        summaries = generate_summaries(articles)
        st.session_state.articles_data = summaries

        for i, a in enumerate(summaries):
            with st.expander(f"{i+1}. {a['title']}"):
                if a.get("image"):
                    st.image(a["image"], use_container_width=True)
                st.caption(f"📰 {a['source']} • {a['publishedAt']}")
                
                st.markdown("### 🔎 AI Summary")
                st.write(a["summary"])
                
                st.markdown(f"[Read full →]({a['url']})")


with tab2:
    if st.session_state.articles_data:
        st.write("Download last summaries as CSV:")
        df = pd.DataFrame(st.session_state.articles_data)
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "news_digest.csv", "text/csv")
    else:
        st.info("Run a search first.")

with tab3:
    if st.session_state.history:
        st.write("Your last searches:")
        for past in st.session_state.history[::-1]:
            st.write(f"- {past}")
    else:
        st.info("No history yet.")
