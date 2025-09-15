# app.py
import os
import time
import requests
from typing import List, Dict
import streamlit as st
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# Optional: for better article text extraction (install newspaper3k)
# from newspaper import Article

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

### ---------- Helpers ----------
@st.cache_data(ttl=300)
def fetch_news(topic: str, num_articles: int = 5, language: str = "en", use_top_headlines: bool=False) -> List[Dict]:
    """Fetch news articles from NewsAPI."""
    base = "https://newsapi.org/v2/top-headlines" if use_top_headlines else "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "pageSize": num_articles,
        "language": language,
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY
    }
    res = requests.get(base, params=params, timeout=10)
    res.raise_for_status()
    data = res.json()
    articles = data.get("articles", [])
    # Normalize fields
    results = []
    for a in articles:
        results.append({
            "title": a.get("title"),
            "description": a.get("description"),
            "content": a.get("content"),   # often truncated
            "url": a.get("url"),
            "source": a.get("source", {}).get("name"),
            "publishedAt": a.get("publishedAt")
        })
    return results

def extract_full_text_with_newspaper(url: str) -> str:
    """Optional: use newspaper3k to extract full article text. Use only if installed."""
    try:
        art = Article(url)
        art.download()
        art.parse()
        return art.text
    except Exception as e:
        return ""

def chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    """Simple char-based chunker to avoid token limits."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+max_chars])
        start += max_chars
    return chunks

def summarize_with_gemini(prompt: str, model: str = "gemini-2.5-flash", max_output_tokens: int = 512) -> str:
    try:
        model = genai.GenerativeModel(model)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=max_output_tokens
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"[Error summarizing: {e}]"

def summarize_article_content(content: str, title: str = "", model: str = "gemini-2.5-flash") -> str:
    if not content:
        return "No article text available to summarize."
    chunks = chunk_text(content, max_chars=3000)
    summaries = []
    for i, c in enumerate(chunks):
        prompt = f"Summarize the following news text (chunk {i+1}/{len(chunks)}) in 2-3 short sentences, include the key facts and why it matters:\n\n{c}"
        s = summarize_with_gemini(prompt, model=model, max_output_tokens=220)
        summaries.append(s)
    if len(summaries) == 1:
        return summaries[0]
    combined_prompt = "Combine the following chunk summaries into a coherent 2–3 sentence news summary, removing repetition:\n\n" + "\n\n".join(summaries)
    final = summarize_with_gemini(combined_prompt, model=model, max_output_tokens=220)
    return final


@st.cache_data(ttl=600)
def generate_summaries_for_articles(articles: List[Dict], model: str="gpt-3.5-turbo") -> List[Dict]:
    """Generate short summaries for list of articles. Uses available description/content."""
    out = []
    for art in articles:
        text = art.get("content") or art.get("description") or ""
        # Optionally try to fetch full text:
        # full = extract_full_text_with_newspaper(art['url'])
        # if full:
        #     text = full
        summary = summarize_article_content(text, title=art.get("title", ""), model=model) if text else "No text available to summarize."
        out.append({**art, "summary": summary})
    return out

### ---------- Streamlit UI ----------
st.set_page_config(page_title="News Summarizer", layout="wide")
st.title("📰 News Summarizer — quick daily digest")
st.write("Enter a topic or choose a category, then click *Fetch & Summarize*.")

# Sidebar controls
with st.sidebar:
    st.header("Options")
    topic = st.text_input("Search topic (leave blank for top news)", value="AI")
    categories = ["General", "Technology", "Business", "Sports", "Health", "Science", "Entertainment"]
    category = st.selectbox("Or choose a category", categories, index=1)
    num_articles = st.slider("Number of articles", 1, 10, 5)
    model_choice = st.selectbox("Model (for demo use gpt-3.5-turbo)", ["gpt-3.5-turbo", "gpt-4"], index=0)
    use_top_headlines = st.checkbox("Use top headlines endpoint (category-based)", value=False)
    if use_top_headlines:
        topic_to_search = category if category != "General" else ""
    else:
        topic_to_search = topic

# Fetch button
if st.button("Fetch & Summarize"):
    if not NEWSAPI_KEY:
        st.error("Set your NEWSAPI_KEY environment variable first.")
    elif not GEMINI_API_KEY:
        st.error("Set your GEMINI_API_KEY environment variable first.")
    else:
        with st.spinner("Fetching articles..."):
            try:
                articles = fetch_news(topic_to_search or "", num_articles=num_articles, use_top_headlines=use_top_headlines)
            except Exception as e:
                st.error(f"Error fetching news: {e}")
                st.stop()
        if not articles:
            st.warning("No articles found for that topic.")
            st.stop()

        st.success(f"Fetched {len(articles)} articles. Generating summaries...")
        summaries = generate_summaries_for_articles(articles, model=model_choice)

        # Show in columns / cards
        for i, a in enumerate(summaries):
            st.markdown("---")
            title_line = f"**{i+1}. {a.get('title','(No Title)')}**  \nSource: {a.get('source')} • {a.get('publishedAt')}"
            st.write(title_line)
            st.write(a.get("summary"))
            st.write(f"[Read original article →]({a.get('url')})")
            # Deep dive button
            if st.button(f"Deep Dive: longer summary ({i+1})", key=f"deep_{i}"):
                full_text = a.get("content") or a.get("description") or ""
                # try to fetch full article - optional heavy
                # full_text_new = extract_full_text_with_newspaper(a['url'])
                # if full_text_new:
                #     full_text = full_text_new
                if not full_text:
                    st.info("No full text available for a deeper summary.")
                else:
                    with st.spinner("Generating deep-dive summary..."):
                        long_prompt = f"Write a 3-paragraph detailed news summary and context (background, implications) for the following article:\n\n{full_text}"
                        long_summary = summarize_with_gemini(long_prompt, model=model_choice, max_tokens=600)
                        st.markdown(long_summary)

        # Download option (CSV)
        df = pd.DataFrame([{"title": a["title"], "source": a["source"], "publishedAt": a["publishedAt"], "summary": a["summary"], "url": a["url"]} for a in summaries])
        csv = df.to_csv(index=False)
        st.download_button("Download digest (CSV)", csv, file_name="news_digest.csv", mime="text/csv")
        st.success("Done! Scroll up to read the digest.")
