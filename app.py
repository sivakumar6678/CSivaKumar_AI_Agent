# app.py (Enhanced UI/UX with async summarization)
import os
import time
import requests
from typing import List, Dict, Tuple
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from serpapi import GoogleSearch
from datetime import datetime, timedelta, date
from urllib.parse import urlparse
import concurrent.futures
import uuid
from newspaper import Article


# -------------------- Load env vars & Model --------------------
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

MODEL_NAME = "gemini-2.0-flash"
DEFAULT_TOPIC = "AI in healthcare"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# -------------------- Helpers --------------------
ISO_FORMATS = ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d")


def parse_date_safe(value: str | None) -> str:
    if not value:
        return ""
    for fmt in ISO_FORMATS:
        try:
            dt = datetime.strptime(value[:len(fmt)], fmt)
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            continue
    try:
        if value.isnumeric():
            ts = datetime.fromtimestamp(int(value))
            return ts.strftime("%Y-%m-%d %H:%M")
        return value
    except Exception:
        return value


def hostname(u: str | None) -> str:
    if not u:
        return "Unknown"
    try:
        return urlparse(u).hostname or "Unknown"
    except Exception:
        return "Unknown"


def get_high_res_image(url: str | None) -> str | None:
    """
    Attempts to fetch the top image from the article page using newspaper3k.
    Returns None if it fails.
    """
    if not url:
        return None
    try:
        art = Article(url)
        art.download()
        art.parse()
        return art.top_image or None
    except Exception:
        return None


# -------------------- Summarization --------------------
def summarize_with_gemini(prompt: str, max_output_tokens: int = 400, temperature: float = 0.2) -> str:
    if not GEMINI_API_KEY:
        return (
            "Demo Summary:\n"
            "• This is a mock summary showing how the News Summarizer works\n"
            "• Recent reports indicate trends and updates on this topic\n"
            "• Configure your Gemini API key for real-time AI summaries"
        )
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        return (response.text or "").strip()
    except Exception as e:
        return f"[Gemini Error: {e}]"


def chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    if not text:
        return []
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def summarize_article_content(content: str, title: str = "", style_choice: str = "Professional", temperature: float = 0.2) -> Tuple[str, str]:
    """
    Returns (summary, key_points_md)
    """
    if not content:
        return ("No article text available.", "")
    chunks = chunk_text(content, max_chars=3000)
    summaries = []
    for i, c in enumerate(chunks):
        prompt = (
            f"You are a news summarizer. Style: {style_choice}.\n"
            f"{summary_styles[style_choice]}\n\n"
            f"TITLE: {title}\n\nCONTENT:\n{content[:4000]}"
        )
        s = summarize_with_gemini(prompt, max_output_tokens=220, temperature=temperature)
        summaries.append(s)
        time.sleep(0.1)

    if len(summaries) == 1:
        main_summary = summaries[0]
    else:
        combined_prompt = (
            "Combine these into one concise 4–8 sentence summary, factual, neutral tone.\n\n"
            + "\n\n".join(summaries)
        )
        main_summary = summarize_with_gemini(combined_prompt, max_output_tokens=220, temperature=temperature)

    kp_prompt = (
        f"Extract 3–5 key bullet points highlighting the most important facts from this article. "
        f"Keep it neutral and factual.\n\nTITLE: {title}\n\nCONTENT:\n{content[:3500]}"
    )
    key_points = summarize_with_gemini(kp_prompt, max_output_tokens=180, temperature=temperature)
    return (main_summary, key_points)


# -------------------- Fetchers --------------------
@st.cache_data(ttl=300)
def fetch_newsapi(topic: str, num_articles: int, from_date: date | None, to_date: date | None) -> List[Dict]:
    if not NEWSAPI_KEY:
        return [
            {"title": f"Sample: {topic} News 1", "content": f"Demo content for {topic} article 1", "url": "https://example.com/1", "image": None, "source": "NewsAPI (Demo)", "publishedAt": "2024-01-15T10:00:00Z"},
            {"title": f"Sample: {topic} News 2", "content": f"Demo content for {topic} article 2", "url": "https://example.com/2", "image": None, "source": "NewsAPI (Demo)", "publishedAt": "2024-01-15T09:00:00Z"},
        ]
    url = "https://newsapi.org/v2/everything"
    to_dt = to_date or datetime.now().date()
    from_dt = from_date or (to_dt - timedelta(days=7))
    params = {
        "q": topic,
        "apiKey": NEWSAPI_KEY,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": max(1, min(num_articles, 100)),
        "from": from_dt.strftime("%Y-%m-%d"),
        "to": to_dt.strftime("%Y-%m-%d"),
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    articles = []
    for a in data.get("articles", []):
        img = a.get("urlToImage")
        # Try fetching high-res image if possible
        high_res = get_high_res_image(a.get("url")) or img
        articles.append({
            "title": a.get("title"),
            "content": a.get("description") or a.get("content"),
            "url": a.get("url"),
            "image": high_res,
            "source": a.get("source", {}).get("name") or "NewsAPI",
            "publishedAt": a.get("publishedAt"),
        })
    return articles


@st.cache_data(ttl=300)
def fetch_google_news(query: str, num_results: int) -> List[Dict]:
    if not SERPAPI_KEY:
        return []
    params = {"q": query, "tbm": "nws", "num": num_results, "api_key": SERPAPI_KEY}
    search = GoogleSearch(params)
    results = search.get_dict()
    articles = []
    for a in results.get("news_results", [])[:num_results]:
        img = a.get("image") or a.get("thumbnail")  # prefer image over thumbnail
        # Optional: try fetch from top_image
        high_res = get_high_res_image(a.get("link")) or img
        articles.append({
            "title": a.get("title"),
            "content": a.get("snippet"),
            "url": a.get("link"),
            "image": high_res,
            "source": a.get("source") or hostname(a.get("link")),
            "publishedAt": a.get("date"),
        })
    return articles



def combine_and_dedupe(articles_lists: List[List[Dict]], max_items: int) -> List[Dict]:
    seen = set()
    combined: List[Dict] = []
    for lst in articles_lists:
        for a in lst:
            key = (a.get("url") or "").strip().lower() or (a.get("title") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            combined.append(a)
    def sort_key(a: Dict):
        ts = a.get("publishedAt") or ""
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return datetime.min
    combined.sort(key=sort_key, reverse=True)
    return combined[:max_items]


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="News Summarizer Agent", page_icon="📰", layout="wide")

# Header
left, right = st.columns([0.8, 0.2])
with left:
    st.markdown("# 📰 News Summarizer Agent")
    st.caption("AI-powered, multi-source news digests with concise summaries and key points.")

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "articles_data" not in st.session_state:
    st.session_state.articles_data = None
if "last_topic" not in st.session_state:
    st.session_state.last_topic = DEFAULT_TOPIC

# Sidebar
with st.sidebar:
    st.header("⚙️ Options")
    topic = st.text_input("Enter topic", value=st.session_state.last_topic)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        num_articles = st.slider("Articles", 1, 20, 8)
    with col_s2:
        temperature = st.slider("Creativity", 0.0, 1.0, 0.2, 0.1)

    sources = st.multiselect("Sources", options=["NewsAPI", "Google News (SerpAPI)"], default=["Google News (SerpAPI)"])

    today = datetime.now().date()
    date_range = st.date_input("Date range (NewsAPI only)", value=(today - timedelta(days=7), today), max_value=today)
    from_date, to_date = None, None
    if isinstance(date_range, tuple) and len(date_range) == 2:
        from_date, to_date = date_range

    summary_styles = {
        "Simple": "Explain in very simple, easy-to-read sentences (like for a school student).",
        "Professional": "Write in a formal, concise, business-professional tone.",
        "Technical": "Include technical details, terminology, and precise data if available.",
        "Detailed": "Write a detailed summary in 3–5 paragraphs.",
        "Bulleted": "Summarize only in 6–8 bullet points with facts."
    }

    style_choice = st.selectbox("Summary Style", list(summary_styles.keys()))
    sort_by = st.selectbox("Sort by", ["Newest", "Oldest", "Source"], index=0)
    use_ai = st.toggle("Use AI Summaries", value=True, help="Disable to see raw descriptions only")
    show_images = st.toggle("Show images", value=True)

    st.subheader("Recent Searches")
    for past in st.session_state.history[-8:][::-1]:
        if st.button(past, key=f"hist-{past}"):
            topic = past

# Tabs
summaries_tab, download_tab, insights_tab, history_tab = st.tabs([
    "📑 Summaries",
    "⬇️ Download",
    "� Insights",
    "�🕒 History",
])

# Fetch and summarize flow
with summaries_tab:
    if st.button("🔍 Get News", type="primary", use_container_width=True):
        if not sources:
            st.warning("Select at least one source.")
            st.stop()

        with st.spinner(f"Fetching up to {num_articles} articles from {', '.join(sources)}..."):
            try:
                lists = []
                if "NewsAPI" in sources:
                    lists.append(fetch_newsapi(topic, num_articles, from_date, to_date))
                if "Google News (SerpAPI)" in sources:
                    lists.append(fetch_google_news(topic, num_articles))
                articles = combine_and_dedupe(lists, max_items=num_articles)
            except Exception as e:
                st.error(f"❌ Error fetching news: {e}")
                st.stop()

        if not articles:
            st.warning("No articles found.")
            st.stop()

        st.session_state.history.append(topic)
        st.session_state.last_topic = topic
        st.session_state.articles_data = []

        article_placeholder = st.empty()
        progress = st.progress(0)

        # Async summarization
        def process_article(i, art):
            text = art.get("content") or ""
            if use_ai and text:
                summary, key_points = summarize_article_content(text, art.get("title", ""), style_choice=style_choice, temperature=temperature)
            else:
                summary, key_points = text or "No summary available.", ""
            return {**art, "summary": summary, "key_points": key_points, "index": i}

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_index = {executor.submit(process_article, i, art): i for i, art in enumerate(articles)}
            for future in concurrent.futures.as_completed(future_to_index):
                i = future_to_index[future]
                result = future.result()
                st.session_state.articles_data.append(result)

                # Update display for all processed articles so far
                with article_placeholder.container():
                    for a in sorted(st.session_state.articles_data, key=lambda x: x["index"]):
                        st.markdown("---")
                        col1, col2 = st.columns([1, 3], vertical_alignment="top")
                        with col1:
                            if show_images and a.get("image"):
                                st.image(a["image"], use_container_width=True)
                        with col2:
                            st.markdown(f"### {a['index']+1}. {a.get('title') or 'Untitled'}")
                            st.caption(f"📰 {a.get('source') or hostname(a.get('url'))} • {parse_date_safe(a.get('publishedAt'))}")
                            st.markdown("### 🔎 AI Summary" if use_ai else "### 📝 Description")
                            st.write(a["summary"])
                            if a.get("key_points"):
                                st.markdown("### ✅ Key Points")
                                st.markdown(a["key_points"])
                            col_a, col_b = st.columns([0.3, 0.7])
                            with col_a:
                                if st.link_button("Open Article", a.get("url") or "#", use_container_width=True):
                                    pass
                            with col_b:

                                # Inside the loop where deep dive button is created
                                deep_key = f"deep_{a['index']}_{hash(a.get('url',''))}_{uuid.uuid4()}"
                                if st.button(f"🔎 Deep Dive {a['index']+1}", key=deep_key):
                                    long_prompt = (
                                        f"Write a detailed 5-paragraph analysis of this article. "
                                        f"Include context, implications, and examples.\n\n{a['content']}"
                                    )
                                    long_summary = summarize_with_gemini(long_prompt, max_output_tokens=1000)
                                    st.markdown("### 📖 Deep Dive")
                                    st.write(long_summary)

                progress.progress(int(len(st.session_state.articles_data)/len(articles)*100))
                time.sleep(0.05)  # small delay for smooth UX

# -------------------- Downloads --------------------
with download_tab:
    if st.session_state.get("articles_data"):
        df = pd.DataFrame(st.session_state.articles_data)
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "news_digest.csv", "text/csv")

        # Markdown export
        md_lines = [f"# News Digest: {st.session_state.last_topic}", ""]
        for a in st.session_state.articles_data:
            md_lines.append(f"## {a.get('title')}")
            md_lines.append(f"- **Source**: {a.get('source')}  ")
            md_lines.append(f"- **Published**: {parse_date_safe(a.get('publishedAt'))}  ")
            md_lines.append("")
            md_lines.append("### Summary")
            md_lines.append(a.get("summary") or "")
            if a.get("key_points"):
                md_lines.append("\n### Key Points")
                md_lines.append(a["key_points"])
            md_lines.append(f"\n[Read full →]({a.get('url')})\n")
        md_content = "\n".join(md_lines)
        st.download_button("📝 Download Markdown", md_content, "news_digest.md", "text/markdown")
    else:
        st.info("Run a search first.")

# -------------------- Insights --------------------
with insights_tab:
    data = st.session_state.get("articles_data")
    if data:
        df = pd.DataFrame(data)
        st.subheader("Sources")
        src_counts = df["source"].fillna("Unknown").value_counts().reset_index()
        src_counts.columns = ["Source", "Count"]
        st.dataframe(src_counts, use_container_width=True)

        st.subheader("Timeline (PublishedAt)")
        df["_date"] = pd.to_datetime(df["publishedAt"], errors="coerce").dt.date
        st.bar_chart(df["_date"].value_counts().sort_index())

        st.subheader("Top Domains")
        df["_domain"] = df["url"].apply(hostname)
        st.bar_chart(df["_domain"].value_counts().head(10))
    else:
        st.info("Run a search first.")

# -------------------- History --------------------
with history_tab:
    if st.session_state.history:
        st.write("Your last searches:")
        for past in st.session_state.history[::-1]:
            st.write(f"- {past}")
        if st.button("Clear History", type="secondary"):
            st.session_state.history = []
    else:
        st.info("No history yet.")
