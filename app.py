````python
"""
AI Newsletter Generator - Streamlit App
Built using LangChain Agent (Tool Calling) + Gemini + Tavily
"""

import datetime
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from tavily import TavilyClient

# To show web-app: complete page layout
st.set_page_config(layout="wide")

# To give title
st.title("AI NEWSLETTER GENERATOR")

st.write("""This app helps you build a curated, styled HTML newsletter
from this week's top trending news using a LangChain agent.""")

st.sidebar.title("Fill Important Details")

# ============ API KEYS ===================
TAVILY_API_KEY = st.sidebar.text_input("Tavily-API", type="password")
GOOGLE_API_KEY = st.sidebar.text_input("Gemini-API", type="password")

all_API = [TAVILY_API_KEY, GOOGLE_API_KEY]

if not all(all_API):
    st.error("Must give API keys")
    st.stop()
else:
    st.success("API KEYS LOADED SUCCESSFULLY")

    # =========== MODEL CREATION ==============
    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GOOGLE_API_KEY
    )

# =========================================================
# FIXED NUMBER OF ARTICLES (ALWAYS 6)
# =========================================================
max_results = 6

# ================== GET USER INFO ======================
st.markdown("### NEWSLETTER DETAILS")
newsletter_title = st.text_input("Newsletter Title", value="Weekly Digest")

# =========== TOOL 1 ======================
def weekly_article_collector(max_results=6):
    """Fetch top trending news headlines from this week."""

    query = "top trending world news headlines this week"
    client = TavilyClient(api_key=TAVILY_API_KEY)

    response = client.search(
        query=query,
        topic="news",
        time_range="week",
        max_results=max_results,
        include_answer=False,
    )

    articles = []
    for result in response.get("results", []):
        articles.append({
            "title": result.get("title"),
            "url": result.get("url"),
            "content": result.get("content"),
            "published_date": result.get("published_date", "N/A"),
        })

    return articles

# =========== TOOL 2 ======================
def article_summarizer(article_text, article_title="Untitled"):
    """Summarize article and extract key information."""

    prompt = f"""
    You are a professional newsletter editor.

    Summarize the article below in 3-4 concise lines,
    then list 2-3 key points as bullets,
    assign a single category (Tech/Business/Science/World/Other)
    and give a relevance score out of 10 for a general audience.

    Article Title: {article_title}
    Article Content: {article_text}

    Give output strictly in this format:
    Summary: <summary>
    Key Points: <point1>; <point2>; <point3>
    Category: <category>
    Relevance: <score>/10
    """

    response = model.invoke(prompt)
    return _extract_text(response)

# =========== TOOL 3 ======================
def newsletter_html_generator(curated_summaries, newsletter_title="Weekly Newsletter"):
    """Generate styled HTML newsletter."""

    current_date = datetime.datetime.now().strftime("%d %B %Y")

    prompt = f"""
    Convert the curated article summaries below into a single
    self-contained HTML newsletter.

    Requirements:
    - Full HTML document.
    - Clean magazine/newspaper style.
    - Exactly 2-column layout.
    - Use all curated articles.
    - No images.
    - Explicit text colors everywhere.
    - Footer with generation date.

    Newsletter Title: {newsletter_title}
    Generated On: {current_date}

    Curated Summaries:
    {curated_summaries}

    Return ONLY HTML.
    """

    response = model.invoke(prompt)
    return _extract_text(response)

# =========== RESPONSE HELPER ======================
def _extract_text(response):
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict) and "text" in last:
            return last["text"]
        return str(last)

    return str(content)

# ========== AGENT CREATION =================
agent = create_agent(
    model=model,
    tools=[
        weekly_article_collector,
        article_summarizer,
        newsletter_html_generator
    ]
)

# ============== MAIN AGENT =================
def main_agent(agent, query):
    """Orchestrates the full newsletter workflow."""

    prompt = """
    Your task is to orchestrate the full newsletter workflow.

    Steps:
    1. Call weekly_article_collector with max_results=6.
    2. Summarize EACH article using article_summarizer.
    3. Keep the best EXACTLY 6 curated articles whenever 6 are available.
    4. Combine title, summary, key points, category and url for each article.
    5. Call newsletter_html_generator once with the full collection.

    Return ONLY the final HTML.
    """

    prompt = prompt + "\n\n" + query

    response = agent.invoke({
        "messages": [{"role": "user", "content": prompt}]
    })

    return _extract_text(response["messages"][-1])

# ========== GENERATE NEWSLETTER ===============
if st.button("Generate Newsletter"):

    with st.spinner("Generating newsletter..."):

        user_query = (
            f"Create this week's newsletter covering the top trending news stories "
            f"across any topic/category. "
            f"Use max_results=6. "
            f"Newsletter Title: {newsletter_title}"
        )

        raw_code = main_agent(agent, user_query)

        code = (
            raw_code
            .replace("```html", "")
            .replace("```", "")
            .strip()
        )

        st.success("Newsletter generated successfully!")

        st.download_button(
            label="Download newsletter.html",
            data=code,
            file_name="newsletter.html",
            mime="text/html"
        )

        st.divider()
        st.subheader("Preview")

        st.components.v1.html(
            code,
            height=900,
            scrolling=True
        )
````
