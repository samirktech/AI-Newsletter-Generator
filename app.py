"""
AI Newsletter Generator - Streamlit App
Built using LangChain Agent (Tool Calling) + Gemini + Tavily
"""

import datetime
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from tavily import TavilyClient

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Newsletter Generator",
    page_icon="📰",
    layout="wide",
)

st.title("📰 AI Newsletter Generator")
st.caption("Powered by a LangChain Agent • Tavily Search • Gemini")

# ----------------------------------------------------------------------------
# API KEYS (secrets first, sidebar input as fallback)
# ----------------------------------------------------------------------------
def get_key(name: str) -> str:
    """Get an API key from st.secrets if available, else from sidebar input."""
    if name in st.secrets:
        return st.secrets[name]
    return st.session_state.get(name, "")


with st.sidebar:
    st.header("🔑 API Keys")
    st.caption(
        "Keys are read from Streamlit **secrets** if configured "
        "(recommended for deployment). Otherwise, enter them below "
        "— they stay in this browser session only."
    )

    if "TAVILY_API_KEY" not in st.secrets:
        st.session_state["TAVILY_API_KEY"] = st.text_input(
            "Tavily API Key", type="password", value=st.session_state.get("TAVILY_API_KEY", "")
        )
    if "GOOGLE_API_KEY" not in st.secrets:
        st.session_state["GOOGLE_API_KEY"] = st.text_input(
            "Google (Gemini) API Key", type="password", value=st.session_state.get("GOOGLE_API_KEY", "")
        )

    TAVILY_API_KEY = get_key("TAVILY_API_KEY")
    GOOGLE_API_KEY = get_key("GOOGLE_API_KEY")

    st.divider()
    st.header("⚙️ Settings")
    gemini_model_name = st.text_input("Gemini model", value="gemini-2.0-flash")
    max_results = st.slider("Articles to fetch", min_value=4, max_value=15, value=8)

    st.divider()
    st.caption(
        "Get a free Tavily key at [tavily.com](https://tavily.com) "
        "and a Gemini key at [aistudio.google.com](https://aistudio.google.com/apikey)."
    )

# ----------------------------------------------------------------------------
# MAIN INPUTS
# ----------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])
with col1:
    newsletter_title = st.text_input("Newsletter Title", value="Weekly Digest")
with col2:
    topic = st.text_input("Topic (optional — leave blank for general top news)", value="")

generate_clicked = st.button("🚀 Generate Newsletter", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# TOOLS (same logic as the notebook)
# ----------------------------------------------------------------------------
def build_tools(model, tavily_api_key: str):
    def weekly_article_collector(topic: str | None = None, max_results: int = 8):
        """Searches the web for the top articles published in the current week
        using the Tavily search API. If a specific topic/niche is given it
        narrows the search to that topic, otherwise it fetches general top
        trending news headlines across any topic for the week. Returns
        article metadata like title, url, content and published date."""
        query = topic if topic else "top trending world news headlines this week"

        client = TavilyClient(api_key=tavily_api_key)
        response = client.search(
            query=query,
            topic="news",
            time_range="week",
            max_results=max_results,
            include_answer=False,
        )

        articles = []
        for result in response.get("results", []):
            articles.append(
                {
                    "title": result.get("title"),
                    "url": result.get("url"),
                    "content": result.get("content"),
                    "published_date": result.get("published_date", "N/A"),
                }
            )
        return articles

    def article_summarizer(article_text: str, article_title: str = "Untitled"):
        """Takes article text and title and produces a concise summary, key
        points, category and relevance score (out of 10) using the LLM."""
        prompt = f"""You are a professional newsletter editor.
        Summarize the article below in 3-4 concise lines,
        then list 2-3 key points as bullets, assign a single
        category (Tech/Business/Science/World/Other) and give
        a relevance score out of 10 for a general tech audience.

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

    def newsletter_html_generator(curated_summaries: str, newsletter_title: str = "Weekly Newsletter"):
        """Converts curated article summaries into a styled, self-contained
        HTML newsletter page suitable for email or web publishing."""
        current_date = datetime.datetime.now().strftime("%d %B %Y")

        prompt = f"""Convert the curated article summaries below into a single
        self-contained HTML page styled like a printed magazine/school
        newsletter front page. Return a full HTML document with a <style>
        block in the <head> (CSS does not need to be inline), max content
        width around 900px, centered on the page with a soft off-white
        background inside a bordered page frame.

        Do NOT include any <img> tags, background-image styles, or image
        urls anywhere in the output - text and colored boxes only, no photos.

        Follow this exact structure and style:
        - Top-left corner: a small bordered box showing the exact text
          "{current_date}" as the generation/publish date.
        - Masthead: one huge bold centered title showing
          "{newsletter_title}" (60-90px, wide letter-spacing, like a
          newspaper nameplate).
        - Sub-banner: a full-width colored horizontal band directly below
          the masthead with small bold centered uppercase tagline text
          summarizing that this covers this week's top stories across
          any topic/category (do not hardcode a single subject like "AI"
          in the tagline - infer a short general tagline from the actual
          mix of categories present in the curated summaries below).
        - Below the banner, use a CSS grid with exactly 2 columns
          (display: grid; grid-template-columns: 1fr 1fr; column-gap and
          row-gap around 24px) to lay out one section per curated article.
          Do NOT create a single "top story" block - turn every curated
          article into its own section: a bold uppercase heading (short,
          based on the article's own title/category) followed by a
          paragraph using that article's summary and key points as body
          text, spanning whatever categories the curated summaries
          actually cover (world news, tech, business, science, etc).
        - CRITICAL - no empty or blank cells: count the curated articles
          first. If the count is odd, make the LAST article's section span
          both columns (grid-column: 1 / -1) instead of leaving an empty
          cell next to it. Never render an empty box, an empty <div>, or a
          solid-colored block with no text in it anywhere on the page -
          every colored box must contain real content from the curated
          summaries. If there are not enough curated summaries to fill a
          section you were going to add, simply omit that section instead
          of leaving it blank.
        - For 2-3 of the sections, add a short colored accent line
          (in a highlight color) above the paragraph showing that article's
          own category or relevance score.
        - At the bottom of each column, add one pale colored info box
          (light blue/light pink) containing that source article's category
          and a bold "Read More" link pointing to the real article url -
          only add this info box if there is a real article category and
          url to put inside it, never as an empty filler box.
          Use the real curated article titles, summaries, categories and
          urls throughout - never placeholder/lorem ipsum text.
        - Footer: a full-width colored strip at the very bottom with small
          bold centered white text reading something like "Compiled
          automatically by a multi-agent AI pipeline | Generated on
          {current_date}".
        Give final response strictly in HTML only, no markdown, no code fences.

        Newsletter Title: {newsletter_title}
        Generated On: {current_date}
        Curated Summaries (multiple topics/categories from this week): {curated_summaries}
        """
        response = model.invoke(prompt)
        return _extract_text(response)

    return [weekly_article_collector, article_summarizer, newsletter_html_generator]


def _extract_text(response) -> str:
    """Safely extract text from a LangChain chat model response, regardless
    of whether .content is a plain string or a list of content blocks."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict) and "text" in last:
            return last["text"]
        return str(last)
    return str(content)


def run_agent(leader_agent, query: str) -> str:
    prompt = """Your task is to orchestrate the full newsletter workflow
    based on the instructions given below:
    1. Call the weekly_article_collector tool with max_results as given. Only
       pass a specific "topic" argument if the instructions below name
       one explicitly - otherwise call it with no topic (topic=None) so
       it fetches general top trending news headlines across ANY subject
       for the week, not a single fixed niche.
    2. Call the article_summarizer tool separately on EACH collected
       article to get its summary, key points, category and relevance
       score.
    3. Drop low relevance (below 6/10) articles, keep only the best
       4-6 curated ones, ideally spanning a few different categories.
    4. Combine all the remaining curated summaries (title, summary,
       key points, category, url for each) into one collection, then
       call the newsletter_html_generator tool once with that full
       collection so all curated topics appear in the final newsletter.
    Give the final response output strictly in HTML, no markdowns,
    no code fences, no explanation text before or after the HTML.
    Instructions given below:
    """
    prompt = prompt + query

    response = leader_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return _extract_text(response["messages"][-1])


def clean_html(code: str) -> str:
    """Strip accidental markdown code fences if the model adds them."""
    code = code.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[-1]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    return code.strip()


# ----------------------------------------------------------------------------
# GENERATE
# ----------------------------------------------------------------------------
if generate_clicked:
    if not TAVILY_API_KEY or not GOOGLE_API_KEY:
        st.error("Please provide both a Tavily API key and a Google (Gemini) API key in the sidebar.")
        st.stop()

    try:
        with st.spinner("Setting up the agent..."):
            model = ChatGoogleGenerativeAI(model=gemini_model_name, google_api_key=GOOGLE_API_KEY)
            tools = build_tools(model, TAVILY_API_KEY)
            leader_agent = create_agent(model=model, tools=tools)

        with st.spinner("Collecting, summarizing and building your newsletter... this can take a minute."):
            user_query = (
                f"Create this week's newsletter covering the top trending "
                f"news stories of the week"
                + (f" about: {topic}" if topic.strip() else " across any topic/category (do not restrict to a single subject)")
                + f".\nUse max_results={max_results} when collecting articles."
                + f"\nNewsletter Title: {newsletter_title}"
            )
            html_code = clean_html(run_agent(leader_agent, user_query))

        st.session_state["last_newsletter_html"] = html_code
        st.success("Newsletter generated!")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

# ----------------------------------------------------------------------------
# DISPLAY + DOWNLOAD
# ----------------------------------------------------------------------------
if "last_newsletter_html" in st.session_state:
    html_code = st.session_state["last_newsletter_html"]

    st.download_button(
        "⬇️ Download newsletter.html",
        data=html_code,
        file_name="newsletter.html",
        mime="text/html",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Preview")
    st.components.v1.html(html_code, height=900, scrolling=True)

    with st.expander("View raw HTML"):
        st.code(html_code, language="html")
