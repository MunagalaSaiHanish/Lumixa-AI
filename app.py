import streamlit as st

from services.youtube_service import (
    extract_video_id
)

from services.transcript_service import (
    get_transcript,
    transcript_to_text,
    transcript_with_timestamps
)

from services.chunk_service import (
    chunk_transcript
)

from services.llm_service import (
    summarize,
    summarize_stream,
    generate_insights,
    ask_question,
    ask_question_stream
)

from services.youtube_metadata import (
    get_video_metadata
)

from services.export_service import (
    generate_pdf
)

from services.context_builder import (
    build_context
)

from services.pdf_service import (
    extract_pdf_text
)

from services.website_service import (
    extract_website_text
)

from services.knowledge_base import (
    KnowledgeBase
)

from services.memory_service import (
    MemoryService
)

# page config
st.set_page_config(
    page_title="Lumixa AI",
    page_icon="assets/icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,500;0,600;1,500;1,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ink: #12141C;
    --ink-raised: #1A1D28;
    --ink-line: #2B2F3F;
    --parchment: #ECE6D8;
    --parchment-dim: #A8A395;
    --slate: #838AA0;
    --brass: #D8A54A;
    --brass-soft: rgba(216, 165, 74, 0.14);
    --ember: #E2704A;
    --moss: #6FA287;
}

/* ---------- base ---------- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    height: 100vh;
    overflow: hidden;
    background: var(--ink) !important;
    color: var(--parchment) !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: #0D0F16 !important;
    border-right: 1px solid var(--ink-line);
}

/* Streamlit's fixed top bar (Deploy / menu) sits as an overlay above
   the content. We deliberately do NOT touch .block-container's
   top padding — that's Streamlit's own built-in clearance for the
   header, and overriding it is what hides content underneath the
   bar. We only constrain overall height (to fit the viewport with
   no page-level scroll) and the side/bottom padding. */
[data-testid="stHeader"] {
    background: var(--ink) !important;
}

.block-container {
    height: calc(100vh - 3.5rem) !important;
    max-height: calc(100vh - 3.5rem) !important;
    padding-top: 3.9rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    padding-bottom: 0.5rem !important;
    overflow: hidden !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
}

/* The row holding the reading pane + source panel grows to fill
   whatever vertical space is left after the header's own padding,
   and each column stretches to full height so its card can fill it. */
[data-testid="stMain"] [data-testid="stHorizontalBlock"] {
    flex: 1 1 auto !important;
    min-height: 0 !important;
}
[data-testid="stMain"] [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    height: 100% !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="stMain"] [data-testid="column"] > div[data-testid="stVerticalBlock"] {
    height: 100% !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="stMain"] [data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"] {
    flex: 1 1 auto !important;
    min-height: 0 !important;
}

/* The two fixed-pixel-height cards (source intake + reading pane)
   become fluid: they fill their column instead of a hardcoded
   632px, and scroll internally — never the page — if their content
   runs long. */
[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"] > div[style*="height"] {
    height: 100% !important;
    max-height: 100% !important;
    flex: 1 1 auto !important;
    min-height: 0 !important;
    overflow-y: auto !important;
}

[data-testid="stVerticalBlock"] { gap: 0.6rem; }

h1 {
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
    font-size: 1.65rem !important;
    color: var(--parchment) !important;
    letter-spacing: -0.015em;
    margin-bottom: 0.5rem !important;
}

h2 {
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
    font-size: 1.40rem !important;
    color: var(--parchment) !important;
    letter-spacing: -0.010em;
    margin-bottom: 0.4rem !important;
}

h3 {
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
    font-size: 1.15rem !important;
    color: var(--parchment) !important;
    letter-spacing: -0.005em;
    margin-bottom: 0.3rem !important;
}

p, li, label, [data-testid="stMarkdownContainer"] p {
    font-size: 0.90rem !important;
    color: var(--parchment) !important;
}

a { color: var(--brass) !important; }

hr { border-color: var(--ink-line) !important; }

/* ---------- eyebrow labels (structural, not decorative) ---------- */
.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--brass);
    margin: 0 0 0.35rem 0;
}

/* ---------- brand lockup: the lamp glow ---------- */
.brand { position: relative; text-align: center; padding: 1.2rem 0 0.8rem; }
.brand-glow {
    position: absolute;
    top: -18px; left: 50%;
    transform: translateX(-50%);
    width: 150px; height: 150px;
    background: radial-gradient(circle, rgba(216,165,74,0.30), transparent 68%);
    filter: blur(4px);
    z-index: 0;
}
.brand-mark {
    position: relative; z-index: 1;
    font-family: 'Fraunces', serif;
    font-style: italic;
    font-weight: 600;
    font-size: 1.85rem;
    color: var(--parchment);
}
.brand-ai {
    font-family: 'IBM Plex Mono', monospace;
    font-style: normal;
    font-size: 0.75rem;
    color: var(--brass);
    vertical-align: super;
    margin-left: 3px;
    letter-spacing: 0.05em;
}
.brand-tagline {
    position: relative; z-index: 1;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.64rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--slate);
    margin-top: 0.35rem;
}

/* ---------- cards / bordered containers ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--ink-raised) !important;
    border: 1px solid var(--ink-line) !important;
    border-top: 2px solid var(--brass) !important;
    border-radius: 8px !important;
}

/* ---------- inputs ---------- */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: var(--ink) !important;
    color: var(--parchment) !important;
    border: 1px solid var(--ink-line) !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--brass) !important;
    box-shadow: 0 0 0 1px var(--brass) !important;
}
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stFileUploader"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em;
    color: var(--slate) !important;
    text-transform: uppercase;
}

[data-testid="stFileUploader"] section {
    background: var(--ink) !important;
    border: 1px dashed var(--ink-line) !important;
    border-radius: 6px !important;
}

/* ---------- buttons ---------- */
[data-testid="stBaseButton-primary"], button[kind="primary"] {
    background: var(--brass) !important;
    color: var(--ink) !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    transition: box-shadow 0.15s ease;
}
[data-testid="stBaseButton-primary"]:hover, button[kind="primary"]:hover {
    box-shadow: 0 0 14px rgba(216, 165, 74, 0.45);
}
[data-testid="stBaseButton-secondary"], button[kind="secondary"] {
    background: transparent !important;
    color: var(--parchment) !important;
    border: 1px solid var(--ink-line) !important;
    border-radius: 6px !important;
}
[data-testid="stBaseButton-secondary"]:hover, button[kind="secondary"]:hover {
    border-color: var(--brass) !important;
    color: var(--brass) !important;
}

/* ---------- status dot ---------- */
.status-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.04em;
}
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-right: 7px;
}
.status-dot--empty { background: var(--ember); }
.status-dot--ready { background: var(--moss); box-shadow: 0 0 6px var(--moss); }

/* ---------- topic tags (excerpt annotations, not pills) ---------- */
.topic-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.03em;
    background: var(--brass-soft);
    color: var(--brass);
    padding: 3px 9px;
    margin: 3px 4px 3px 0;
    display: inline-block;
    border: 1px solid rgba(216, 165, 74, 0.3);
    border-radius: 4px;
}

.channel-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--brass);
    margin: 4px 0;
}

/* ---------- alerts ---------- */
[data-testid="stAlert"] {
    background: var(--ink-raised) !important;
    border: 1px solid var(--ink-line) !important;
    border-left: 3px solid var(--brass) !important;
    border-radius: 6px !important;
    color: var(--parchment) !important;
}

/* ---------- expander ---------- */
[data-testid="stExpander"] {
    border: 1px solid var(--ink-line) !important;
    border-radius: 6px !important;
    background: var(--ink) !important;
}

/* ---------- chat ---------- */
[data-testid="stChatMessage"] {
    background: var(--ink-raised) !important;
    border: 1px solid var(--ink-line) !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] textarea {
    background: var(--ink-raised) !important;
    color: var(--parchment) !important;
    border: 1px solid var(--ink-line) !important;
}

/* WhatsApp User Message Style Alignment */
div[data-testid="stChatMessage"]:has(.whatsapp-user-message) {
    flex-direction: row-reverse !important;
    background-color: var(--brass-soft) !important; /* Soft brass like Takeaways */
    border: 1px solid rgba(216, 165, 74, 0.3) !important;
    border-radius: 12px 12px 0px 12px !important;
    margin-left: auto !important;
    margin-right: 0.5rem !important;
    max-width: 80% !important;
    width: fit-content !important;
    padding: 8px 12px !important;
}

/* Hide user avatar in WhatsApp style */
div[data-testid="stChatMessage"]:has(.whatsapp-user-message) [data-testid="stChatMessageAvatar"] {
    display: none !important;
}

/* Clean up padding for the content container */
div[data-testid="stChatMessage"]:has(.whatsapp-user-message) [data-testid="stChatMessageContent"] {
    padding: 0 !important;
    margin: 0 !important;
}

/* ================= CHAT INPUT ================= */

[data-testid="stBottom"]{
    padding:0.4rem 1rem !important;
    background:var(--ink);
}

[data-testid="stBottomBlockContainer"]{
    padding:0 !important;
    max-width:100% !important;
}

[data-testid="stChatInput"]{
    width:100% !important;
    max-width:100% !important;
    box-sizing:border-box !important;
}

/* Streamlit 1.59.2 nests the layout container one level deeper:
   stChatInput > div (wrapper, 1 child) > div (actual flex row).
   Target the grandchild to switch from column to row. */
[data-testid="stChatInput"] > div > div {
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 8px !important;
}

/* Textarea wrapper: grow to fill available width */
[data-testid="stChatInput"] > div > div > div:first-child {
    width: auto !important;
    flex: 1 1 auto !important;
    min-width: 0 !important;
}

/* Button wrapper: shrink to fit */
[data-testid="stChatInput"] > div > div > div:last-child {
    width: auto !important;
    flex: 0 0 auto !important;
}

[data-testid="stChatInput"] textarea{
    min-height:42px !important;
    max-height:42px !important;
    resize:none !important;
    overflow:hidden !important;
}

[data-testid="stChatInputSubmitButton"]{
    margin:0 !important;
    flex-shrink:0 !important;
}

/* ---------- misc ---------- */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: var(--ink-line); border-radius: 4px; }
::-webkit-scrollbar-track { background: transparent; }

/* Prevent horizontal page scroll */
html, body { overflow-x: hidden !important; }

/* ===================== RESPONSIVE: TABLET (<=1024px) ===================== */
/* Below this width the two panels stack instead of sitting
   side-by-side, so there's no single viewport-height row to fit
   them both into. We drop the no-scroll layout here and let the
   page scroll normally, same as mobile. */
@media (max-width: 1024px) {
    html, body, [data-testid="stAppViewContainer"] { height: auto !important; overflow: auto !important; }
    .block-container {
        height: auto !important;
        max-height: none !important;
        padding-top: 3.9rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 3.5rem !important;
        overflow: visible !important;
        display: block !important;
    }
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        flex: none !important;
    }
    [data-testid="stHorizontalBlock"] > div { width: 100% !important; flex: 1 1 100% !important; height: auto !important; }
    [data-testid="stMain"] [data-testid="column"] > div[data-testid="stVerticalBlock"] { height: auto !important; }
    [data-testid="stMain"] [data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"] { flex: none !important; }
    [data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"] > div[style*="height"] {
        height: auto !important;
        max-height: 60vh !important;
    }
}

/* ===================== RESPONSIVE: MOBILE (<=640px) ===================== */
@media (max-width: 640px) {
    .block-container {
        padding-top: 3.9rem !important;
        padding-left: 0.6rem !important;
        padding-right: 0.6rem !important;
        padding-bottom: 3.5rem !important;
    }
    h3 { font-size: 1.05rem !important; }
    .brand-mark { font-size: 1.5rem; }
    .topic-tag { font-size: 0.64rem; padding: 2px 7px; }
    [data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"] > div[style*="height"] {
        height: auto !important;
        max-height: 60vh !important;
        overflow-y: auto !important;
    }
}
</style>
""", unsafe_allow_html=True)

# session state
DEFAULT_STATE = {
    "knowledge_base": KnowledgeBase(),
    "memory": MemoryService(),
    "metadata": None,
    "summary": "",
    "takeaways": [],
    "topics": [],
    "transcript": "",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

def clear_workspace():
    st.session_state.knowledge_base.clear()
    st.session_state.memory.clear()
    st.session_state.summary = ""
    st.session_state.takeaways = []
    st.session_state.topics = []
    st.session_state.transcript = ""
    st.session_state.metadata = None

# sidebar
with st.sidebar:
    st.image("assets/logo.png", width="stretch")
    st.divider()

    st.markdown('<div class="eyebrow">Session</div>', unsafe_allow_html=True)
    history_container = st.container(height=200)
    with history_container:
        messages = st.session_state.memory.get_messages()
        if len(messages) == 0:
            st.caption("No conversation yet.")
        else:
            conv_title = st.session_state.topics[0] if st.session_state.topics else "Current session"
            st.button(conv_title, key="conv_btn_0", use_container_width=True)

    st.divider()
    st.button(
        "Clear workspace",
        key="clear_workspace_btn",
        on_click=clear_workspace,
        use_container_width=True
    )

# main layout
center_panel, source_panel = st.columns([8, 2.5])

# right panel
with source_panel:
    with st.container(height=566, border=True):
        st.subheader("Knowledge sources")

        video_url = st.text_input("Video URL", placeholder="https://www.youtube.com/watch?v=...")
        uploaded_files = st.file_uploader("Files", type=["pdf", "docx", "pptx", "xlsx", "xls", "csv", "txt", "md"], accept_multiple_files=True)
        website_url = st.text_input("Website URL", placeholder="https://example.com")
        notes_text = st.text_area(
            "Notes",
            height=120,
            placeholder="Paste notes, documentation, or any text here..."
        )

        analyze_clicked = st.button("Analyze sources", use_container_width=True, type="primary")

        if st.session_state.knowledge_base.index is None:
            st.markdown(
                '<div class="status-line"><span class="status-dot status-dot--empty"></span>No source loaded</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="status-line"><span class="status-dot status-dot--ready"></span>Knowledge base ready</div>',
                unsafe_allow_html=True
            )

# center panel
with center_panel:
    workspace_container = st.container(height=566, border=True)
    with workspace_container:

        if not st.session_state.summary and not st.session_state.metadata and not analyze_clicked:
            st.info("Add a source on the right, then analyze it to begin.")

        # analyze pipeline
        if analyze_clicked:
            kb = st.session_state.knowledge_base
            kb.clear()
            st.session_state.memory.clear()
            st.session_state.metadata = None
            st.session_state.summary = ""
            st.session_state.takeaways = []
            st.session_state.topics = []
            st.session_state.transcript = ""

            source_loaded = False
            combined_text = ""

            # Use st.status to show step-by-step progress natively
            with st.status("Analyzing Knowledge...", expanded=True) as status:
                # YouTube
                if video_url.strip():
                    video_id = extract_video_id(video_url)
                    if video_id is None:
                        st.error("That doesn't look like a valid YouTube URL.")
                    else:
                        try:
                            metadata = get_video_metadata(video_url)
                            try:
                                transcript = get_transcript(video_id)
                                transcript_text = transcript_to_text(transcript)
                                st.session_state.transcript = transcript_text
                                transcript_chunks = chunk_transcript(transcript_with_timestamps(transcript))
                            except Exception as e:
                                print(f"Transcript extraction failed: {e}. Falling back to description.")
                                desc = metadata.get("description", "")
                                if not desc or desc.strip() == "":
                                    desc = "No transcript or description available for this video."
                                fallback_text = f"Video Title: {metadata.get('title')}\n\nVideo Description:\n{desc}"
                                st.session_state.transcript = f"[No transcript available. Falling back to video description]\n\n{fallback_text}"
                                from services.transcript_service import TranscriptSegment
                                transcript = [TranscriptSegment(fallback_text, 0.0, 10.0)]
                                transcript_chunks = chunk_transcript(transcript_with_timestamps(transcript))
                            for chunk in transcript_chunks:
                                chunk["metadata"] = metadata
                            kb.add_chunks(transcript_chunks)
                            st.session_state.metadata = metadata
                            combined_text += st.session_state.transcript + "\n\n"
                            source_loaded = True
                        except Exception as e:
                            st.error(f"Couldn't reach that video: {e}")

                # Files (mixed formats)
                if uploaded_files:
                    file_names = []
                    for file in uploaded_files:
                        name = file.name.lower()
                        try:
                            doc = None
                            if name.endswith('.pdf'):
                                from services.loaders.pdf_loader import load_pdf
                                doc = load_pdf(file)
                            elif name.endswith('.docx'):
                                from services.loaders.docx_loader import load_docx
                                doc = load_docx(file)
                            elif name.endswith('.pptx'):
                                from services.loaders.pptx_loader import load_pptx
                                doc = load_pptx(file)
                            elif name.endswith('.xlsx') or name.endswith('.xls'):
                                from services.loaders.excel_loader import load_excel
                                doc = load_excel(file)
                            elif name.endswith('.csv'):
                                from services.loaders.csv_loader import load_csv
                                doc = load_csv(file)
                            elif name.endswith('.txt'):
                                from services.loaders.txt_loader import load_txt
                                doc = load_txt(file)
                            elif name.endswith('.md'):
                                from services.loaders.markdown_loader import load_markdown
                                doc = load_markdown(file)
                            
                            if doc and doc.content.strip():
                                kb.add_document(
                                    text=doc.content,
                                    metadata=doc.metadata
                                )
                                combined_text += doc.content + "\n\n"
                                source_loaded = True
                                file_names.append(file.name)
                            else:
                                st.warning(f"No readable text found in {file.name}.")
                        except Exception as e:
                            st.error(f"Couldn't read file {file.name}: {e}")
                    if file_names:
                        st.session_state.metadata = {
                            "source": "files",
                            "title": "Uploaded Documents" if len(file_names) > 1 else file_names[0],
                            "channel": f"{len(file_names)} files loaded" if len(file_names) > 1 else "Document File"
                        }

                # Website
                if website_url.strip():
                    try:
                        website_text = extract_website_text(website_url)
                        if website_text:
                            kb.add_document(
                                text=website_text,
                                metadata={"source": "website", "title": website_url, "url": website_url}
                            )
                            combined_text += website_text + "\n\n"
                            source_loaded = True
                        else:
                            st.warning("No readable text found at that URL.")
                    except Exception as e:
                        st.error(f"Couldn't reach that page: {e}")

                # Notes
                if notes_text.strip():
                    kb.add_document(text=notes_text, metadata={"source": "notes", "title": "Notes"})
                    combined_text += notes_text + "\n\n"
                    source_loaded = True

                if not source_loaded:
                    st.warning("Add at least one source before analyzing.")
                    status.update(label="Analysis failed.", state="error")
                    st.stop()

                status.update(label="Analyzing Knowledge Complete.", state="complete")

            # summary
            try:
                st.markdown('<div class="eyebrow">Generating Summary...</div>', unsafe_allow_html=True)
                st.session_state.summary = st.write_stream(summarize_stream(combined_text))
            except Exception as e:
                st.error(f"Couldn't generate a summary: {e}")
                st.stop()

            # Generate insights
            try:
                with st.spinner("Finding takeaways..."):
                    insights = generate_insights(st.session_state.summary)
                st.session_state.takeaways = insights.get("takeaways", [])
                st.session_state.topics = insights.get("topics", [])
            except Exception as e:
                st.warning(f"Summary is ready, but takeaways couldn't be generated: {e}")
                st.session_state.takeaways = []
                st.session_state.topics = []

            st.success("Knowledge base ready.")
            st.rerun()

        # metadata card
        if st.session_state.metadata:
            with st.container(border=True):
                metadata = st.session_state.metadata
                col1, col2 = st.columns([1.2, 3])
                with col1:
                    thumbnail = metadata.get("thumbnail", "")
                    if thumbnail:
                        st.image(thumbnail, width="stretch")
                with col2:
                    st.markdown(f"### {metadata.get('title', 'Knowledge Source')}")
                    if metadata.get("channel"):
                        st.markdown(f"<p class='channel-name'>{metadata['channel']}</p>", unsafe_allow_html=True)
                    if metadata.get("url"):
                        st.markdown(f"[{metadata['url']}]({metadata['url']})")

        # summary
        if st.session_state.summary:
            with st.container(border=True):
                st.markdown('<div class="eyebrow">Summary</div>', unsafe_allow_html=True)
                st.write(st.session_state.summary)

        # takeaways
        if st.session_state.takeaways or st.session_state.topics:
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                if st.session_state.takeaways:
                    with st.container(border=True):
                        st.markdown('<div class="eyebrow">Takeaways</div>', unsafe_allow_html=True)
                        for takeaway in st.session_state.takeaways:
                            st.write(f"— {takeaway}")
            with t_col2:
                if st.session_state.topics:
                    with st.container(border=True):
                        st.markdown('<div class="eyebrow">Topics</div>', unsafe_allow_html=True)
                        tags_html = "".join(f"<span class='topic-tag'>{t}</span>" for t in st.session_state.topics)
                        st.markdown(tags_html, unsafe_allow_html=True)

        # export
        if st.session_state.summary:
            with st.container(border=True):
                st.markdown('<div class="eyebrow">Export</div>', unsafe_allow_html=True)
                report_title = st.session_state.topics[0] if st.session_state.topics else "Lumixa AI Report"
                try:
                    pdf = generate_pdf(
                        metadata=st.session_state.metadata,
                        topic=report_title,
                        summary=st.session_state.summary,
                        takeaways=st.session_state.takeaways,
                        topics=st.session_state.topics
                    )
                    st.download_button(
                        label="Download PDF",
                        data=pdf,
                        file_name=f"{report_title}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Couldn't generate the PDF: {e}")

        # transcript
        if st.session_state.transcript:
            with st.container(border=True):
                st.markdown('<div class="eyebrow">Transcript</div>', unsafe_allow_html=True)
                with st.expander("View transcript"):
                    st.text_area(
                        "Transcript",
                        value=st.session_state.transcript,
                        height=250,
                        label_visibility="collapsed"
                    )

        # chat history
        if st.session_state.knowledge_base.index is not None:
            for message in st.session_state.memory.get_messages():
                with st.chat_message(message["role"]):
                    if message["role"] == "user":
                        st.markdown('<div class="whatsapp-user-message"></div>', unsafe_allow_html=True)
                    st.write(message["content"])

            # handle active question
            if "active_question" in st.session_state:
                active_q = st.session_state.active_question
                with st.chat_message("user"):
                    st.markdown('<div class="whatsapp-user-message"></div>', unsafe_allow_html=True)
                    st.write(active_q)

                try:
                    kb = st.session_state.knowledge_base
                    memory = st.session_state.memory

                    with st.status("Searching Vector Database...", expanded=True) as status:
                        retrieved_documents = kb.retrieve(question=active_q, top_k=5)
                        context_data = build_context(retrieved_documents)
                        status.update(label="Generating Response...", state="complete")

                    with st.chat_message("assistant"):
                        answer = st.write_stream(
                            ask_question_stream(
                                question=active_q,
                                context=context_data["context"],
                                messages=memory.get_recent_messages()
                            )
                        )

                        citation_block = ""
                        if context_data["sources"]:
                            citation_block += "\n\n**Sources:**\n"
                            for source in context_data["sources"]:
                                citation_block += f"— {source}\n"

                        youtube_chunks = [doc for doc in retrieved_documents if "start" in doc]
                        if youtube_chunks:
                            citation_block += "\n\n**Mentioned in video:**\n"
                            displayed = set()
                            for chunk in youtube_chunks:
                                timestamp = int(chunk["start"])
                                if timestamp in displayed:
                                    continue
                                displayed.add(timestamp)
                                minutes, seconds = timestamp // 60, timestamp % 60
                                citation_block += f"— {minutes:02}:{seconds:02}\n"

                        if citation_block:
                            st.markdown(citation_block)

                    # save to memory
                    memory.add_message("user", active_q)
                    memory.add_message("assistant", answer + citation_block)

                except Exception as e:
                    st.error(f"Couldn't answer that: {e}")

                del st.session_state.active_question
                st.rerun()

# chat input
question = st.chat_input("Ask Lumixa...")

if question:
    if st.session_state.knowledge_base.index is None:
        st.error("Add and analyze a source before asking a question.")
    else:
        st.session_state.active_question = question
        st.rerun()
