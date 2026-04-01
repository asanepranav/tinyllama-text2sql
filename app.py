"""
app.py — Text-to-SQL Demo (Groq API version)
No GPU needed — runs via Groq's free API
"""

import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Text-to-SQL", page_icon="🛢️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #0a0f1e; color: #e8e6df; }
    .sql-box {
        background: #0d1117;
        border: 1px solid #2563eb44;
        border-left: 3px solid #2563eb;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        font-family: 'Space Mono', monospace;
        font-size: 0.95rem;
        color: #60a5fa;
        margin: 12px 0;
    }
    .stButton > button {
        background: #2563eb !important;
        color: #fff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        width: 100% !important;
        padding: 0.6rem !important;
    }
    div[data-testid="stSidebar"] { background: #060d1a; border-right: 1px solid #1f2937; }
    .stTextArea textarea {
        background: #111827 !important;
        color: #e8e6df !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }
    .stTextInput > div > div > input {
        background: #111827 !important;
        color: #e8e6df !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:1.1rem;font-weight:700;color:#60a5fa;margin-bottom:1rem">Text-to-SQL</div>', unsafe_allow_html=True)

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        placeholder="gsk_..."
    )

    schema = st.text_area(
        "Database schema (optional)",
        placeholder="""e.g.
students(id, name, age, gpa, department)
courses(id, name, credits, professor)
enrollments(student_id, course_id, grade)""",
        height=160
    )

    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem;color:#4b5563;font-family:Space Mono,monospace;line-height:2">
    powered by<br>
    Groq LLaMA 3.1 70B<br><br>
    fine-tuned model<br>
    huggingface.co/<br>
    CopyNinja3223/<br>
    tinyllama-text2sql
    </div>
    """, unsafe_allow_html=True)


# ── SQL generation ─────────────────────────────────────────────────────────────
def generate_sql(question, schema, api_key):
    client = Groq(api_key=api_key)

    schema_section = f"\n\nDatabase Schema:\n{schema}" if schema.strip() else ""

    system_prompt = f"""You are an expert SQL query generator. Convert natural language questions into accurate SQL queries.
Rules:
- Return ONLY the SQL query, nothing else
- No explanations, no markdown, no backticks
- Use standard SQL syntax
- If no schema is provided, use reasonable table/column names{schema_section}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Convert to SQL: {question}"}
        ],
        temperature=0.1,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown('<div style="font-family:Space Mono,monospace;font-size:1.8rem;font-weight:700;color:#60a5fa">🛢️ Text → SQL</div>', unsafe_allow_html=True)
st.markdown('<div style="color:#6b7280;margin-bottom:2rem">Natural language to SQL — powered by Groq LLaMA 3.1</div>', unsafe_allow_html=True)

if not api_key:
    st.warning("Enter your Groq API key in the sidebar to get started.")
    st.stop()

# Quick examples
st.markdown("**Try an example:**")
examples = [
    "How many students are there?",
    "Show all students with GPA above 3.5",
    "Find the average salary by department",
    "List all orders from 2024 sorted by amount",
    "Which products have never been ordered?",
    "Count employees hired in each year",
]

cols = st.columns(3)
selected = None
for i, ex in enumerate(examples):
    if cols[i % 3].button(ex, key=f"ex_{i}"):
        selected = ex

st.markdown("---")

question = st.text_area(
    "Your question in plain English",
    value=selected or "",
    placeholder="e.g. Show me all customers from Mumbai who spent more than 5000",
    height=80
)

if st.button("Generate SQL ⚡"):
    if question.strip():
        with st.spinner("Generating SQL..."):
            try:
                sql = generate_sql(question, schema, api_key)
                st.markdown("**Generated SQL:**")
                st.markdown(f'<div class="sql-box">{sql}</div>', unsafe_allow_html=True)
                st.code(sql, language="sql")

                # Show copy-friendly version
                with st.expander("Copy SQL"):
                    st.text(sql)

            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Enter a question first.")

# History
if "history" not in st.session_state:
    st.session_state.history = []

if question.strip() and st.session_state.get("last_question") != question:
    st.session_state.last_question = question

if st.session_state.history:
    st.markdown("---")
    st.markdown("**Recent queries**")
    for item in reversed(st.session_state.history[-5:]):
        with st.expander(item["question"]):
            st.code(item["sql"], language="sql")
