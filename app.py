"""
app.py — Text-to-SQL Demo
Run after fine-tuning: streamlit run app.py
Loads your fine-tuned model from HuggingFace Hub
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

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
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-family: 'Space Mono', monospace;
        margin: 2px;
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
        font-family: 'DM Sans', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(hf_username):
    BASE_MODEL  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    LORA_MODEL  = f"{hf_username}/tinyllama-text2sql"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, LORA_MODEL)
    model.eval()
    return model, tokenizer


def generate_sql(model, tokenizer, question):
    prompt = f"""### Task: Convert the natural language question to a SQL query.

### Question: {question}

### SQL:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### SQL:" in generated:
        sql = generated.split("### SQL:")[-1].strip()
        return sql.split("\n")[0].strip()
    return generated.strip()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:1.1rem;font-weight:700;color:#60a5fa;margin-bottom:1rem">Text-to-SQL</div>', unsafe_allow_html=True)
    hf_username = st.text_input("HuggingFace username", value="asanepranav")
    load_btn    = st.button("Load model")

    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem;color:#4b5563;font-family:Space Mono,monospace;line-height:2">
    model info<br>
    base: TinyLlama-1.1B<br>
    method: LoRA (PEFT)<br>
    dataset: Spider SQL<br>
    rank r=8, alpha=16<br>
    trainable: ~0.5%
    </div>
    """, unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown('<div style="font-family:Space Mono,monospace;font-size:1.8rem;font-weight:700;color:#60a5fa">🛢️ Text → SQL</div>', unsafe_allow_html=True)
st.markdown('<div style="color:#6b7280;margin-bottom:2rem">Fine-tuned TinyLlama with LoRA/PEFT on Spider dataset</div>', unsafe_allow_html=True)

if "model" not in st.session_state:
    st.info("Enter your HuggingFace username in the sidebar and click Load model.")
elif load_btn:
    with st.spinner("Loading fine-tuned model from HuggingFace..."):
        model, tokenizer = load_model(hf_username)
        st.session_state.model     = model
        st.session_state.tokenizer = tokenizer
    st.success("Model loaded!")

if "model" in st.session_state:
    # Quick examples
    st.markdown("**Try an example:**")
    examples = [
        "How many singers are there?",
        "What are the names of all stadiums with capacity over 50000?",
        "Find all employees in the engineering department",
        "Show the average salary by department",
        "List all orders placed in 2023 sorted by total amount",
    ]
    cols = st.columns(3)
    selected = None
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, key=f"ex_{i}"):
            selected = ex

    st.markdown("---")
    question = st.text_area(
        "Your question",
        value=selected or "",
        placeholder="e.g. How many students have a GPA above 3.5?",
        height=80
    )

    if st.button("Generate SQL", key="gen"):
        if question.strip():
            with st.spinner("Generating..."):
                sql = generate_sql(st.session_state.model, st.session_state.tokenizer, question)
            st.markdown("**Generated SQL:**")
            st.markdown(f'<div class="sql-box">{sql}</div>', unsafe_allow_html=True)
            st.code(sql, language="sql")
        else:
            st.warning("Enter a question first.")
