import streamlit as st
from groq import Groq

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Virtual Doctor — Groq AI",
    page_icon="🩺",
    layout="centered",
)

SYSTEM_PROMPT = (
    "You are an experienced, compassionate medical doctor. "
    "Explain things in clear, simple language, ask clarifying questions when needed, "
    "and always remind the user to consult a licensed physician for diagnosis or treatment. "
    "Do not prescribe specific medication dosages."
)

MODEL_OPTIONS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]

# ---------------------------------------------------------
# Sidebar — API key & settings
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    model = st.selectbox("Model", MODEL_OPTIONS, index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.caption("⚠️ For informational purposes only. Not a substitute for professional medical advice.")

# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.title("🩺 Virtual Doctor")
st.caption("Powered by Groq API — fast LLM inference")

api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None

if not api_key:
    st.error(
        "GROQ_API_KEY not found. Add it to `.streamlit/secrets.toml` locally, "
        "or in your app's Settings → Secrets on Streamlit Cloud."
    )
    st.stop()

client = Groq(api_key=api_key)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Describe your symptoms or ask a health question...")

if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build message list for the API (system prompt + full history)
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages

    # Stream the assistant's reply
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
                max_tokens=1024,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ Error communicating with Groq API: {e}"
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})