import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from groq import Groq

# ================== CONFIG ==================
DB_DIR = "./vectorstore"
PDF_DIR = "./"   # PDFs stored in repo root
COLLECTION_NAME = "lawbot"
MODEL_NAME = "llama-3.1-8b-instant"   # ✅ Updated to supported Groq model

# ================== GROQ INIT ==================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ Please set GROQ_API_KEY in your Hugging Face Space Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ================== STREAMLIT UI ==================
st.set_page_config(page_title="Agentic RAG — Pakistan Law Bot", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #006600, #ffffff);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Agentic RAG — Pakistan Law Bot")
st.caption("Ask questions from Pakistan's laws and constitution.")

# ================== SESSION ==================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "collection" not in st.session_state:
    st.session_state.collection = None


# ================== HELPERS ==================
def load_pdfs(pdf_dir):
    docs = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            try:
                reader = PdfReader(os.path.join(pdf_dir, file))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                docs.append((file, text))
            except Exception as e:
                st.warning(f"⚠️ Could not read {file}: {e}")
    return docs


def build_index(force=False):
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device="cpu"   # ✅ force CPU
    )

    try:
        collection = chroma_client.get_collection(COLLECTION_NAME, embedding_function=embedding_func)
    except:
        collection = chroma_client.create_collection(COLLECTION_NAME, embedding_function=embedding_func)

    if force or collection.count() == 0:
        st.info("📚 Building index from PDFs...")
        try:
            collection.delete(where={"source": {"$exists": True}})
        except:
            pass

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = load_pdfs(PDF_DIR)

        for fname, rawtext in docs:
            chunks = splitter.split_text(rawtext)
            ids = [f"{fname}-{i}" for i in range(len(chunks))]
            collection.add(
                documents=chunks,
                ids=ids,
                metadatas=[{"source": fname} for _ in chunks]
            )

        st.success("✅ Index built and saved.")

    return collection


def retrieve(query, collection, k=3):
    results = collection.query(query_texts=[query], n_results=k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return list(zip(docs, metas))


def ask_llm(question, context):
    prompt = f"""You are a legal assistant specialized in Pakistan law.
Answer the following question using the provided context.
If unsure, say "I don’t know". Always cite the sources.
Question: {question}
Context:
{context}
Answer:"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=512,
    )
    return response.choices[0].message.content


# ================== CONTROLS ==================
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("♻️ Rebuild Index"):
        st.session_state.collection = build_index(force=True)

if st.session_state.collection is None:
    st.session_state.collection = build_index(force=False)

# ================== CHAT ==================
query = st.text_input("💬 Ask your legal question:")
if query:
    try:
        docs = retrieve(query, st.session_state.collection, k=3)
        context = "\n\n".join([d[0] for d in docs])
        answer = ask_llm(query, context)

        st.session_state.chat_history.append({"role": "user", "text": query})
        st.session_state.chat_history.append(
            {"role": "bot", "text": answer, "evidence": docs}
        )
    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# ================== DISPLAY CHAT ==================
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"👤 **You:** {msg['text']}")
    else:
        st.markdown(f"🤖 **LawBot:** {msg['text']}")
        if "evidence" in msg:
            with st.expander("📖 Retrieved Evidence"):
                for d, meta in msg["evidence"]:
                    st.markdown(f"- **{meta['source']}**: {d[:300]}...")
