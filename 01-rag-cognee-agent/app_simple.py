import os
from dotenv import load_dotenv
from pathlib import Path
import streamlit as st

# Gerekli kütüphaneleri import et
try:
    from PyPDF2 import PdfReader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
except ImportError as e:
    st.error(f"Gerekli bir kütüphane eksik: {e}. Lütfen 'py311env' ortamınızı kontrol edin.")
    st.stop()

# --- 1. Ortam Değişkenlerini Yükle ve Kontrol Et ---
script_dir = Path(__file__).parent
dotenv_path = script_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Streamlit Arayüzü ---
st.set_page_config(page_title="Basit RAG Chatbot", page_icon="📄")
st.title("📄 Basit PDF Chatbot (Sadece LangChain + FAISS)")

if not LLM_API_KEY:
    st.error("❌ HATA: LLM_API_KEY bulunamadı! Lütfen .env dosyanızı kontrol edin.")
    st.stop()

# --- Yardımcı Fonksiyonlar ---
@st.cache_resource(show_spinner="PDF okunuyor ve vektörler oluşturuluyor...")
def process_pdf(pdf_file):
    try:
        # Metni al
        raw_text = ""
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
        
        # Parçalara ayır
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(raw_text)
        
        # Vektör veritabanını oluştur
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"PDF işlenirken hata oluştu: {e}")
        return None

# --- Ana Uygulama Mantığı ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

pdf_file = st.file_uploader("Bir PDF yükleyin:", type=["pdf"])

if pdf_file:
    vectorstore = process_pdf(pdf_file)
    if vectorstore:
        st.success("✅ PDF başarıyla işlendi. Şimdi soru sorabilirsiniz.")
        
        # Sohbet zincirini oluştur
        llm = ChatOpenAI(model=LLM_MODEL, openai_api_key=LLM_API_KEY, base_url=LLM_ENDPOINT)
        st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )
        # Yeni PDF yüklendiğinde sohbet geçmişini temizle
        st.session_state.chat_history = []

# Sohbet geçmişini ekranda göster
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan soru al
user_question = st.chat_input("❓ PDF hakkında bir soru sorun...")

if user_question:
    if "conversation_chain" in st.session_state:
        try:
            with st.spinner("Cevap oluşturuluyor..."):
                result = st.session_state.conversation_chain({
                    "question": user_question,
                    "chat_history": st.session_state.chat_history
                })
                response = result["answer"]

                # Sohbet geçmişini güncelle
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                st.session_state.chat_history.append({"role": "assistant", "content": response})

                # Sayfayı yeniden çizerek yeni mesajları göster
                st.rerun()

        except Exception as e:
            st.error(f"Soru işlenirken bir hata oluştu: {e}")
            st.warning("API anahtarınızı (kredi/geçerlilik) veya internet bağlantınızı kontrol edin.")
    else:
        st.warning("Lütfen önce bir PDF dosyası yükleyin.")