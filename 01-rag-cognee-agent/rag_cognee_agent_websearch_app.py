import os
import asyncio
from dotenv import load_dotenv
from pathlib import Path  # Dosya yolunu bulmak için eklendi
import streamlit as st
import nest_asyncio

# --- Kütüphaneleri Güvenli Bir Şekilde Import Et ---
try:
    from PyPDF2 import PdfReader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    import cognee
except ImportError as e:
    st.error(f"Gerekli bir kütüphane eksik: {e}. Lütfen ortamınızı kontrol edin.")
    st.stop()


# --- 1. "KURŞUN GEÇİRMEZ" ORTAM DEĞİŞKENİ YÜKLEME ---

# YORUM: Bu blok, betiğin nerede çalıştırıldığından bağımsız olarak, 
# .env dosyasını her zaman betiğin kendi klasöründe arar.
script_dir = Path(__file__).parent
dotenv_path = script_dir / ".env"

if dotenv_path.is_file():
    load_dotenv(dotenv_path=dotenv_path)
    # Terminalde bir onay mesajı göster (hata ayıklama için)
    print(f".env dosyası şu yoldan yüklendi: {dotenv_path}")
else:
    # Eğer .env dosyası bulunamazsa, sistem geneli değişkenleri kullanmayı dene
    print(f".env dosyası '{dotenv_path}' konumunda bulunamadı. Sistem ortam değişkenleri kullanılacak.")
    load_dotenv()


LLM_API_KEY = os.getenv("LLM_API_KEY")
# Terminalde okunan anahtarı göster (hata ayıklama için)
print(f"[DEBUG] Okunan API Anahtarı: {LLM_API_KEY}") 

# --- Streamlit Arayüzü Başlangıcı ---
st.set_page_config(page_title="Cognee + RAG Chatbot", page_icon="🧠")
st.title("📄 Hafızalı PDF Chatbot (Cognee + FAISS + RAG)")

# YORUM: API Anahtarı en başta KESİN olarak kontrol ediliyor.
if not LLM_API_KEY:
    st.error("❌ HATA: LLM_API_KEY ortam değişkeni bulunamadı veya yüklenemedi!")
    st.error(f"Lütfen '{dotenv_path}' konumunda bir .env dosyası olduğundan ve içinde 'LLM_API_KEY=...' satırının bulunduğundan emin olun.")
    st.stop()

# Ortam değişkenlerinin geri kalanını al
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "custom")
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "https://openrouter.ai/api/v1")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "fastembed")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATA_ROOT_DIRECTORY = os.getenv("DATA_ROOT_DIRECTORY", "./cognee_data")

nest_asyncio.apply()

# --- Cognee Kütüphanesini Yapılandır (Versiyon Uyumluluğu ile) ---
try:
    if hasattr(cognee, 'configure'):
        cognee.configure(
            llm_api_key=LLM_API_KEY,
            llm_provider=LLM_PROVIDER,
            llm_model=LLM_MODEL,
            llm_endpoint=LLM_ENDPOINT,
            embedding_provider=EMBEDDING_PROVIDER,
            embedding_model=EMBEDDING_MODEL,
            data_root_directory=DATA_ROOT_DIRECTORY
        )
    else:
        cognee.config.llm_api_key = LLM_API_KEY
        cognee.config.llm_provider = LLM_PROVIDER
        cognee.config.llm_model = LLM_MODEL
        cognee.config.llm_endpoint = LLM_ENDPOINT
        cognee.config.embedding_provider = EMBEDDING_PROVIDER
        cognee.config.embedding_model = EMBEDDING_MODEL
        cognee.config.data_root_directory = DATA_ROOT_DIRECTORY
except Exception as e:
    st.error(f"Cognee yapılandırılırken bir hata oluştu: {e}")
    st.stop()


# --- Yardımcı Fonksiyonlar ---
def get_pdf_text(pdf_file):
    raw_text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
    except Exception as e:
        st.error(f"PDF okunurken hata oluştu: {e}")
        return None
    return raw_text

def get_vectorstore(raw_text):
    try:
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(raw_text)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Vektör veritabanı oluşturulurken hata oluştu: {e}")
        return None

def get_conversation_chain(vectorstore):
    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=LLM_API_KEY,
            base_url=LLM_ENDPOINT
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        return chain
    except Exception as e:
        st.error(f"Sohbet zinciri oluşturulurken hata oluştu: {e}")
        return None

# --- Ana Uygulama Mantığı ---

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

pdf_file = st.file_uploader("Bir PDF yükleyin ve işlenmesini bekleyin:", type=["pdf"])

if pdf_file:
    st.session_state.conversation = None
    st.session_state.chat_history = []
    with st.spinner("PDF işleniyor..."):
        raw_text = get_pdf_text(pdf_file)
        if raw_text:
            vectorstore = get_vectorstore(raw_text)
            if vectorstore:
                st.success("✅ PDF başarıyla işlendi. Şimdi soru sorabilirsiniz.")
                st.session_state.conversation = get_conversation_chain(vectorstore)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("❓ PDF hakkında bir soru sorun...")

if user_question:
    if st.session_state.conversation:
        try:
            with st.spinner("Cevap oluşturuluyor..."):
                st.chat_message("user").markdown(user_question)
                
                result = st.session_state.conversation({
                    "question": user_question,
                    "chat_history": st.session_state.chat_history
                })
                response = result["answer"]

                st.session_state.chat_history.append({"role": "user", "content": user_question})
                st.session_state.chat_history.append({"role": "assistant", "content": response})

                st.chat_message("assistant").markdown(response)

                with st.expander("📌 Kaynak metin parçaları"):
                    for i, doc in enumerate(result["source_documents"], 1):
                        st.markdown(f"**{i}. Parça:**")
                        st.write(doc.page_content)

        except Exception as e:
            st.error(f"Soru işlenirken bir hata oluştu: {e}")
            st.warning("API anahtarınızı (kredi/geçerlilik) veya internet bağlantınızı kontrol edin.")
    else:
        st.warning("Lütfen önce bir PDF dosyası yükleyin.")