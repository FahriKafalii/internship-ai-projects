import streamlit as st
import requests

# 🔐 OpenRouter ayarları
API_KEY = ""
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mistral-7b-instruct:free"

# 🎨 Arayüz başlığı
st.set_page_config(page_title="SağlıkGPT", page_icon="💊")
st.title("💊 Sağlık Asistanı")

# 🔁 Hafıza sistemi
if "memory" not in st.session_state:
    st.session_state.memory = []

# 🧠 Sistem prompt
system_prompt = {
    "role": "system",
    "content": (
        "Sen bir sağlık danışmanısın. Tüm konuşmaları hatırlıyormuş gibi davran. "
        "Kullanıcının adı, geçmişteki şikayetleri ve sorularını unutma. "
        "Sadece sağlıkla ilgili cevaplar ver. Gerekirse kullanıcıya geçmişe referans vererek cevap ver."
    )
}
if not any(msg["role"] == "system" for msg in st.session_state.memory):
    st.session_state.memory.insert(0, system_prompt)

# 💬 Kullanıcı girişi
user_input = st.text_input("Bir şey yazın...")

if user_input:
    st.session_state.memory.append({"role": "user", "content": user_input})

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": st.session_state.memory
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()
        reply = result["choices"][0]["message"]["content"]
        st.session_state.memory.append({"role": "assistant", "content": reply})
    except Exception as e:
        reply = f"❗ Hata oluştu: {e}"
        st.session_state.memory.append({"role": "assistant", "content": reply})

# 💬 Sohbet geçmişi
for msg in st.session_state.memory[1:]:  # system mesajını gösterme
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
