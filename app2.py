import streamlit as st
import sounddevice as sd
import numpy as np
import whisper
import tempfile
import wavio
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
import os
from gtts import gTTS  # Biblioteca para converter texto em áudio

# Carregar variáveis de ambiente
load_dotenv()

# Definir modelo da GROQ
llm = "groq/llama3-8b-8192"

# Configurações do áudio
SAMPLERATE = 16000  
CHANNELS = 1
SAMPWIDTH = 2  

# Estado global para armazenar conversa
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Função para iniciar gravação
def start_recording():
    st.session_state.recording = True
    st.write("🎤 Gravando... Fale agora!")

    duration = 5  
    st.session_state.audio_data = sd.rec(
        int(duration * SAMPLERATE),
        samplerate=SAMPLERATE,
        channels=CHANNELS,
        dtype=np.int16
    )
    
    sd.wait()
    stop_recording()

# Função para parar gravação, transcrever e enviar para o chatbot
def stop_recording():
    st.session_state.recording = False

    if st.session_state.audio_data is None or len(st.session_state.audio_data) == 0:
        st.error("Nenhum áudio foi gravado.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        wavio.write(tmpfile.name, st.session_state.audio_data, SAMPLERATE, sampwidth=SAMPWIDTH)
        audio_file = tmpfile.name
        st.audio(audio_file)  # Reproduzir o áudio gravado

    # Transcrição do áudio
    model = whisper.load_model("small")
    result = model.transcribe(audio_file)
    transcribed_text = result["text"]

    st.session_state.conversation_history.append({"user": transcribed_text})
    st.success("🎧 Gravação concluída!")
    #st.write("📢 **Você disse:**", transcribed_text)

    if transcribed_text.strip():
        send_to_agent()
    else:
        st.error("Não foi possível transcrever o áudio. Tente novamente.")

    os.unlink(audio_file)
    
    
# Função para enviar conversa ao CrewAI
def send_to_agent():
    conversation_text = "\n".join(
        [f"Usuário: {msg['user']}" if 'user' in msg else f"Professor: {msg['bot']}" for msg in st.session_state.conversation_history]
    )

    teacher = Agent(
        role="English Teacher",
        goal="Ajude os alunos a aprender inglês corrigindo seus erros e sugerindo melhorias.",
        backstory="Sou um professor paciente e experiente em ensinar inglês para iniciantes.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    task = Task(
        description=f"Continue essa conversa:\n\n{conversation_text}\n\nResponda educadamente e de forma clara.",
        expected_output="Uma resposta coerente com o contexto, incluindo explicações gramaticais se necessário.",
        agent=teacher
    )

    crew = Crew(agents=[teacher], tasks=[task])
    with st.spinner("🧑‍🏫 Professor está pensando..."):
        response_obj = crew.kickoff()

    # ✅ Garantir que a resposta seja string pura
    response_text = str(response_obj).strip()  

    st.session_state.conversation_history.append({"bot": response_text})
    #st.write("🧑‍🏫 **Teacher:**", response_text)

    # Converter resposta do professor em áudio
    generate_audio(response_text)

# Função para executar áudio automaticamente com JavaScript
def autoplay_audio(file_path):
    audio_html = f"""
        <audio autoplay>
            <source src="{file_path}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


# Função para gerar e reproduzir áudio da resposta do professor
def generate_audio(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        tts = gTTS(text, lang="en")
        tts.save(tmpfile.name)
        audio_path = tmpfile.name

    # Exibir o player de áudio
    st.audio(audio_path, format="audio/mp3")

    # Injetar JavaScript para autoplay
    autoplay_audio(audio_path)


html_page_title = """
<div style="background-color:black;padding=60px">
        <p style='text-align:center;font-size:60px;font-weight:bold; color:red'>🎙️ Chatbot</p>
</div>
"""               
st.markdown(html_page_title, unsafe_allow_html=True)

html_page_title2 = """
<div style="background-color:black;padding=60px">
        <p style='text-align:center;font-size:60px;font-weight:bold; color:red'>Teacher English Speaker</p>
</div>
"""               
st.markdown(html_page_title2, unsafe_allow_html=True)

col11, col12, col13 = st.columns(3)
col21, col22, col23 = st.columns(3)

with col11:
    if st.button("🎤 Iniciar Gravação"):
        start_recording()

        
with col13:
    if st.button("⏹️ Encerrar Gravação"):
        stop_recording()
        
#with col22:
# Mostrar histórico da conversa
#    st.subheader("📝 Histórico da Conversa")
#    for msg in st.session_state.conversation_history:
#        if "user" in msg:
#            st.write(f"**🗣️ Você:** {msg['user']}")
#        else:
#            st.write(f"**🧑‍🏫 Teacher:** {msg['bot']}")
  