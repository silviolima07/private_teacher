import streamlit as st
import whisper
import tempfile
import os
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
from gtts import gTTS  # Biblioteca para converter texto em áudio
from st_audiorec import st_audiorec  # Biblioteca para gravação de áudio no Streamlit
from PIL import Image

#__import__('pysqlite3')
import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Carregar variáveis de ambiente
load_dotenv()

# Definir modelo da GROQ
llm = "groq/llama3-8b-8192"

# Estado global para armazenar conversa
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Função para processar áudio capturado
def process_audio_data(audio_data):
    # Criar arquivo temporário de áudio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_data)
        audio_file = tmpfile.name
        #st.audio(audio_file)  # Reproduzir o áudio gravado

    # Transcrição do áudio
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    transcribed_text = result["text"]

    st.session_state.conversation_history.append({"user": transcribed_text})
    st.success("🎧 Audio -> Texto: Transcricao  concluida!")

    if transcribed_text.strip():
        send_to_agent()
        os.unlink(audio_file)
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

    # Garantir que a resposta seja string pura
    response_text = str(response_obj).strip()  

    st.session_state.conversation_history.append({"bot": response_text})

    # Converter resposta do professor em áudio
    generate_audio(response_text)

# Função para gerar e reproduzir áudio da resposta do professor
def generate_audio(text):
    st.write("Tacher speeking...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        tts = gTTS(text, lang="en")
        tts.save(tmpfile.name)
        audio_path = tmpfile.name

    # Exibir o player de áudio
    st.audio(audio_path, format="audio/mp3")

# Interface Streamlit
#st.title("🎙️ Chatbot Teacher - Fale e Aprenda Inglês")

html_page_title = """
<div style="background-color:black;padding=60px">
        <p style='text-align:center;font-size:60px;font-weight:bold; color:red'>🎙️ Chatbot Teacher </p>
</div>
"""               
st.markdown(html_page_title, unsafe_allow_html=True)

col1,col2 = st.columns(2)

with col2:
    # Usar st_audiorec para capturar áudio
    audio_data = st_audiorec()
    
with col1:
    st.image(Image.open('img/aprender.png'))    

    

# Botão para processar o áudio gravado
if audio_data is not None:
    #if st.button("⏹️ Processar Áudio"):
    #    process_audio_data(audio_data)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_data)
        audio_file = tmpfile.name
        #st.audio(audio_file)  # Reproduzir o áudio gravado

    # Transcrição do áudio
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    transcribed_text = result["text"]

    st.session_state.conversation_history.append({"user": transcribed_text})
    st.success("🎧 Audio -> Texto: Transcricao  concluida!")

    if transcribed_text.strip():
        send_to_agent()
    else:
        st.error("Não foi possível transcrever o áudio. Tente novamente.")

    os.unlink(audio_file)
        
        
        
        

# Mostrar histórico da conversa
st.subheader("📝 Histórico da Conversa")
for msg in st.session_state.conversation_history:
    if "user" in msg:
        st.write(f"**🗣️ Você:** {msg['user']}")
    else:
        st.write(f"**🧑‍🏫 Teacher:** {msg['bot']}")