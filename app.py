import streamlit as st
import whisper
import tempfile
import os
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
from gtts import gTTS  # Biblioteca para converter texto em áudio
from st_audiorec import st_audiorec  # Biblioteca para gravação de áudio no Streamlit
from PIL import Image
from io import BytesIO
import base64

#import pysqlite3 as sqlite3
#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



# Carregar variáveis de ambiente
load_dotenv()

# Definir modelo da GROQ
llm = "groq/llama3-8b-8192"

# Estado global para armazenar conversa
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Função para converter dicionários em tuplas (hashable)
def to_hashable(item):
    if isinstance(item, dict):
        return tuple((key, to_hashable(value)) for key, value in item.items())
    elif isinstance(item, list):
        return tuple(to_hashable(x) for x in item)
    else:
        return item
        
def show_historico(msg):
    # Mostrar histórico da conversa
    st.subheader("📝 Histórico da Conversa")
    for msg in st.session_state.conversation_history[::-1]:
        if "user" in msg:
            # st.write(f"**🗣️ Você:** {msg['user']}")
            st.markdown("#### 🗣️ Você:")
            st.markdown(
                         f"<div style='font-size: 20px;'>{msg['user']}</div>",
                          unsafe_allow_html=True
                        )
        else:
            #st.write(f"**🧑‍🏫 Teacher:** {msg['bot']}")
            st.markdown('#### 🧑‍🏫 Teacher:')
            #st.markdown('#### '+ f'{msg['bot']}')
            # Aumentar o tamanho da fonte usando HTML
            st.markdown(
                         f"<div style='font-size: 20px;'>{msg['bot']}</div>",
                          unsafe_allow_html=True
                        )    

# Função para processar áudio capturado
def process_audio_data(audio_data):
    # Criar arquivo temporário de áudio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_data)
        audio_file = tmpfile.name
        #st.audio(audio_file)  # Reproduzir o áudio gravado

    # Transcrição do áudio
    with st.spinner(' 🎧 Audio Transformando audio em texto...'): # Transcricao
        model = whisper.load_model("small")
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
def send_to_agent(lang, agente, acao1, acao2 ):
    # Remover duplicatas mantendo a ordem
    itens_vistos = set()
    lista_sem_repeticao = []
    for item in st.session_state.conversation_history:
        hashable_item = to_hashable(item)
        if hashable_item not in itens_vistos:
            lista_sem_repeticao.append(item)
            itens_vistos.add(hashable_item)
            
    conversation_text = "\n".join(
        [f"Usuário: {msg['user']}" if 'user' in msg else f"Professor: {msg['bot']}" for msg in lista_sem_repeticao]
    )

    teacher = Agent(
        role="Teacher",
        goal="Ajude os alunos a aprender {idioma} corrigindo seus erros e sugerindo melhorias.",
        backstory="Sou um professor paciente e experiente em ensinar {idioma} para iniciantes.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    task = Task(
        description=f"Responda no idioma {idioma}.Continue essa conversa:\n\n{conversation_text}\n\nResponda educadamente e de forma clara e sucinta.",
        expected_output="Uma resposta coerente com o contexto, incluindo explicações gramaticais se necessário.",
        agent=teacher
    )

    crew = Crew(agents=[teacher], tasks=[task])
    inputs = {'idioma': idioma}
    with st.spinner(f"🧑‍🏫 {agente} {acao1}"): # Pensando na resposta
        response_obj = crew.kickoff(inputs=inputs)

    # Garantir que a resposta seja string pura
    response_text = str(response_obj).strip()  

    st.session_state.conversation_history.append({"bot": response_text})

    # Converter resposta do professor em áudio
    generate_audio(lang, response_text, agente, acao2)

# Função para gerar e reproduzir áudio da resposta do professor
def generate_audio(lang, text, agente, acao):
    st.markdown(f"### {agente} {acao}") # Respondendo ao aluno
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        #st.write("Text to Speech")
        tts = gTTS(text, lang=lang)
        tts.save(tmpfile.name)
        audio_path = tmpfile.name

    # Exibir o player de áudio
    st.write("Play audio")
    st.audio(audio_path, format="audio/wav")

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
    
    
def center_img(img):
    image_path="image.png"
    img.save(image_path)
    # Getting the base64 string
    base64_image = encode_image(image_path)
    # Usando HTML para centralizar a imagem
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{base64_image}" alt="Imagem" style="width: 40%; height: auto;">
        </div>
        """,
        unsafe_allow_html=True
    )

def center_text(texto):
    st.markdown(
        f"""
        <div style="padding:60px; text-align:center; 
                    font-size:30px; font-weight:bold; color:red;">
            {texto}
        </div>
        """,
        unsafe_allow_html=True
    )    
html_page_title = """
    <div style="background-color:black;padding=60px">
        <p style='text-align:center;font-size:60px;font-weight:bold; color:red'>🎙️ Chatbot Teacher </p>
    </div>
    """               

st.markdown(html_page_title, unsafe_allow_html=True)

st.sidebar.image(Image.open('img/microfones.png'))

# Dicionário de idiomas
idiomas = {
    "Português": {'lang': 'pt',"start": "Iniciar Gravação", "stop": "Parar Gravação", 'reset':'Reiniciar', 'download': 'Baixar', 'agente':'Professor','acao1':'pensando...', 'acao2': 'respondendo'},
    "English": {'lang':'en',"start": "Start", "stop": "Stop",'reset': 'Reset', 'download': 'Download', 'agente':'Teacher', 'acao1':'thinking...', 'acao2': 'speeking'},
    "Español": {'lang':'es', "start": "Comenzar Grabación", "stop": "Detener Grabación", 'reset': 'Reiniciar', 'download': 'Bajar', 'agente':'Maestro', 'acao1': 'piensando...', 'acao2':'hablando' },
}

# Seleção de idioma
idioma = st.selectbox("Escolha o idioma", list(idiomas.keys()))
start= idiomas[idioma]["start"]
stop = idiomas[idioma]["stop"]
reset = idiomas[idioma]["reset"]
download = idiomas[idioma]["download"]
agente = idiomas[idioma]['agente']
acao1 = idiomas[idioma]['acao1']
acao2 = idiomas[idioma]['acao2']
lang = idiomas[idioma]['lang']



image_br = Image.open('img/br.png')
image_es = Image.open('img/es.png')
image_us = Image.open('img/us.png')
 
col11, col12 = st.columns(2)
 
with col11:
    if idioma == 'Português':
        center_img(image_br)     
    elif idioma == 'Español':
        center_img(image_es)
    else:
        center_img(image_us)  
        
    st.write('')        
    st.markdown(f' #### START Recording: ' f'<span style="color: green; font-size:30px">{start}</span>',unsafe_allow_html=True)

    st.markdown(f' #### STOP: ' f'<span style="color: red; font-size:30px">{stop}</span>',unsafe_allow_html=True)
    
    st.markdown(f"#### Reset: " f'<span style="color: blue; font-size:30px">{reset}</span>',unsafe_allow_html=True)
    
    st.markdown(f"#### Download: " f'<span style="color: yellow; font-size:30px">{download}</span>',unsafe_allow_html=True)    

with col12:
    center_text('🎧 Recorder')
    audio_data = st_audiorec()
    
       

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
    if transcribed_text != '':

        st.session_state.conversation_history.append({"user": transcribed_text})
        st.success("🎧 Audio -> Texto: Transcricao  concluida!")
        

        if transcribed_text.strip():
            send_to_agent(lang, agente, acao1, acao2)
    else:
        st.error("Não foi possível transcrever o áudio. Tente novamente.")

    os.unlink(audio_file)
        
        
        
# Verificar se há mensagens já enviadas        
num_msg = len(st.session_state.conversation_history[::-1])



if num_msg > 0:
    if st.sidebar.button('Show Histórico'):
        msg = st.session_state.conversation_history[::-1]
        show_historico(msg)


            
st.sidebar.subheader(" Apenas 3 passos")
st.sidebar.markdown("#### 1) Inicie o bate-papo com Start Recording")
st.sidebar.markdown("#### 2) Ao final clique Stop")
st.sidebar.markdown('#### 3) Um agente como professor de linguas irá falar contigo.')    