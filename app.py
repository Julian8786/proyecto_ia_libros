import base64
from groq import Groq
from pathlib import Path
import os
import streamlit as st
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
#import ollama

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Configuración de página
st.set_page_config(page_title="Asistente de Libros", layout="wide")

# Cargar estilos personalizados si existen
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    st.markdown(
    """
    <style>
        /* Estilo para los mensajes del ASISTENTE (burbuja principal oscura) */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            background-color: #1e293b !important;
            border-radius: 12px;
            padding: 12px;
            border: 1px solid #334155;
        }

        /* Estilo para los mensajes del USUARIO (más limpios o con otro tono) */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            background-color: #0f172a !important;
            border-radius: 12px;
            padding: 12px;
            border: 1px solid #1e293b;
        }
        
        /* Si quieres ocultar el avatar rojo por defecto del usuario para que se vea más minimalista */
        [data-testid="stChatMessageAvatarUser"] {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --- INICIALIZAR ESTADO DE LA SESIÓN ---
if "chats" not in st.session_state:
    st.session_state.chats = {"Conversación 1": [
        {"role": "assistant", "content": "👋 ¡Hola! Soy tu asistente de mantenimiento. ¿En que puedo ayudarte hoy?"}
    ]}
if "active_chat" not in st.session_state:
    st.session_state.active_chat = "Conversación 1"
if "messages" not in st.session_state:
    st.session_state.messages = st.session_state.chats[st.session_state.active_chat]

if "mostrar_menu" not in st.session_state:
    st.session_state.mostrar_menu = False

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    if os.path.exists("logo.png.png"):
        st.image("logo.png.png", use_container_width=True)
    
    st.title("Asistente Virtual")

    # Botón de nueva conversación funcional
    if st.button("➕ Nueva conversación", use_container_width=True, type="primary"):
        new_chat_name = f"Conversación {len(st.session_state.chats) + 1}"
        st.session_state.chats[new_chat_name] = [
            {"role": "assistant", "content": "👋 ¡Hola! Soy tu asistente de mantenimiento. ¿En que puedo ayudarte hoy?"}
        ]
        st.session_state.active_chat = new_chat_name
        st.session_state.messages = st.session_state.chats[new_chat_name]
        st.rerun()
        
    st.markdown("---")
    st.markdown("### HISTORIAL DE CONVERSACIONES")
    
    for chat_name in list(st.session_state.chats.keys()):
        if st.button(f"🟢 {chat_name}", key=f"btn_{chat_name}", use_container_width=True):
            st.session_state.active_chat = chat_name
            st.session_state.messages = st.session_state.chats[chat_name]
            st.rerun()
    
    st.markdown("---")
    st.info("💡 Modo Móvil / PC Activo")

st.title("📚 Asistente de Libros")

# 1. Configurar la base de datos vectorial local con ChromaDB
@st.cache_resource
def inicializar_base_datos():
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="biblioteca_libros",
        embedding_function=embedding_fn
    )
    return collection

coleccion = inicializar_base_datos()

# 2. Función para procesar PDFs en bloques grandes para resúmenes profundos
def cargar_documentos_si_necesario():
    if coleccion.count() == 0:
        carpeta = "Documentos"
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
            return
        
        chunk_id = 0
        for archivo in os.listdir(carpeta):
            if archivo.endswith(".pdf"):
                ruta = os.path.join(carpeta, archivo)
                try:
                    lector = PdfReader(ruta)
                    texto_total = ""
                    for pagina in lector.pages:
                        texto = pagina.extract_text()
                        if texto:
                            texto_total += texto + "\n"
                    
                    if texto_total.strip():
                        # Bloques grandes de 3000 caracteres para no perder el hilo narrativo
                        tamano_chunk = 3000 
                        for i in range(0, len(texto_total), tamano_chunk):
                            fragmento = texto_total[i:i+tamano_chunk]
                            if len(fragmento.strip()) > 100:
                                coleccion.add(
                                    documents=[fragmento],
                                    metadatas=[{"source": archivo}],
                                    ids=[f"doc_{chunk_id}"]
                                )
                                chunk_id += 1
                except Exception as e:
                    print(f"Error procesando {archivo}: {e}")

cargar_documentos_si_necesario()

# 3. Interfaz del Chat (Mostrar mensajes actuales)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- ENTRADA DE TEXTO Y MENÚ FLOTANTE PEGADO ABAJO ---
if st.session_state.mostrar_menu:
    with st.container():
        st.markdown("### Opciones de entrada")
        if st.button("📸 Usar Cámara", use_container_width=True):
            st.session_state["modo_camara"] = True
            st.session_state.mostrar_menu = False
            st.rerun()
        if st.button("📎 Adjuntar archivos", use_container_width=True):
            st.session_state["modo_archivo"] = True
            st.session_state.mostrar_menu = False
            st.rerun()
        if st.button("⬅️ Volver", use_container_width=True, type="secondary"):
            st.session_state.mostrar_menu = False
            st.rerun()

# Pantallas de cámara o archivos si están activas...
if st.session_state.get("modo_camara", False):
    foto = st.camera_input("Toma una foto")
    if foto is not None:
        st.success("¡Foto capturada con éxito!")
    if st.button("Cerrar cámara"):
        st.session_state["modo_camara"] = False
        st.rerun()

if st.session_state.get("modo_archivo", False):
    archivo_subido = st.file_uploader("Sube tus documentos", type=["pdf", "png", "jpg", "jpeg"])
    if archivo_subido is not None:
        carpeta_docs = "Documentos"
        if not os.path.exists(carpeta_docs):
            os.makedirs(carpeta_docs)
        ruta_archivo = os.path.join(carpeta_docs, archivo_subido.name)
        with open(ruta_archivo, "wb") as f:
            f.write(archivo_subido.getbuffer())
        st.success(f"¡Archivo '{archivo_subido.name}' guardado correctamente!")
    if st.button("Cerrar adjuntos"):
        st.session_state["modo_archivo"] = False
        st.rerun()

# 1. Dibujamos el botón "+" dentro de la clase CSS flotante para que baje al fondo
st.markdown('<div class="floating-popover-container">', unsafe_allow_html=True)
if st.button("➕", key="btn_mas_flotante_abajo"):
    st.session_state.mostrar_menu = not st.session_state.mostrar_menu
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Capturar la entrada del usuario con el componente de chat de Streamlit
if prompt := st.chat_input("Escribe tu pregunta..."):
    # Mostrar el mensaje del usuario en la interfaz
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            resultados = coleccion.query(
                query_texts=[prompt],
                n_results=10 
            )
    
            fragmentos_recuperados = resultados['documents'][0] if resultados['documents'] else []
            contexto_encontrado = "\n---\n".join(fragmentos_recuperados)
            
            system_prompt = (
                "Eres un asistente literario estricto. Responde a la pregunta del usuario "
                "únicamente basándote en el siguiente contexto extraído de los libros. "
                "Si la respuesta no se encuentra en el contexto, di exactamente: "
                "'No encuentro información sobre eso en los textos disponibles.' "
                "No inventes nombres, datos ni eventos.\n\n"
                f"Contexto:\n{contexto_encontrado}"
            )
   # Llamar a la API de Groq únicamente cuando hay una pregunta activa
    try:
        respuesta_modelo = client.chat.completions.create(        model="qwen/qwen3.6-27b",
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.1           
                )
        
      
        
        # Extraer la respuesta de forma segura
        respuesta_final = respuesta_modelo.choices[0].message.content
        
        # Mostrar la respuesta del asistente
        with st.chat_message("assistant"):
            st.markdown(respuesta_final)
            
        st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
        
    except Exception as e:
        st.error(f"Ocurrió un error al conectar con la API: {e}")
# --- LOGO FLOTANTE EN LA ESQUINAS ---
if os.path.exists("logo.png.png"):
    with open("logo.png.png", "rb") as f:
        logo_bytes = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
            .logo-chat-derecha {{
                position: fixed !important;
                bottom: 20px !important;
                right: 25px !important;
                z-index: 999999 !important;
                pointer-events: none !important;
            }}
            .logo-chat-derecha img {{
                width: 35px !important;
                height: auto !important;
            }}
        </style>
        
        <div class="logo-chat-derecha">
            <img src="data:image/png;base64,{logo_bytes}" alt="Logo">
        </div>
        """,
        unsafe_allow_html=True
    )