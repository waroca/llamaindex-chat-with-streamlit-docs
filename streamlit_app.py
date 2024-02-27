import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.llms.openai import OpenAI
import openai
from llama_index.core import download_loader

st.set_page_config(page_title="Ask about Prezi", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Ask me about Prezi... 💬🦙")
st.info("This bot uses our Knowledge Base from Prezi's public Zendesk articles.", icon="📃")
st.info("This chat uses GPT3.5 without safeguards, this means: **It, Can, Hallucinate.** Verify the content before sending it to customers.", icon="⚠️")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help?"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        ZendeskReader = download_loader("ZendeskReader", custom_path="./")
        loader = ZendeskReader(zendesk_subdomain="prezi", locale="en-us")
        docs = loader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an support expert on Prezi and your job is to answer user questions. Assume that all questions are related to using Prezi. When given a question or statement in other languages, translate them to english before providing an answer in English. Keep your answers useful and based on facts – do not hallucinate features. When you don't know an answer just say 'I Don't Know'. Always add the original user content at the beginning of every response."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        PERSIST_DIR = "./zendesk"
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
