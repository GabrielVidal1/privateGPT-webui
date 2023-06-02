#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
from langchain.chat_models import ChatOpenAI
import streamlit as st
from typing import Any
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()

embeddings_model_type = os.environ.get('EMBEDDINGS_MODEL_TYPE')
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs the response to streamlit."""

    def __init__(self) -> None:
        self.tokens_area = st.empty()
        self.tokens_stream = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.tokens_stream += token
        self.tokens_area.write(self.tokens_stream)

@st.cache_resource(show_spinner="Loading Private GPT...")
def get_qa_chain():
    # Parse the command line arguments
    match embeddings_model_type:
        case "HuggingFace":
            embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        case "OpenAI":
            organizationId = os.environ.get('ORGANIZATION_ID')
            apiKey = os.environ.get('OPENAI_API_KEY')
            if organizationId is None or apiKey is None:
                print("Please set the OPENAI_API_KEY and ORGANIZATION_ID environment variables.")
                exit
            embeddings = OpenAIEmbeddings(model=embeddings_model_name, openai_api_key=apiKey, openai_organization=organizationId)
        case _default:
            print(f"Model {embeddings_model_type} not supported!")
            exit
    
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    # Prepare the LLM
    match model_type:
        case "ChatGPT":
            organizationId = os.environ.get('ORGANIZATION_ID')
            apiKey = os.environ.get('OPENAI_API_KEY')
            if organizationId is None or apiKey is None:
                print("Please set the OPENAI_API_KEY and ORGANIZATION_ID environment variables.")
                exit
            llm = ChatOpenAI(temperature=0, openai_api_key=apiKey, openai_organization=organizationId, verbose=False, streaming=True)
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

# Initialization
if not "qa" in st.session_state:
    st.spinner("Initializing qa chain...")
    st.session_state["qa"] = get_qa_chain()
    
st.title("Private GPT")

# Interactive questions and answers
query = st.text_input("Enter a query: ")

if query :
    # Get the answer from the chain
    res = st.session_state["qa"](query, callbacks=[StreamlitCallbackHandler()])
    answer, docs = res['result'], res['source_documents']

    # display the relevant sources used for the answer
    for document in docs:
        with st.expander(f'Source : {document.metadata["title"]}') :
            st.markdown(document.page_content)

