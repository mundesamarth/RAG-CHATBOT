import fitz
import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# Load PDF and create a vector database
def load_and_embed_documents():
    with fitz.open("annual_report.pdf") as doc:
        text = "\n".join([page.get_text("text") for page in doc])  # Extract text from each page

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # Split text
    texts = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Create embeddings
    vector_db = FAISS.from_texts(texts, embeddings)  # Store embeddings in FAISS
    return vector_db

# Initialize OpenAI model
llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=api_key,
    model="meta-llama/llama-3.1-70b-instruct:free",
)

# Streamlit UI setup
st.title("RAG CHATBOT")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input handling
user_input = st.chat_input("Ask a question:")

if user_input:
    if st.session_state.vector_db is None:
        with st.spinner("Loading document..."):
            st.session_state.vector_db = load_and_embed_documents()

    retriever = st.session_state.vector_db.as_retriever()  # Retrieve relevant documents

    # Define prompt for rephrasing follow-up questions
    question_generator_prompt = PromptTemplate(
        template="""
        Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question.

        Chat History:
        {chat_history}

        Follow-Up Input: {question}

        Standalone question:""",
        input_variables=["chat_history", "question"],
    )

    # Create LLM Chain for generating standalone questions
    question_generator_chain = LLMChain(
        llm=llm,
        prompt=question_generator_prompt
    )

    # Load the question-answering chain
    qa_chain = load_qa_chain(llm, chain_type="stuff")

    # Create the conversational retrieval chain
    qa = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator_chain,
        combine_docs_chain=qa_chain,
        memory=st.session_state.memory,
    )

    st.session_state.qa = qa  # Store the chatbot instance

    # Store and display user input
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.spinner("Thinking..."):
        response = st.session_state.qa.invoke({
            "question": user_input,
            "chat_history": st.session_state.memory.buffer
        })["answer"]

    # Store and display AI response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)