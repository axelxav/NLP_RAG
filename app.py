import streamlit as st
from BilliardRAG.chatbot import RAGChatbot

@st.cache_resource
def load_chatbot():
    # Initialize the chatbot
    bl_RAG = RAGChatbot(doc_path="data/billiard_rules.pdf", db_path="vectordb")
    return bl_RAG

bl_RAG = load_chatbot()

st.title("BilliardRAG Chatbot")

# Initialize session state to store chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input for user
if prompt := st.chat_input("Ask me anything about billiard!"):
    # Add the user message to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response from RAGChatbot
    with st.chat_message("assistant"):
        response = bl_RAG.chat(prompt)  # Use the chatbot to generate a response
        st.markdown(response)  # Display the response in the chat

    # Add the assistant's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": response})