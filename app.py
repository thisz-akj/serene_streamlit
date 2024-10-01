import os
import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document

# Initialize session state for storing chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Load environment variable for Google API Key
try:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCBlij-6wxrBIO9ThKgD7yKmImQFIq-gSQ"
    api_key = os.environ["GOOGLE_API_KEY"]
except Exception as e:
    st.error(f"An error occurred: {e}")

# System prompt for the mental health assistant
system_prompt = (
    '''You are a mental health virtual assistant, Alexa, designed to support students with their emotional and mental health concerns.
    You can understand and respond with empathy while offering advice and resources based on the queries.
    You should assist in helping students manage stress, anxiety, academic pressures, and any other emotional or mental health concerns.
    Use your knowledge and sense to understand emotions, maintain a comforting tone, and offer solutions based on the context but be empathetic and kind at all times.
    When needed, retrieve helpful resources or suggestions from the knowledge base for students to explore.
    '''
    "{context}"
)

# Create the prompt template with system message and user input
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Memory for chat history retention and context
memory = ConversationBufferMemory(
    memory_key="chat_history",  # Store conversation history
    return_messages=True        # Return previous messages as part of the context
)

# Initialize Google Generative AI (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, max_tokens=500)

# Create the question-answering chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Embedding and vectorstore initialization
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Example documents (replace with real documents)
docs = ["Sample document 1", "Sample document 2"]

# Convert docs to list of Document objects
documents = [Document(page_content=doc) for doc in docs]

# Create vectorstore from documents
vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)

# Create retriever from the vectorstore
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

# Function to fetch documents using the retriever
def fetch_documents(input_text):
    retrieved_docs = retriever.get_relevant_documents(input_text)

    # Convert each retrieved document (if dict) into a Document object
    documents = []
    for doc in retrieved_docs:
        # Ensure doc is of the right type
        if isinstance(doc, dict) and "page_content" in doc:
            documents.append(Document(page_content=doc["page_content"]))
        elif isinstance(doc, Document):
            documents.append(doc)
        else:
            st.error("Retrieved document is not in a recognized format.")

    return documents

# Process input to generate a response
def process_input(input_text):
    try:
        # Fetch relevant documents
        docs = fetch_documents(input_text)

        # Retrieve chat history
        chat_history = st.session_state["chat_history"]

        # Generate the response using the chain
        response = question_answer_chain.invoke({
            "input_documents": docs,
            "input": input_text,
            "context": chat_history
        })

        # Handle and return the response
        if isinstance(response, dict) and "output" in response:
            return response["output"]
        else:
            return str(response)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "I'm sorry, I couldn't process your request right now."

# Streamlit interface
st.title("Mental Health Assistant")

# Display chat history from session state
def display_chat_history():
    for message in st.session_state["chat_history"]:
        if message['role'] == "user":
            st.write(f"**You**: {message['content']}")
        else:
            st.write(f"**Alexa**: {message['content']}")

# Input box for user query
user_input = st.text_input("Ask me anything related to mental health:")

# Button to submit input
if st.button("Ask"):
    if user_input:
        # Append user input to the chat history
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        # Process the input and get a response
        response = process_input(user_input)

        # Append bot response to the chat history
        st.session_state["chat_history"].append({"role": "assistant", "content": response})

        # Display updated chat history
        display_chat_history()

    else:
        st.error("Please enter a question.")
else:
    # Display chat history if no new input
    display_chat_history()

