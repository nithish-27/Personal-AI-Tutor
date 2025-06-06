import os
import tempfile
from typing import List, Dict, Tuple

import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Configuration
st.set_page_config(page_title="Personal AI Tutor", page_icon="📚", layout="wide")

# Constants
ALLOWED_FILE_TYPES = ["pdf", "txt"]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TEMPERATURE = 0.3
MAX_FILE_SIZE_MB = 50  # Hugging Face Spaces limit

# Initialize session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_text" not in st.session_state:
    st.session_state.document_text = ""

def init_gemini(api_key: str):
    """Initialize the Gemini API with the provided API key."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using file path."""
    try:
        pdf_reader = PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle None returns
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_txt(file_content) -> str:
    """Extract text from a text file."""
    try:
        return file_content.decode("utf-8")
    except Exception as e:
        st.error(f"Error reading text file: {str(e)}")
        return ""

def process_uploaded_files(uploaded_files) -> str:
    """Process all uploaded files and extract text with proper error handling."""
    full_text = ""
    for uploaded_file in uploaded_files:
        try:
            # Check file size
            if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"File {uploaded_file.name} exceeds {MAX_FILE_SIZE_MB}MB limit")
                continue

            file_extension = uploaded_file.name.split(".")[-1].lower()
            
            if file_extension == "pdf":
                # Create temporary file for PDF processing
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name
                
                text = extract_text_from_pdf(temp_file_path)
                os.unlink(temp_file_path)  # Clean up temp file
                
            elif file_extension == "txt":
                text = extract_text_from_txt(uploaded_file.getvalue())
            else:
                continue
            
            if text.strip():
                full_text += f"\n\n{text}"
            else:
                st.warning(f"No text extracted from {uploaded_file.name}")
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    return full_text.strip()

def split_text_into_chunks(text: str) -> List[str]:
    """Split text into chunks for embedding."""
    if not text.strip():
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    return text_splitter.split_text(text)

def create_vector_store(text_chunks: List[str], api_key: str):
    """Create a FAISS vector store from text chunks."""
    if not text_chunks:
        raise ValueError("No text chunks provided for vector store")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    return FAISS.from_texts(text_chunks, embedding=embeddings)

def get_semantic_search_results(query: str, k: int = 3) -> List[Dict]:
    """Perform semantic search and return relevant chunks."""
    if st.session_state.vector_store is None:
        return []
    
    try:
        docs = st.session_state.vector_store.similarity_search(query, k=k)
        return docs
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def generate_answer_with_context(query: str, context: str, api_key: str) -> str:
    """Generate an answer using Gemini with the provided context."""
    if not context.strip():
        return "I couldn't find any relevant context to answer this question."
    
    prompt_template = """
    You are a helpful AI tutor. Use the following context to answer the student's question.
    Context: {context}
    
    Question: {question}
    
    Answer in a clear and concise manner. If the answer isn't in the context, say "I don't know" rather than making something up.
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=TEMPERATURE
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )
        
        response = chain.run({"query": query})
        return response
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Sorry, I encountered an error while generating an answer."

def generate_study_suggestions(document_text: str, chat_history: List[Tuple[str, str]], api_key: str):
    """Generate study suggestions using Gemini."""
    if not document_text.strip():
        return "No document content available to generate suggestions."
    
    prompt = f"""
    You are an AI tutor helping a student learn from their study materials. 
    Based on the following document content and previous questions asked, 
    suggest what the student should study next.
    
    Document Content:
    {document_text[:5000]}
    
    Previous Questions:
    {chat_history[-5:]}
    
    Provide 3-5 specific suggestions for what to study next, focusing on:
    1. Important topics not covered in previous questions
    2. Potential knowledge gaps based on the questions asked
    3. Foundational concepts that would help understand the material better
    
    Format your response as bullet points.
    """
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating suggestions: {str(e)}")
        return "Sorry, I couldn't generate study suggestions at this time."

# Streamlit UI
st.title("📚 Personal AI Tutor")
st.markdown("Upload your study materials and ask questions to your AI tutor!")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    # Try to get API key from secrets first, then from input
    # In your sidebar section, replace this:
    # api_key = st.secrets.get("GOOGLE_API_KEY", "") if hasattr(st, "secrets") else ""
    # if not api_key:
    #     api_key = st.text_input("Enter your Google Gemini API Key", type="password")

    # With just this:
    api_key = st.text_input("Enter your Google Gemini API Key", type="password")
    st.markdown("[Get a Gemini API key](https://ai.google.dev/)")
    st.markdown("---")
    st.header("Upload Study Materials")
    uploaded_files = st.file_uploader(
        f"Upload PDF or TXT files (max {MAX_FILE_SIZE_MB}MB each)",
        type=ALLOWED_FILE_TYPES,
        accept_multiple_files=True
    )
    
    if uploaded_files and not st.session_state.processed_docs:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    # Validate files first
                    valid_files = []
                    for file in uploaded_files:
                        if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                            st.error(f"Skipping {file.name} - exceeds size limit")
                            continue
                        valid_files.append(file)
                    
                    if not valid_files:
                        st.error("No valid files to process")
                        st.stop()  # Changed from return to st.stop()
                    
                    # Process files
                    document_text = process_uploaded_files(valid_files)
                    
                    if not document_text.strip():
                        st.error("No text could be extracted from the documents")
                        st.stop()  # Changed from return to st.stop()
                    
                    st.session_state.document_text = document_text
                    text_chunks = split_text_into_chunks(document_text)
                    
                    if not text_chunks:
                        st.error("Failed to split document into chunks")
                        st.stop()  # Changed from return to st.stop()
                    
                    if api_key:
                        st.session_state.vector_store = create_vector_store(text_chunks, api_key)
                        st.session_state.processed_docs = True
                        st.session_state.uploaded_files = valid_files
                        st.success(f"Processed {len(valid_files)} document(s) successfully!")
                    else:
                        st.error("Please enter your Gemini API key first")
                        st.stop()  # Changed from return to st.stop()
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
                    st.stop()  # Optional, depending on whether you want to stop on error

# Main content area
if st.session_state.processed_docs:
    # Question input
    st.subheader("Ask a Question")
    question = st.text_input("Type your question here", key="question_input")
    
    if question and api_key:
        if st.button("Get Answer"):
            with st.spinner("Searching for answers..."):
                try:
                    # Get relevant context
                    relevant_docs = get_semantic_search_results(question)
                    context = "\n".join([doc.page_content for doc in relevant_docs])
                    
                    # Generate answer
                    answer = generate_answer_with_context(question, context, api_key)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, answer, context))
                    
                    # Display answer
                    st.subheader("Answer")
                    st.markdown(answer)
                    
                    # Display context
                    with st.expander("Reference Context"):
                        st.write(context)
                except Exception as e:
                    st.error(f"Error answering question: {str(e)}")
    
    # Study suggestions
    st.markdown("---")
    st.subheader("Study Suggestions")
    if st.button("Generate Study Suggestions") and api_key:
        with st.spinner("Analyzing your study materials..."):
            suggestions = generate_study_suggestions(
                st.session_state.document_text,
                [(q, a) for q, a, _ in st.session_state.chat_history],
                api_key
            )
            st.markdown(suggestions)
    
    # Chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, (question, answer, context) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q: {question}"):
                st.markdown(f"**A:** {answer}")
                st.markdown("---")
                st.markdown("**Reference Context:**")
                st.write(context)
else:
    st.info("Please upload and process your study materials to begin.")