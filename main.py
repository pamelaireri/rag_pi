import os
import tempfile
from pathlib import Path


# Vector store and embedding imports
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, Pinecone
from langchain_community.vectorstores import Pinecone as pvs  # This is Langchain's vectorstore wrapper
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from pinecone import Pinecone, ServerlessSpec


# Document processing imports
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter  # This should stay in langchain core

# LLM and chain imports
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI  # This replaces OpenAIChat

# UI imports
import streamlit as st

# Set up our directory structure
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

# Create directories if they don't exist
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Set up the Streamlit page
st.set_page_config(page_title="RAG System")
st.title("📚 Document Q&A System")

def load_documents():
    """
    Loads PDF documents from the temporary directory.
    
    Returns:
        documents: List of loaded document objects
    """
    try:
        # TODO: Add validation to check if directory is empty
        if not any(TMP_DIR.iterdir()):
            st.error("No documents found in the temporary directory.")
            return []
        loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
        documents = loader.load()

        if not documents:
            st.error("Documents could not be loaded. Check file formats or permissions")
        else:
            st.success(f"Loaded {len(documents)} documents scucessfully")
        return documents
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return []

def split_documents(documents):
    """
    Splits documents into chunks for processing.
    
    Args:
        documents: List of loaded documents
    Returns:
        texts: List of document chunks
    """
    # TODO: Experiment with different chunk sizes and overlap values
    if not documents:
        st.error("No documents found to split.")
        return []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    if not texts:
        st.error("Splitting failed. No text chunks created.")
    else:
        st.success(f"Created {len(texts)} text chunks.")
       

    return texts

def embeddings_on_local_vectordb(texts):
    """
    Creates and manages a local vector store using Chroma.
    
    Args:
        texts: List of document chunks
    Returns:
        retriever: Document retriever object
    """
    try:
        # TODO: Add progress indicator for embedding creation

        if not texts:
            st.error("No text chunks were created. Check document splitting.")
            return None
        
         # Debug: Display the first chunk
        vectordb = Chroma.from_documents(
            texts, 
            embedding=OpenAIEmbeddings(),
            persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix()
        )
        
        vectordb.persist()
        retriever = vectordb.as_retriever(search_kwargs={'k': 7})

        if retriever is None:
            st.error("Error: Retriever could not be initialized from Chroma.")
        else:
            st.success("Retriever successfully created!")

        return retriever
    except Exception as e:
        st.error(f"Error creating local vector store: {str(e)}")
        return None

def embeddings_on_pinecone(texts):
    """
    Creates and manages a Pinecone vector store.
    
    Args:
        texts: List of document chunks
    Returns:
        retriever: Document retriever object
    """
    try:
        if not texts:
            st.error("No text chunks received for Pinecone vectorization.")
            return None
            
        pc = Pinecone(api_key=st.session_state.pinecone_api_key)
        index_name = st.session_state.pinecone_index
       
        #First check if index exists and delete if necessary
        #if index_name in pc.list_indexes().names():
            #st.warning(f"Deleting existing index {index_name} to recreate with correct dimensions")
            #pc.delete_index(index_name)
            
            # Wait for deletion
            #import time
            #time.sleep(5)
        
         # Create new index with correct dimensions and spec for free tier
        #st.info("Creating new index with 1536 dimensions...")
        spec = ServerlessSpec(
            cloud="aws",  # Changed from aws to gcp
            region="us-east-1"  # Changed to free tier region
        )
        
        # Create new index with correct dimensions
        st.info("Creating new index with correct dimensions...")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=spec
        )
        
        # Wait for index initialization
        time.sleep(5)
        
        # Verify dimensions
        index_info = pc.describe_index(index_name)
        #st.write(f"Index dimensions: {index_info.dimension}")
        if index_info.dimension != 1536:
            st.error(f"Index dimension mismatch. Expected 1536 but got {index_info.dimension}")
            return None
            
        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
        
        #st.write("Creating Pinecone vector store...")
        
        vectordb = PineconeVectorStore.from_documents(
            documents=texts,
            embedding=embeddings,
            index_name=index_name
        )

        retriever = vectordb.as_retriever(search_kwargs={'k': 2})

        if retriever is None:
            st.error("Error: Pinecone retriever is None.")
            return None
        else:
            st.success("Pinecone retriever successfully created!")
            return retriever
            
    except Exception as e:
        st.error(f"Error creating Pinecone vector store: {str(e)}")
        # Add more detailed error information
        #st.write("Debug - Error details:", str(e))
        if hasattr(e, 'response'):
            st.write("Response content:", e.response.content)
        return None
    
def query_llm(retriever, query):
    """
    Processes queries using the retrieval chain.
    
    Args:
        retriever: Document retriever object
        query: User question string
    Returns:
        result: Generated answer
    """
    #st.write("Debug in query_llm:")
    #st.write(f"1. Received retriever type: {type(retriever)}")
    #st.write(f"2. Retriever is None: {retriever is None}")
    try:
        # TODO: Add custom prompting for better answers
        
        if retriever is None:
            st.error("Error: The retriever is None. Ensure documents are processed correctly.")
            return "Error: The retriever is not initialized."
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(openai_api_key=st.session_state.openai_api_key),
            retriever=retriever,
            return_source_documents=True,
        )
        
        # Process the query
        result = qa_chain({
            'question': query, 
            'chat_history': st.session_state.messages
        })
        
        # Update conversation history
        # TODO: Add source citations to the response
        st.session_state.messages.append((query, result['answer']))
        
        return result['answer']
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return "I encountered an error processing your question. Please try again."

def setup_interface():
    """
    Sets up the Streamlit interface components.
    """
    with st.sidebar:
        # API keys and configuration
        if "openai_api_key" in st.secrets:
            st.session_state.openai_api_key = st.secrets.openai_api_key
        else:
            st.session_state.openai_api_key = st.text_input(
                "OpenAI API Key", 
                type="password",
                help="Enter your OpenAI API key"
            )
        
        # TODO: Add validation for API keys
        if "pinecone_api_key" in st.secrets:
            st.session_state.pinecone_api_key = st.secrets.pinecone_api_key
        else:
            st.session_state.pinecone_api_key = st.text_input(
                "Pinecone API Key",
                type="password",
                help="Enter your Pinecone API key"
            )
        
        if "pinecone_env" in st.secrets:
            st.session_state.pinecone_env = st.secrets.pinecone_env
        else:
            st.session_state.pinecone_env = st.text_input(
                "Pinecone Environment",
                help="Enter your Pinecone environment"
            )
        
        if "pinecone_index" in st.secrets:
            st.session_state.pinecone_index = st.secrets.pinecone_index
        else:
            st.session_state.pinecone_index = st.text_input(
                "Pinecone Index Name",
                help="Enter your Pinecone index name"
            )
    
    # Vector store selection
    st.session_state.pinecone_db = st.toggle(
        'Use Pinecone Vector DB',
        help="Toggle between local and cloud vector storage"
    )
    
    # File upload
    # TODO: Add file size validation
    st.session_state.source_docs = st.file_uploader(
        label="Upload PDF Documents",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF documents"
    )

def process_documents():
    """
    Processes uploaded documents and creates vector store.
    """
    # Validate required fields
    if not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API key.")
        return
    
    if st.session_state.pinecone_db and (
        not st.session_state.pinecone_api_key or
        not st.session_state.pinecone_env or
        not st.session_state.pinecone_index
    ):
        st.warning("Please provide all Pinecone credentials.")
        return
    
    if not st.session_state.source_docs:
        st.warning("Please upload at least one document.")
        return
    
    try:
        with st.spinner("Processing documents..."):
            # Save uploaded files to temporary directory
            for source_doc in st.session_state.source_docs:
                with tempfile.NamedTemporaryFile(
                    delete=False, 
                    dir=TMP_DIR.as_posix(),
                    suffix='.pdf'
                ) as tmp_file:
                    tmp_file.write(source_doc.read())
            
            # Load and process documents
            documents = load_documents()
            if not documents:
                st.error("Document loading failed.")
                return
            
            # Clean up temporary files
            for file in TMP_DIR.iterdir():
                TMP_DIR.joinpath(file).unlink()
            
            # Split documents into chunks
            texts = split_documents(documents)
            if not texts:
                st.error("Text splitting failed. No text chunks generated.")
            else:
                st.success(f"Generated {len(texts)} text chunks.")

            
            # Create vector store
            if not st.session_state.pinecone_db:
                retriever = embeddings_on_local_vectordb(texts)
            else:
                retriever = embeddings_on_pinecone(texts)

         #   if retriever is None:
          #      st.error("Retriever initialization failed. Debug embeddings function.")
           #     return
                #st.session_state.retriever = embeddings_on_local_vectordb(texts)
            #else:
               # st.session_state.retriever = embeddings_on_pinecone(texts)
            
            st.success("Documents processed successfully!")

            #store retriever in session state
            st.session_state.retriever = retriever
            #st.write("Debug in process_documents:")
            #st.write(f"1. Retriever type: {type(retriever)}")
            #st.write(f"2. Retriever in session state: {st.session_state.retriever is not None}")
            #st.write(f"3. Session state keys: {st.session_state.keys()}")
            #st.success("Documents processed successfully")

            #st.success("Documents processed successfully")
            #st.write("✅ Retriever stored in session state.")

            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def main():
    """
    Main application loop.
    """
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Set up the interface
    setup_interface()
    
    # Process documents button
    st.button("Process Documents", on_click=process_documents)
    
    # Display chat history
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('assistant').write(message[1])
    
    # Chat input
    if query := st.chat_input():
        if "retriever" not in st.session_state:
            st.warning("Please process documents first.")
            return
        
        #st.write("Debug in main before query_llm:")
        #st.write(f"1. Retriever in session state: {st.session_state.retriever is not None}")
        #st.write(f"2. Session state keys: {st.session_state.keys()}")
            
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("assistant").write(response)

if __name__ == '__main__':
    main()
