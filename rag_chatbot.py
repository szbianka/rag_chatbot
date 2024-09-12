### RAG Chatbot ###

# Import libraries 
import os
import re

from langchain import PromptTemplate, LLMChain, HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage 
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS 
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

##### UTILITY FUNCTIONS #####
def clean_text(text):
    """
    Cleans the input PDFs if there are formatting issues (eg. excessive spaces or non-ASCII characters).
    It also puts line breaks between sentences for better readability.
    
    Args:
        text (str): The input text to be cleaned.
    
    Returns:
        str: Cleaned and formatted text.
    """
    # Remove excessive newlines and replace multiple spaces between word by single space
    cleaned_text = re.sub(r'\n+', ' ', text.replace(u'\xa0', ' ')).strip()
    
    # Replace multiple spaces between words with a single space
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)

    # Remove hyphenated line breaks
    cleaned_text = re.sub(r'-\s+', '', cleaned_text)
    
    # Remove non-ASCII characters
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)  # Remove non-ASCII characters
    
    # Line breaks after sentences for better readability
    cleaned_text = re.sub(r'(\.|\?|!) ', r'\1\n', cleaned_text)
    
    return cleaned_text


def load_and_clean_pdfs(list_of_filepaths):
    """
    Loads PDFs and cleans their content.
    
    Args:
        List containing filepaths (str): Paths to the PDF files.
    
    Returns:
        list: A list of cleaned documents.
    """
    # Initialize a list to store the cleaned documents
    cleaned_documents = []

    for pdf in list_of_filepaths:
        try:
            # Load pdf document
            loader = PyPDFLoader(pdf)
            documents = loader.load()
            
            # Clean up the documents using clean_text() function
            for doc in documents:
                doc.page_content = clean_text(doc.page_content)
            
            # Add cleaned docs to the list
            cleaned_documents.extend(documents)
        # Error handling    
        except Exception as e:
            print(f"Error loading {pdf}: {e}")
    
    return cleaned_documents


def split_docs_into_chunks(cleaned_docs, chunk_size=1000, chunk_overlap=200):
    """
    Splits given pdfs into chunks (as input info to RAG is limited).
    
    Args:
        cleaned_docs (list): List of cleaned documents.
        chunk_size (int): Size of each chunk in characters.
        chunk_overlap (int): Number of overlapping characters between chunks.
    
    Returns:
        list: List of document chunks.
    """
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(cleaned_docs)
    print(f"Created {len(chunks)} chunks")
    return chunks

def create_vectorstore(chunks, k=3, fetch_k=5):
    """
    Create FAISS vectorstore for the document chunks.
    
    Args:
        chunks (list): List of document chunks.
        k (int): Number of top results to return.
        fetch_k (int): Number of over-fetched results.
    Returns:
        FAISS retriever for document search.
    """
    # Convert the chunks into embeddings
    embeddings = HuggingFaceEmbeddings()
    # Create database to store the embeddings
    db = FAISS.from_documents(chunks, embeddings)
    
    # Set up vectorstore as retriever
    retriever = db.as_retriever(
        search_type="similarity", # cosine similarity
        search_kwargs={"k": k}, # Retrieve top X (3) results
        fetch_k=5,  # Over-fetch X (5) results, rerank them to return top k
        )
    return retriever


def generate_answer(query, retriever, llm, memory_size=15):
    """
    Generates an answer to the given question (or query) using the retriever and a prompt template.
    
    Args:
        query (str): The input question or query.
        retriever: The predefined FAISS retriever.
        llm: The language model used to generate answers.
        memory_size (int): Maximum number of conversation turns to remember.

    Returns:
        str: The generated and cleaned answer.
    """
    # Predefined prompt template with system and human message templates
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessage(content="You are an intelligent assistant that provides clear, concise, and precise answers based on the provided context. If the answer is not in the context, respond with 'Sadly, I cannot provide a proper answer from the given documents'."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{question}")
        ]
    )
    
    # Use Langchain's ConversationalRetrievalChain to combine retriever and LLM to generate answers
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=ConversationBufferMemory(memory_key="chat_history", output_key="answer", max_memory_size=memory_size)
    )
    
    # Run the chain with the user query
    result = qa_chain.run({"question": query})
    
    # Parse the output using StrOutputParser to extract only the answer in the output
    parser = StrOutputParser()

    # Extracting part after "Helpful Answer:" in the output
    if "Helpful Answer:" in result:
        parsed_result = parser.parse(result.split("Helpful Answer:", 1)[1].strip())
    else:
        parsed_result = parser.parse(result)

    return parsed_result

##### MAIN FUNCTION #####

def rag_pipeline(pdf_files, query, model_name="microsoft/phi-1_5", temperature=0.2, max_new_tokens=200, repetition_penalty=1.5, top_p=0.9):
    """
    Processes the uploaded PDFs, creates the vectorstore, and answers the query by combining the previously defined functions. 
    Follows RAG steps.
    
    Args:
        pdf_files (list): List of paths to the PDF files.
        query (str): The input query to ask about the documents.
        model_name (str): Language model to use for generating answers.
        temperature (float): Sampling temperature for the language model (the lower the value, the more specific, the higher the more creative).
        max_new_tokens (int): Maximum number of new tokens to generate in the answer.
        repetition_penalty (float): Penalty for repeated phrases (above 1 is stricter).
        top_p (float): Nucleus sampling probability (controls how tokens are chosen during generation).
        
    Returns:
        str: The generated and cleaned answer.
    """
    print(f"Initializing RAG pipeline...")

    # 1. Load and clean the PDFs
    cleaned_docs = load_and_clean_pdfs(pdf_files)
    print(f"Loaded and cleaned {len(cleaned_docs)} documents...")
    
    # 2. Split documents into chunks
    chunks = split_docs_into_chunks(cleaned_docs)
    print(f"Created {len(chunks)} chunks...")
    
    # 3. Create the FAISS vectorstore from the chunks
    retriever = create_vectorstore(chunks)
    print(f"Created FAISS vectorstore...")

    # Set up the LM for answer generating (with personal API token from HuggingFace)
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    llm = HuggingFaceHub(
        huggingfacehub_api_token=huggingfacehub_api_token,
        repo_id= model_name,
        verbose=False,
        model_kwargs={
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "top_p": top_p
        }
    )

    print(f"Generating an answer to the query...")

    # 4. Generate an answer to the query
    return generate_answer(query, retriever, llm)
    print(f"Answer generated successfully!")


