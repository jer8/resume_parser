from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os

# Initialize FastAPI app
app = FastAPI(title="Resume Extractor API", description="An API to extract resume information using LLMs.", version="1.0.0")

# Function to extract resume info
def extract_resume_info(file):
    # Step 1: Read the PDF and extract text
    reader = PdfReader(file)
    formatted_document = []
    for page in reader.pages:
        formatted_document.append(page.extract_text())
    
    # Step 2: Split the document into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100  # Slight overlap for better context retention
    )
    docs = text_splitter.create_documents(formatted_document)
    
    # Step 3: Create embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_documents(docs, embeddings)
    
    # Step 4: Initialize the GroqChat model
    llm = ChatGroq(
        temperature=0.2,
        model="llama-3.1-70b-versatile",
        api_key="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"  # Ensure the API key is set as an environment variable
    )
    
    # Step 5: Create the retrieval chain
    retriever = store.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 matches for better context
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    # Step 6: Define extraction queries
    queries = {
        "relevant_title": "What is the main job title or role of the person?",
        "years_of_experience": "How many years of experience does this person have?",
        "techstack": "List the technologies, tools, and frameworks this person is proficient in.",
        "current_location": "Where is the current location of the person?",
        "certifications": "List all certifications the person has obtained.",
        "native_languages_known": "What languages does this person speak natively?",
        "computer_languages_known": "What programming or computer languages does this person know?"
    }
    
    # Step 7: Extract information
    resume_info = {}
    for key, query in queries.items():
        try:
            response = retrieval_chain.run(query)
            resume_info[key] = response
        except Exception as e:
            resume_info[key] = f"Error extracting data: {e}"
    
    return resume_info

# API endpoint
@app.post("/extract")
async def extract_resume(file: UploadFile = File(...)):
    """
    Upload a PDF resume, and extract information such as job title, experience, tech stack, and more.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        # Extract resume information
        resume_info = extract_resume_info(file.file)
        return JSONResponse(content=resume_info, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")

# Health check endpoint
@app.get("/")
def health_check():
    """
    Health check for the API.
    """
    return {"status": "ok", "message": "Resume Extractor API is running!"}
