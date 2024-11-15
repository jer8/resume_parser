import streamlit as st
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def extract_resume_info(file):
    # Format PDF file content
    reader = PdfReader(file)
    formatted_document = []
    for page in reader.pages:
        formatted_document.append(page.extract_text())
    
    # Split the content into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    docs = text_splitter.create_documents(formatted_document)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings()
    
    # Load to FAISS vector database
    store = FAISS.from_documents(docs, embeddings)
    
    # Create retrieval chain with targeted queries
    llm = ChatGroq(
        temperature=0.2,
        model="llama-3.1-70b-versatile",
        api_key=st.secrets["GROQ_API_KEY"]
    )
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=store.as_retriever()
    )
    
    # Define queries for specific information extraction
    queries = {
        "work_experience": "Extract the work experience of the candidate from the document.",
        "tech_stack": "List the technical skills or tech stack the candidate is proficient in.",
        "certifications": "Identify any certifications the candidate has completed.",
        "industry_domain": "Specify the industry domain in which the candidate has experience.",
        "location": "Identify the candidate's location.",
        "current_employment": "Is the candidate currently employed? If so, specify the organization."
    }
    
    # Run each query and collect responses
    extracted_info = {}
    for field, query in queries.items():
        response = retrieval_chain.invoke(query)
        extracted_info[field] = response.get('result', 'Not found')
    
    return extracted_info


# Streamlit app setup
st.set_page_config(page_title="Resume Info Extractor")
st.markdown("<h1 style='text-align: center; font-size:5rem;'>Resume Info Extractor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:1rem; font-weight: 200;'>Extract structured information from resume PDFs</p><br><br>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a resume (.pdf format) to extract information",
    type="pdf"
)

if uploaded_file:
    with st.spinner("Extracting information from the resume..."):
        extracted_info = extract_resume_info(uploaded_file)
        
    # Display extracted information
    st.subheader("Extracted Information")
    st.write("**Work Experience:**", extracted_info.get("work_experience", "Not available"))
    st.write("**Tech Stack:**", extracted_info.get("tech_stack", "Not available"))
    st.write("**Certifications:**", extracted_info.get("certifications", "Not available"))
    st.write("**Industry Domain:**", extracted_info.get("industry_domain", "Not available"))
    st.write("**Location:**", extracted_info.get("location", "Not available"))
    st.write("**Current Employment Status:**", extracted_info.get("current_employment", "Not available"))
    
st.markdown("<a style='text-align: center; font-size:0.7rem; font-weight: 200;' href='https://www.namanverma.in/'>By: Naman Verma</a><br><br>", unsafe_allow_html=True)
