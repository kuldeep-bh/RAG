1. Introduction

DeepSeek-R1 is a multi-modal Question Answering (QA) system designed to extract meaningful answers from PDF documents using Retrieval-Augmented Generation (RAG). Unlike simple text-based QA systems, DeepSeek-R1 combines:

Text and audio processing

Vector embeddings

Language models (LLMs)

RAG pipelines

This enables users to ask questions interactively and receive accurate, context-aware answers grounded in document data.

Key Objectives of DeepSeek-R1:

Efficient document ingestion and processing

Semantic search across large knowledge bases

Contextual, AI-powered responses with source attribution

Scalable architecture using Chroma as a vector database

2. Understanding RAG (Retrieval-Augmented Generation)
2.1 What is RAG?

RAG is an AI method that combines retrieval and generative AI:

Traditional LLMs: Rely only on pre-trained knowledge, which is limited by model memory.

RAG systems: Retrieve relevant external information before generating answers.

Benefits:

Reduces hallucinations

Handles large knowledge bases

Generates accurate, contextually relevant answers

Simplified Explanation:

“RAG is when an AI first looks up relevant information and then uses that to answer questions.”

2.2 Why RAG is Used

RAG is essential for:

Handling Large Knowledge

LLMs have token limits; RAG accesses information outside the model’s memory.

Increasing Accuracy

Retrieved real-world data helps answers stay grounded in facts.

Reducing Hallucinations

Generative models sometimes produce fabricated facts; retrieval ensures answers are data-backed.

Domain-Specific Q&A

Useful for PDFs, research papers, enterprise databases, or highly specialized content.

2.3 Components of RAG

RAG typically involves three main components:

1. Retriever

Purpose: Finds relevant information from a knowledge base.

How it works:

Converts a query into a vector (embedding)

Searches a vector database (e.g., Chroma, FAISS, Pinecone) for similar vectors

Returns top relevant document chunks

Example:
Query: “What is RAG in AI?”
Retriever fetches PDF chunks or documents explaining RAG.

2. Knowledge Base / Document Store

Purpose: Stores all retrievable information.

Can include PDFs, text files, web pages, or structured data

Uses embedding-based vector databases for fast similarity search

Example Tools: ChromaDB, FAISS, Pinecone

3. Generator

Purpose: Generates coherent responses from retrieved information.

Uses retrieved documents as context

Produces natural language answers via LLMs like GPT, LLaMA, or T5

Ensures responses are informative, contextually relevant, and grounded in real data

3. Step-by-Step Workflow of RAG
Step 1: Input Question

User asks a question, e.g., “What is Python?”

Step 2: Retrieval Step

Retriever searches the vector database for relevant document chunks.

Database stores embeddings of all documents or PDF chunks

Finds semantically similar chunks to the query

Step 3: Augmentation Step

Retrieved chunks are combined with the query

Form a context window for the LLM

Step 4: Generation Step

LLM uses both context and pre-trained knowledge to generate answers

Can include source attribution for transparency

Simplified Diagram:

[User Query] --> [Retriever] --> [Relevant Docs] --> [LLM] --> [Answer]

4. Chroma Vector Database
4.1 What is Chroma?

Stores text as vectors, representing semantic meaning

Allows semantic search (finding similar content by meaning, not exact words)

Used in DeepSeek-R1 to store PDF chunks for fast retrieval

Analogy:

Chroma is like a smart library that finds books matching the meaning of your question, not just the title.

4.2 Why Chroma?

Efficient for large document sets

Integrates seamlessly with LangChain

Enables fast semantic search for QA workflows

5. Setting Up the Environment
5.1 Python Packages
pip install chromadb==0.5.5 langchain-chroma==0.1.2 langchain==0.2.11 \
langchain-community==0.2.10 langchain-text-splitters==0.2.2 langchain-groq==0.1.6 \
transformers==4.43.2 sentence-transformers==3.0.1 unstructured==0.15.0 unstructured[pdf]==0.15.0

5.2 System Utilities
apt-get install poppler-utils


poppler-utils helps extract text from PDFs on Linux

6. Core Libraries and Roles
Library	Role
langchain	Orchestrates LLM workflows, chains, retrievers
langchain-community	Provides embeddings & vector store integrations
langchain-groq	Access to Groq LLMs
chromadb	Vector database for storing embeddings
unstructured	Extract text from PDFs & other documents
sentence-transformers	Generate embeddings for text
transformers	Supports Hugging Face models
poppler-utils	Needed to read PDFs on Linux
7. Code Walkthrough
7.1 Loading PDFs
from langchain.document_loaders import UnstructuredFileLoader
loader = UnstructuredFileLoader(r"C:\Users\ASUS\Downloads\Python_small.pdf")
documents = loader.load()


Reads PDFs and converts content to Document objects

7.2 Splitting Text
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)


Splits text for LLM token limits

chunk_overlap preserves context continuity

7.3 Creating Embeddings and Vector Store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings()
vector_db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="doc_db")
retriever = vector_db.as_retriever()


Converts text chunks to vectors

Stores in Chroma for semantic search

7.4 LLM Setup (Groq)
from langchain_groq import ChatGroq
import os
os.environ['GROQ_API_KEY'] = "gsk....................Ckg"

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


Connects to Groq LLM

temperature=0 ensures deterministic, factual answers

7.5 RetrievalQA Pipeline
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


Combines retriever + LLM for Q&A

return_source_documents=True keeps track of source chunks

7.6 Querying the System
query = "What is Python?"
result = qa_chain.invoke({"query": query})
print(result["result"])


Retriever finds relevant chunks

LLM generates context-aware answers

Returns answer + source documents

8. Overall Workflow

Load PDF → UnstructuredFileLoader

Split Text → CharacterTextSplitter

Embed Chunks → HuggingFaceEmbeddings

Store in Vector DB → Chroma

Retrieve Chunks → as_retriever()

Generate Answer → ChatGroq via RetrievalQA

9. Example Queries

"What is Python?" → Returns definition and explanation

"What are Python functions?" → Returns explanation from PDF content

10. Advantages of DeepSeek-R1

Handles large PDF documents efficiently

Supports domain-specific Q&A

Answers are grounded in retrieved content, reducing hallucinations

Scalable using Chroma vector DB and LangChain pipelines

11. Summary

DeepSeek-R1 is a complete PDF-based QA system using RAG, vector embeddings, and LLMs. It provides:

Interactive question-answering

Grounded and accurate answers

Easy extension for enterprise and research use

This system is ideal for developers, researchers, and data scientists exploring document-driven AI applications.
