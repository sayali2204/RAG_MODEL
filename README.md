Purpose of this project is to build a rag model

FLOW OF THE PROJECT:
    This is the phase one i.e before a user enters his/her query:
        PDF / Docs
        ↓
        Load & split text
        ↓
        Embeddings (documents)
        ↓
        Vector DB (Pinecone stores them)

    After the user enters the query:
        User Query
        ↓
        Query Embedding (same embedding model)
        ↓
        Vector DB (similarity search)
        ↓
        Retriever (top-k relevant chunks)
        ↓
        LLM (Gemini)
        ↓
        Final Answer

Folder Structure:
        rag-dsa-assistant/
        │
        ├── data/
        │   └── dsa.pdf
        │
        ├── app/
        │   │
        │   ├── indexing/
        │   │   ├── __init__.py
        │   │   └── index_documents.py
        │   │
        │   ├── chat/
        │   │   ├── __init__.py
        │   │   ├── chat.py
        │   │   └── query_rewrite.py
        │   │
        │   ├── prompts/
        │   │   └── dsa_prompt.txt
        │   │
        │   ├── vectorstore/
        │   │   └── __init__.py
        │   │
        │   └── config/
        │       └── settings.py
        │
        ├── .env
        ├── requirements.txt
        ├── README.md
        └── run.py






step 1: Git initiation
        git init
        git status
        git add .
        git commit -m "first commit -> added the readme.md file"
        git branch -m main
        git remote add origin https://github.com/sayali2204/Langchain_Basics.git
        git push -u origin main

step 2: uv initialization
        uv init

step 3: Virtual env
        uv venv
        .venv\Scripts\activate

step 4: Installation of the required lib's
        uv add -r reuirements.txt

        langchain-community -> community mainted integrations, used to load the files and provides text splitters ,vector store connectors
        langchain -> this is the core langchain framework; we use this to build pipelines and Manages agents, memory, tools
        langchain-google-genai -> this is the langchain integration with the google gemini
        langchain-pinecone -> this is the langchain and pinecone connector which helps langchain talk to the vector db
        pinecone-client -> pinecone python SDk; it creates indexes , conects to pinecone cloud , manages vector DB. this is internally used by langchain-pinecone connector but it has to be externally installed.
        python-dotenv -> .env file stores all the keys and confidential info that we do not want to hardcode in the code.
        pypdf -> this is used to read the pdf and extract the metadata and pages from it.
        sentence-transformers -> used for creating vector embedding and semantic search.

step 5: created a .env file to store the secret info i.e the api credentials.

step 6: Created Data
        I have used deep learning pdf as the data source for the rag model and this pdf is stored in the data folder.

step 7: Created the index_docs.py 
        this file does the phase 1 of the RAG Model i.e 
        * loading the file
        * chunking the file
        * creating the embedding
        * storing the embeddings in the pinecone db

step 8: Creating chat.py file 
               