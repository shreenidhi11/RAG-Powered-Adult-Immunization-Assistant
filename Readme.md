This is an Adult Immunization Schedule! (powered by FastAPI + RAG + Redis) where the dataset is downloaded from the WHO website  file.

The vector database used here is Chroma, LLM used is Gemini Flash 2.5 and embeddings from Hugging face

Note: For accessing the LLM use your own LLM key for Gemini Flash 2.5

Steps to run this project on your machine

    Run the requirements.txt file : pip -r requirements.txt
    Run the main.py file : uvicorn server:app --reload
    Run the streamlit UI: streamlit run app.py

Technologies Used:

    •	Programming Language: Python
    •	LLM/Embedding API: gemini-2.5-flash
    •	LangChain – for chaining embedding, vector store, and query operations
    •	Hugging Face Transformers – model used is sentence-transformers/all-MiniLM-L6-v2
    •	Streamlit - For User Interface
    •	Redis - For caching similar or same user queries

