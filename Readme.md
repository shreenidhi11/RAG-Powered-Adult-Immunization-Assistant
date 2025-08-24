This is an Adult Immunization Schedule! (powered by FastAPI + RAG + Redis Vector Search) where the dataset is downloaded from the WHO website  file.

The vector database used here is Chroma, LLM used is Gemini Flash 2.5 and embeddings from Hugging face

Note: For accessing the LLM use your own LLM key for Gemini Flash 2.5

Steps to run this project on your machine
    
    Run redis docker container : docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
    Run prometheus docker container: docker run -d --name=prometheus -p 9090:9090 -v "<main_code_location>/prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus
    Run grafana docker container : docker run -d --name=grafana -p 3000:3000 grafana/grafana
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

