import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import pandas as pd
import sys
import os

# --- Fix for working with ChromaDB and Streamlit ---
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Global Constants & Paths ---
CHROMA_DB_PATH = "./ChromaDB_News"
CHROMA_COLLECTION_NAME = "LegalNewsCollection"
NEWS_CSV_FILE = "Example_news_info_for_testing.csv" 

# Initialize ChromaDB client.
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

# --- Helper Functions ---

def setup_vector_db(df, force_rebuild=False):
    """
    Creates the Vector DB from a DataFrame if it's empty or a rebuild is forced.
    It processes CSV rows, chunks them, and embeds each chunk using OpenAI.
    """
    global collection

    if collection.count() > 0 and not force_rebuild:
        st.sidebar.info(f"Vector DB already contains {collection.count()} document chunks.")
        return

    st.sidebar.warning("Building Vector DB from the CSV. Please wait...")
    with st.spinner("Processing rows, chunking, and creating embeddings..."):
        if 'Document' not in df.columns:
            st.error("CSV must contain a 'Document' column.")
            st.stop()

        openai_client = st.session_state.openai_client
        
        if force_rebuild:
            try:
                chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
            except Exception as e:
                print(f"Info: Could not delete collection (it might not have existed): {e}")
            collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150, length_function=len
        )
        
        all_chunks = []
        all_ids = []

        for index, row in df.iterrows():
            doc_full_text = row['Document']
            parts = str(doc_full_text).split(' Description: ', 1)
            
            headline = parts[0]
            summary = parts[1] if len(parts) > 1 else "No summary provided."

            doc_text = f"Headline: {headline}\n\nSummary: {summary}"

            chunks = text_splitter.split_text(doc_text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"row_{index}_chunk_{i+1}"
                all_chunks.append(chunk)
                all_ids.append(chunk_id)

        batch_size = 100 
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i+batch_size]
            batch_ids = all_ids[i:i+batch_size]
            
            try:
                response = openai_client.embeddings.create(
                    input=batch_chunks, model="text-embedding-3-small"
                )
                embeddings = [item.embedding for item in response.data]
                collection.add(documents=batch_chunks, ids=batch_ids, embeddings=embeddings)
            except Exception as e:
                st.error(f"Failed to embed and add a batch of chunks: {e}")
    
    st.sidebar.success(f"Vector DB built successfully with {collection.count()} document chunks.", icon="✅")

def query_vector_db(prompt, n_results=4):
    """Queries the vector database to find relevant document chunks for the user's prompt."""
    try:
        openai_client = st.session_state.openai_client
        query_response = openai_client.embeddings.create(input=[prompt], model="text-embedding-3-small")
        query_embedding = query_response.data[0].embedding
        
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
        context = "\n---\n".join(results['documents'][0]) if results.get('documents') else "No relevant context found."
        return context
    except Exception as e:
        st.error(f"Error querying Vector DB: {e}")
        return "Error retrieving context from the database."

def get_llm_response(llm_provider, model_name, final_prompt, system_prompt):
    """Calls the selected LLM with the prompt, context, and chat history."""
    try:
        with st.spinner(f"Asking {model_name}..."):
            if llm_provider == "OpenAI":
                client = st.session_state.openai_client
                response = client.chat.completions.create(model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": final_prompt}], max_tokens=2048)
                return response.choices[0].message.content
            elif llm_provider == "Google":
                client = st.session_state.gemini_client
                model = genai.GenerativeModel(model_name)
                full_prompt_for_gemini = system_prompt + "\n" + final_prompt
                response = model.generate_content(full_prompt_for_gemini)
                return response.text
            elif llm_provider == "Anthropic":
                client = st.session_state.anthropic_client
                response = client.messages.create(model=model_name, system=system_prompt, messages=[{"role": "user", "content": final_prompt}], max_tokens=2048)
                return response.content[0].text
    except Exception as e:
        st.error(f"An error occurred with {llm_provider}: {e}")
        return f"Sorry, I encountered an error while contacting {llm_provider}."

def main():
    st.set_page_config(page_title="Legal News Analyst Bot", page_icon="⚖️")
    st.title("⚖️ Multi-LLM Legal News Analyst Bot")
    st.write(f"Analyzing news from **`{NEWS_CSV_FILE}`**. Ask questions relevant to a global law firm.")

    # --- Sidebar Setup ---
    st.sidebar.header("Settings")
    try:
        if 'openai_client' not in st.session_state:
            st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        if 'gemini_client' not in st.session_state:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            st.session_state.gemini_client = genai 
        if 'anthropic_client' not in st.session_state:
            st.session_state.anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    except Exception as e:
        st.error(f"Failed to initialize LLM clients. Check your API keys in secrets.toml. Error: {e}")
        st.stop()

    selected_llm = st.sidebar.selectbox("Choose an LLM Provider:", ("OpenAI", "Google", "Anthropic"))

    model_mapping = {
        "OpenAI": "gpt-4o-mini",
        "Google": "gemini-1.5-flash-latest",
        "Anthropic": "claude-3-haiku-20240307"
    }
    selected_model = model_mapping[selected_llm]
    st.sidebar.markdown(f"**Selected Model:** `{selected_model}`")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Management")
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if st.sidebar.button("Re-Build Vector DB"):
        if 'news_df' in st.session_state and st.session_state.news_df is not None:
            setup_vector_db(st.session_state.news_df, force_rebuild=True)
            st.rerun()
        else:
            st.sidebar.warning(f"Could not find {NEWS_CSV_FILE} to rebuild the database.")

    # --- Main Chat Logic ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    data_loaded = False
    if 'db_initialized' not in st.session_state:
        try:
            st.session_state.news_df = pd.read_csv(NEWS_CSV_FILE)
            setup_vector_db(st.session_state.news_df, force_rebuild=True)
            st.session_state.db_initialized = True
            data_loaded = True
        except FileNotFoundError:
            st.error(f"Error: The file `{NEWS_CSV_FILE}` was not found in the same directory as the app.")
            st.stop()
    else:
        data_loaded = True
        
    if data_loaded:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about the news..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            chat_history = st.session_state.messages[-11:-1]
            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
            
            response_content = ""
            if "most interesting news" in prompt.lower():
                system_prompt = "You are a senior legal analyst at a top global law firm. Your task is to identify and rank the most commercially and legally significant news stories from the provided list for our firm's practice areas (e.g., M&A, Litigation, Regulatory, IP)."
                
                # --- MODIFIED SECTION START ---
                # This logic now correctly parses the 'Document' column instead of looking for 'Headline' and 'Summary'.
                all_news_items = []
                for _, row in st.session_state.news_df.iterrows():
                    doc_full_text = row['Document']
                    parts = str(doc_full_text).split(' Description: ', 1)
                    headline = parts[0]
                    summary = parts[1] if len(parts) > 1 else "No summary provided."
                    all_news_items.append(f"Headline: {headline}\nSummary: {summary}")
                
                all_news_content = "\n\n".join(all_news_items)
                # --- MODIFIED SECTION END ---

                final_prompt = f"Review all the following news articles and provide a ranked analysis of the top 3 most important stories, explaining your reasoning for each.\n\nARTICLES:\n{all_news_content}"
                response_content = get_llm_response(selected_llm, selected_model, final_prompt, system_prompt)
            else:
                system_prompt = "You are a precise legal news analyst. Answer the user's question based *only* on the provided context from the news documents and the conversation history. If the answer isn't in the provided materials, state that clearly."
                context = query_vector_db(prompt)
                final_prompt = f"CONTEXT FROM DOCUMENTS:\n{context}\n\nCONVERSATION HISTORY:\n{history_str}\n\nUSER'S QUESTION:\n{prompt}"
                response_content = get_llm_response(selected_llm, selected_model, final_prompt, system_prompt)

            full_response = f"**Answer from `{selected_model}`:**\n\n{response_content}"
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            with st.chat_message("assistant"):
                st.markdown(full_response)

if __name__ == "__main__":
    main()