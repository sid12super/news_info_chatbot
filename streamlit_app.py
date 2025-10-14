import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import pandas as pd
import sys
import os
import tiktoken

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

# --- Helper Functions (No changes) ---

def setup_vector_db(df, force_rebuild=False):
    """Creates the Vector DB from a DataFrame if it's empty or a rebuild is forced."""
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, length_function=len)
        all_chunks, all_ids = [], []
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
                response = openai_client.embeddings.create(input=batch_chunks, model="text-embedding-3-small")
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
                response = client.chat.completions.create(model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": final_prompt}], max_completion_tokens=2048)
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

def get_ranked_news_map_reduce(df, llm_provider, model_name):
    """Analyzes all news articles in batches to overcome context length limits."""
    st.info("Large document analysis initiated. This may take a moment...")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    MAX_TOKENS_PER_BATCH = 10000
    all_articles_text = []
    for _, row in df.iterrows():
        doc_full_text = row['Document']
        parts = str(doc_full_text).split(' Description: ', 1)
        headline = parts[0]
        summary = parts[1] if len(parts) > 1 else "No summary provided."
        all_articles_text.append(f"--- ARTICLE START ---\nHeadline: {headline}\nSummary: {summary}\n--- ARTICLE END ---")
    batches, current_batch, current_batch_tokens = [], [], 0
    for article in all_articles_text:
        article_tokens = len(tokenizer.encode(article))
        if current_batch_tokens + article_tokens > MAX_TOKENS_PER_BATCH:
            batches.append("\n\n".join(current_batch))
            current_batch = [article]
            current_batch_tokens = article_tokens
        else:
            current_batch.append(article)
            current_batch_tokens += article_tokens
    if current_batch:
        batches.append("\n\n".join(current_batch))
    map_system_prompt = "You are a legal analyst. From the following list of news articles, identify the TOP 2 most legally and commercially significant for a global law firm. For each one you select, copy its full text (from '--- ARTICLE START ---' to '--- ARTICLE END ---') exactly as provided."
    top_articles_from_batches = []
    progress_bar = st.progress(0, text="Analyzing batches...")
    for i, batch in enumerate(batches):
        final_prompt = f"Please analyze these articles:\n\n{batch}"
        response = get_llm_response(llm_provider, model_name, final_prompt, map_system_prompt)
        top_articles_from_batches.append(response)
        progress_bar.progress((i + 1) / len(batches), text=f"Analyzed batch {i+1} of {len(batches)}")
    progress_bar.empty()
    st.info("All batches analyzed. Performing final ranking...")
    reduce_system_prompt = "You are a senior legal analyst. From the following pre-selected list of important articles, provide a final ranked analysis of the top 3 most important stories overall. Explain your reasoning for each, connecting it to specific legal practice areas like M&A, Litigation, or Regulatory Compliance."
    combined_top_articles = "\n\n".join(top_articles_from_batches)
    final_ranking_prompt = f"Here are the most important articles identified from several batches of news. Please perform a final analysis and ranking on them:\n\n{combined_top_articles}"
    final_response = get_llm_response(llm_provider, model_name, final_ranking_prompt, reduce_system_prompt)
    return final_response


def main():
    st.set_page_config(page_title="Legal News Analyst Bot", page_icon="⚖️")
    st.title("⚖️ Multi-LLM Legal News Analyst Bot")

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
        "OpenAI": "gpt-5-mini", "Google": "gemini-2.5-flash", "Anthropic": "claude-sonnet-4-5-20250929"
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
    st.write(f"Analyzing news from **`{NEWS_CSV_FILE}`**. Ask questions relevant to a global law firm.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # --- MODIFIED STARTUP LOGIC ---
    try:
        # Load the dataframe into session_state if it's not already there.
        if 'news_df' not in st.session_state:
            st.session_state.news_df = pd.read_csv(NEWS_CSV_FILE)
        
        # Check the persistent ChromaDB storage directly.
        if collection.count() == 0:
            # If the DB is empty on disk, build it once.
            with st.spinner("Performing first-time setup of the Vector DB..."):
                setup_vector_db(st.session_state.news_df, force_rebuild=True)
            st.rerun() # Rerun to show the success message in the sidebar
        
        # In all other cases (including refreshes), the DB is already populated, so we do nothing.
        
    except FileNotFoundError:
        st.error(f"Error: The file `{NEWS_CSV_FILE}` was not found in the same directory as the app.")
        st.stop()
        
    # --- Chat interface logic continues here ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask a question about the news..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        chat_history = st.session_state.messages[-11:-1]
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        
        if "most interesting news" in prompt.lower():
            response_content = get_ranked_news_map_reduce(st.session_state.news_df, selected_llm, selected_model)
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