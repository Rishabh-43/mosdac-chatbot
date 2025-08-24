import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import google.generativeai as genai

# -------------------- Load API Key from Streamlit Secrets --------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found in Streamlit secrets")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# -------------------- Load and Merge Datasets --------------------
@st.cache_data
def load_all_data():
    dfs = []

    # 1. data.csv (already in title/content format)
    try:
        df1 = pd.read_csv("datasets/data.csv")
        if {"title", "content"}.issubset(df1.columns):
            dfs.append(df1[["title", "content"]])
        else:
            st.warning("data.csv has unexpected columns.")
    except Exception as e:
        st.warning(f"Could not load data.csv: {e}")

    # 2. all_missions.csv (mission_name, url, description)
    try:
        df2 = pd.read_csv("datasets/all_missions.csv")
        if {"mission_name", "url", "description"}.issubset(df2.columns):
            df2 = df2.rename(columns={"mission_name": "title", "description": "content"})
            df2["content"] = df2["content"] + " More: " + df2["url"]
            dfs.append(df2[["title", "content"]])
        else:
            st.warning("all_missions.csv has unexpected columns.")
    except Exception as e:
        st.warning(f"Could not load all_missions.csv: {e}")

    # 3. cleaned_missions.csv
    try:
        df3 = pd.read_csv("datasets/cleaned_missions.csv")
        if {"mission_name", "url", "description"}.issubset(df3.columns):
            df3["content"] = df3["description"].fillna("") + \
                             " Payloads: " + df3["payloads"].fillna("") + \
                             " Applications: " + df3["applications"].fillna("") + \
                             " Orbit type: " + df3["orbit_type"].fillna("") + \
                             " Status: " + df3["mission_status"].fillna("") + \
                             " More: " + df3["url"]
            df3 = df3.rename(columns={"mission_name": "title"})
            dfs.append(df3[["title", "content"]])
        else:
            st.warning("cleaned_missions.csv has unexpected columns.")
    except Exception as e:
        st.warning(f"Could not load cleaned_missions.csv: {e}")

    # 4. mosdac_docs.csv
    try:
        df4 = pd.read_csv("datasets/mosdac_docs.csv")
        if {"title", "url"}.issubset(df4.columns):
            df4["content"] = df4["title"] + " (Document) More: " + df4["url"]
            dfs.append(df4[["title", "content"]])
        else:
            st.warning("mosdac_docs.csv has unexpected columns.")
    except Exception as e:
        st.warning(f"Could not load mosdac_docs.csv: {e}")

    # 5. mosdac_product_data_with_sensors.csv
    try:
        df5 = pd.read_csv("datasets/mosdac_product_data_with_sensors.csv")
        if {"product_name", "product_url", "sensor_info"}.issubset(df5.columns):
            df5["content"] = df5["product_name"] + " uses sensor: " + df5["sensor_info"] + \
                             ". More: " + df5["product_url"]
            df5 = df5.rename(columns={"product_name": "title"})
            dfs.append(df5[["title", "content"]])
        else:
            st.warning("mosdac_product_data_with_sensors.csv has unexpected columns.")
    except Exception as e:
        st.warning(f"Could not load mosdac_product_data_with_sensors.csv: {e}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=["title", "content"])

df = load_all_data()

# -------------------- Create TF-IDF --------------------
@st.cache_resource
def create_vectorizer(data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['content'].tolist())
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = create_vectorizer(df)

# -------------------- Search --------------------
def retrieve_answer(query, top_k=2):
    query_vec = vectorizer.transform([query])
    scores = (tfidf_matrix * query_vec.T).toarray()
    ranked_indices = np.argsort(scores[:, 0])[::-1][:top_k]
    return "\n".join(df.iloc[idx]['content'] for idx in ranked_indices)

# -------------------- Ask Gemini --------------------
def ask_gemini(question, context):
    prompt = f"Answer based on the context below:\n\n{context}\n\nQuestion: {question}"
    model_g = genai.GenerativeModel("gemini-1.5-flash")
    return model_g.generate_content(prompt).text.strip()

# -------------------- UI --------------------
st.set_page_config(page_title="MOSDAC Query Chatbot", page_icon="🛰️")
st.title("🛰️ MOSDAC Query Chatbot")
st.write("Ask me about satellites or MOSDAC data access.")

user_q = st.text_input("Your question:")
if st.button("Ask") and user_q.strip():
    context = retrieve_answer(user_q)
    answer = ask_gemini(user_q, context)
    st.markdown(f"**Answer:** {answer}")
    st.caption("📚 Sources: Based on MOSDAC datasets.")