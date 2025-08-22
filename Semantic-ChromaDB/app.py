import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="products")
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("E-commerce Semantic Search Demo")
st.write("Upload your CSV file with at least `name` and `description` columns.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "name" not in df.columns or "description" not in df.columns:
        st.error("CSV must contain 'name' and 'description' columns!")
    else:
        st.success(f"Uploaded {len(df)} products successfully!")
        for idx, row in df.iterrows():
            emb = model.encode(row['description']).tolist()
            collection.add(
                ids=[str(row.get("id", idx))],
                metadatas=[{"name": row['name'], "description": row['description']}],
                embeddings=[emb]
            )
        st.success("Product embeddings stored in VectorDB successfully!")
        query = st.text_input("Search for products:")
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=3)
        if query:
            query_emb = model.encode(query).tolist()
            results = collection.query(
                query_embeddings=[query_emb],
                n_results=top_k
            )
            st.write("### Top Results:")
            for i in range(len(results['ids'][0])):
                st.write(
                    f"**{results['metadatas'][0][i]['name']}** - "
                    f"{results['metadatas'][0][i]['description']}"
                )
