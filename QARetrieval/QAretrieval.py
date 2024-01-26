import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
import dill
import os

# Using Streamlit secrets to access environment variables
google_api_key = st.secrets["google_api_key"]

# Path to the combined FAISS index file
script_dir = os.path.dirname(os.path.abspath(__file__))
combined_faiss_index_path = os.path.join(script_dir, 'combined_faiss_index.pickle')

# Load the combined FAISS index and texts
with open(combined_faiss_index_path, 'rb') as f:
    combined_faiss_data = dill.load(f)
    combined_faiss_index = combined_faiss_data['combined_index']
    all_texts = combined_faiss_data['all_texts']

# Load the question-answering chain
chain = load_qa_chain(GooglePalm(), chain_type="stuff")

# Streamlit app
st.set_page_config("Question Answering App")

st.title("Question Answering App")

# User input for the question
question = st.text_input("Ask a question:")

# Button to trigger question answering
if st.button("Get Answer"):
    if question:
        # Run the question-answering chain on the combined documents and question
        docs = combined_faiss_index.similarity_search(question)
        answer = chain.run(input_documents=docs, question=question)
        st.success(answer)
    else:
        st.warning("Please enter a question.")

# Additional components or visualizations can be added as needed.
