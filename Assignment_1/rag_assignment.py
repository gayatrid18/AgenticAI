import pandas as pd
import json

import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.callbacks import get_openai_callback


load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

#Open AI 
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore=FAISS.load_local("vectorstore_openai", embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
prompt = ChatPromptTemplate.from_template(""""
Answer the question using only the context below:
                                          {context}
Question: {question}
""")

# query = st.text_input("Ask your question")
# if query:
#     docs = retriever.invoke(query)
#     context = "\n\n".join([doc.page_content for doc in docs])

#     chain = prompt | llm | StrOutputParser()
    
#     response = chain.invoke({
#         "context" : context,
#         "question": query
#     })

#     st.write (response)


#Gemini
embeddings_gemini = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore_gemini = FAISS.load_local("vectorstore_gemini", embeddings_gemini, allow_dangerous_deserialization=True)
retriever_gemini = vectorstore_gemini.as_retriever(search_kwargs={"k":3})
llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

prompt_gemini = ChatPromptTemplate.from_template(""""
Answer the question using only the context below:
                                          {context}
Question: {question}
""")

# query = st.text_input("Ask your question for gemini")
# if query:
#     docs = retriever_gemini.invoke(query)
#     context = "\n\n".join([doc.page_content for doc in docs])

#     chain = prompt_gemini | llm_gemini | StrOutputParser()
    
#     response = chain.invoke({
#         "context" : context,
#         "question": query
#     })

#     st.write (response)

#Ollama
embeddings_ollama = OllamaEmbeddings(model="embeddinggemma")
vectorstore_ollama= FAISS.load_local("vectorstore_ollama", embeddings_ollama, allow_dangerous_deserialization=True)
retriever_ollama = vectorstore_ollama.as_retriever(search_kwargs={"k":3})
llm_ollama = ChatOllama(model="gemma3:1b", temperature=0.7)

prompt_ollama = ChatPromptTemplate.from_template(
    """
Answer the question using only the context below:
{context}
Question:{question}
"""
)

# query = st.text_input("Ask your question for Ollama")
# if query:
#   docs = retriever_ollama.invoke(query)
#   context="\n\n".join([doc.page_content for doc in docs])
#   chain = prompt_ollama | llm_ollama | StrOutputParser()

#   response = chain.invoke({
#      "context":context,
#      "question": query
#   })

#   st.write(response)

option = st.sidebar.radio(
    "Choose Model",
    ["Open AI", "Gemini", "Ollama", "Show Comparision"]
)

if option == "Open AI":
    query = st.text_input("Ask your question for Open AI")
    if query:
        start_time = time.time()
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        with get_openai_callback() as cb:
            response = llm.invoke(
                prompt.format(context=context, question=query)
            )

            output = StrOutputParser().invoke(response.content)
            
            end_time = time.time()
            st.write(output)
            response_time= end_time - start_time
            st.write(f"Response Time: {response_time:.2f} seconds")
            st.write(f"Total Token: {cb.total_tokens}")


elif option == "Gemini":
    query = st.text_input("Ask your question for gemini")
    if query:
        start_time = time.time()
        docs = retriever_gemini.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        response = llm_gemini.invoke(
            prompt_gemini.format(context=context, question=query)
        )

        prompt_tokens = response.usage_metadata["input_tokens"]
        completion_tokens = response.usage_metadata["output_tokens"]

        total_tokens = prompt_tokens + completion_tokens

        output = StrOutputParser().invoke(response)
        
        end_time = time.time()
        st.write(output)
        response_time= end_time - start_time
        st.write(f"Response Time: {response_time:.2f} seconds")
        st.write(f"Total Token: {total_tokens}")


elif option == "Ollama":
    query = st.text_input("Ask your question for Ollama")
    if query:
        start_time = time.time()
        docs = retriever_ollama.invoke(query)
        context="\n\n".join([doc.page_content for doc in docs])

        response = llm_ollama.invoke(
            prompt_ollama.format(context=context, question=query)
        )

        prompt_tokens = response.response_metadata["prompt_eval_count"]
        completion_tokens = response.response_metadata["eval_count"]

        total_tokens = prompt_tokens + completion_tokens

        output = StrOutputParser().invoke(response)
        
        end_time = time.time()
        st.write(output)
        response_time= end_time - start_time
        st.write(f"Response Time: {response_time:.2f} seconds")
        st.write(f"Total Token: {total_tokens}")

elif option == "Show Comparision":
    with open("comparision.json") as f:
        data = json.load(f)

    rows = []

    # Flatten JSON
    for model_data in data:
        model_name = model_data["model"]
        ease = model_data.get("easeOfSetup", "")

        for metric in model_data["metrics"]:
            row = {
                "model": model_name,
                "question": metric["question"],
                "accuracy": metric["accuracy"],
                "hallucination": metric["hallucination"],
                "responseSpeed": metric["responseSpeed"],
                "tokenPerQuery": metric["tokenPerQuery"],
                "easeOfSetup": ease
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Display table
    st.dataframe(df, use_container_width=True)