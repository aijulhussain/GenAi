import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()


#langsmith traking
os.environ['LANGCHAIN_API_KEY']= os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"]= "true"
os.environ["LANGCHAIN_PROJECT"]= "Q&A Chatbot With OPENAI"


#prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful assistent. Please respose to the user queries"),
        ("user", "Question:{question}")
    ]
)


# engine= LLM model
#temperatue = 0 to 1, non creation to creative
#chain = how the interaction going to happend

def generate_response(question, api_key, engine, temperature, max_tokens):
    openai.api_key= api_key
    engine = ChatOpenAI(model = engine)
    output_parser= StrOutputParser()
    chain = prompt|engine|output_parser
    answer = chain.invoke({'question': question})
    return answer


#title of the aap
st.title("Enhanced Q&A Chatbot With OpenAI")

#sidebar for settings
st.sidebar.title("Settings")
api_key= st.sidebar.text_input("Enter your Open API Key:", type ="password")

#drop down to select various open AI models
engine= st.sidebar.selectbox("Select an OpenAI model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])

#adjust response parameter
#Max Tokens (): This simply sets a limit on the total number of tokens (words) the generative AI can generate in the response
temperature = st.sidebar.slider("Temperature", min_value = 0.0, max_value=1.0, value = 0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

#define main interface
st.write("Go ahead and ask any question")
user_input= st.text_input("You:")

if user_input and api_key:
    response=generate_response(user_input, api_key, engine, temperature, max_tokens)
    st.write(response)
elif user_input:
    st.waring("Please enter the OpenAI key in the side bar")

else:
    st.write("Please provide the user input")