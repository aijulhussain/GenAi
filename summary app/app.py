import validators, streamlit as st
from langchain.prompts import PromptTemplate
# import nltk
# import urllib3

from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader



# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


#streamlit app
st.set_page_config(page_title="Summarize Text From YouTube or Website")
st.title("Summarize Text From YouTube or Website")
st.subheader("Summarize URL")

# llm = ChatGroq(model = "Gemma-7b-It", groq_api_key= groq_api_key)

## get the gorq api key and url to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type= "password" )

generic_url = st.text_input("URL", label_visibility="collapsed")

## gemma model using groq api
# llm = ChatGroq(model = "Gemma-7b-It", groq_api_key= groq_api_key)



prompt_template = """
    Provide a summary of the following content in 300 words:
    Content: {text}
"""

prompt = PromptTemplate(template= prompt_template, input_variables =["text"])

if st.button("Summarize the Content from YT or website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the Groq API key and a valid URL.")

    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can may be a YT video URL or Website url")
    else: 
        try:
            with st.spinner("Extracting and summarizing content..."):
                llm = ChatGroq(model = "mixtral-8x7b-32768", groq_api_key= groq_api_key)

                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url], ssl_verify= False, 
                                                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"},)
                
                docs = loader.load()

                # docs = [doc.page_content for doc in data]
                # docs = [doc.page_content for doc in data if doc.page_content.strip()]
                ## chain for summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt= prompt)
                output_summary = chain.run(docs)
                # output_summary = chain.invoke({"input_documents": docs})


                st.success(output_summary)
        except Exception as e:
            st.error(f"Exception:{e}")

