import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import nltk
import urllib3

# Suppress HTTPS warnings (development only; not recommended for production)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Streamlit app
st.set_page_config(page_title="Summarize Text From YouTube or Website")
st.title("Summarize Text From YouTube or Website")
st.subheader("Summarize URL")

# Sidebar input for Groq API key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# Prompt template for summarization
prompt_template = """
    Provide a summary of the following content in 300 words:
    Content: {text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from YT or website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the Groq API key and a valid URL.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (e.g., a YouTube video or website URL).")
    else:
        try:
            with st.spinner("Extracting and summarizing content..."):
                # Initialize LLM with the provided Groq API key
                llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

                # Load content based on the URL type
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        },
                    )

                # Extract data from the loader
                data = loader.load()

                # Ensure `data` is in the correct format for summarization
                if isinstance(data, list) and isinstance(data[0], str):
                    docs = data  # Directly use the string list if loader returns plain text
                else:
                    docs = [doc.page_content for doc in data if hasattr(doc, "page_content")]

                if not docs:
                    st.warning("No content could be extracted from the provided URL. Please check the URL or try a different one.")
                else:
                    # Summarization chain
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.invoke({"input_documents": docs})
                    st.success(output_summary)
        except Exception as e:
            st.error(f"Exception: {e}")
