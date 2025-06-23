import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio
from utils import clean_text

# Initialize Streamlit page configuration
st.set_page_config(layout="wide", page_title="Smart Cold Email Generator", page_icon="📧")

def create_streamlit_app(llm, portfolio, clean_text):
    st.title("📧 Smart Cold Email Generator")
    url_input = st.text_input("Just enter a URL and let the Smart Cold Email Generator help you craft the perfect email for your next opportunity! 🎯", value ="Enter Your URL!")
    submit_button = st.button("Submit")
    
    st.markdown(
        """
        ### Description
        The **Smart Cold Email Generator** is an AI-powered tool designed to help you craft professional, personalized cold emails. 
        Simply provide a URL (e.g., job postings, business opportunities, or relevant articles), and the app will:
        - Extract content from the webpage.
        - Match the skills in your portfolio with the extracted data.
        - Generate tailored cold emails for outreach.

        #### Steps to Use:
        1. Enter the URL of a webpage that contains relevant content.
        2. Click on "Submit" to process the content.
        3. View the generated cold emails and copy them to use in your outreach.

        ---
        """
    )

    if submit_button:
        try:
            # Load and clean data from URL
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)

            # Load portfolio and extract jobs
            portfolio.load_portfolio()
            jobs = llm.extract_jobs(data)

            # Generate emails for each job
            for job in jobs:
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                email = llm.write_mail(job, links)
                st.code(email, language='markdown')

        except Exception as e:
            st.error(f"An Error Occurred: {e}")

if __name__ == "__main__":
    # Initialize necessary components
    chain = Chain()
    portfolio = Portfolio()

    # Launch the Streamlit app
    create_streamlit_app(chain, portfolio, clean_text)
