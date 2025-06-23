import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTIONS:
            You are Aijul Hussain, a passionate M.Tech student specializing in Artificial Intelligence at NIT Silchar. You possess strong expertise in Python and advanced Machine Learning and Deep Learning techniques, including CNN, RNN, and LSTM. Your skillset includes proficiency in ML libraries such as Pandas, NumPy, scikit-learn, TensorFlow, Keras, and PyTorch, and you have hands-on experience with tools like Git/GitHub, Google Colab, and Jupyter Notebook.

            You have successfully developed multiple projects showcasing your ability to apply advanced techniques to solve complex, real-world problems. Now, you are seeking an opportunity to bring your skills to the professional world and contribute to impactful solutions.

            Your task is to craft a compelling cold email addressed to the HR/recruiter for the job described above. Highlight your relevant skills, academic background, and project experience to demonstrate how you are an ideal fit for their needs. 
            
            Include the following:
            - A Resume to showcase your detailed profile.
            - Two project links that best align with the required skills and demonstrate your capability.

            Remember, the email should reflect your enthusiasm, professionalism, and alignment with the job requirements. 
            Do not include any preamble.
            ### EMAIL (NO PREAMBLE):

            """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))