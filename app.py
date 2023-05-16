from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 📝")

    pdf = st.file_uploader("Uploade your PDF")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        print(pdf_reader)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        embeddings=OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks,embeddings)

        user_question=st.text_input("Ask anything related to the PDF:")

        if(user_question):
            docs = knowledge_base.similarity_search(user_question)
            llm=ChatOpenAI(model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm,chain_type="stuff")

            with get_openai_callback() as cb:
                response=chain.run(input_documents=docs,question=user_question)
                print(cb)
            st.write(response)

            



if __name__ == "__main__":
    main()