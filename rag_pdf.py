from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

from dotenv import load_dotenv
import os

load_dotenv()

loader = PyPDFLoader("doc.pdf")
pages = loader.load_and_split()

db = FAISS.from_documents(pages, OpenAIEmbeddings())

retriever = db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
)

query = """
Quels les paroles du chant de la faculté polytechnique?
"""

docs = retriever.get_relevant_documents(query)

llm = ChatOpenAI(temperature=0, openai_api_key=os.environ['OPENAI_API_KEY'])
chain = load_qa_chain(llm, chain_type="stuff")

res = chain.run(input_documents=docs, question=query)

print(res)