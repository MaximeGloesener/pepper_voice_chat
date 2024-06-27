from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
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

llm = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'])

retriever = db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
)

def route(info):
    if "faculte" in info["topic"].lower():
        docs = retriever.get_relevant_documents(info["question"])
        chain = load_qa_chain(llm, chain_type="stuff")
        res = chain.run(input_documents=docs, question=info["question"])
        return res
    
    elif "mouvement" in info["topic"].lower():
        pass
    else:
        res = llm.invoke(info["question"])
        return res.content

chain = (
    PromptTemplate.from_template(
        """En fonction de la question de l'utilisateur: 
        - Si la question est liée à la faculté, l'université, les cours, la vie étudiante, les cercles, les professeurs, réponds avec le mot "faculte"
        - Si la question est à de la connaissance générale, réponds avec le mot "general"
        - Si l'utisateur te donne un demande de faire un mouvement ou une action, réponds avec le mot "mouvement"

Ne réponds que avec un mot.

<question>
{question}
</question>

Classification:"""
    )
    | llm
    | StrOutputParser()
)

while True:
    query = input("Entrez votre question: ")
    if query == "exit":
        break
    topic = chain.invoke({"question": query})
    prompt = {"topic": topic, "question": query}
    response = route(prompt)
    print(response)

