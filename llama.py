import json

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

json_schema = {
    "title": "Person",
    "description": "Identifying information about a person.",
    "type": "object",
    "properties": {
        "name": {"title": "Name", "description": "The person's name", "type": "string"},
        "age": {"title": "Age", "description": "The person's age", "type": "integer"},
        "gender": {
            "title": "Gender",
            "description": "Is the person a male",
            "type": "string",
            "enum": ["male", "female"],
        },
        "fav_food": {
            "title": "Fav Food",
            "description": "The person's favorite food",
            "type": "string",
        },
    },
    "required": ["name", "age", "gender"],
}

llm = ChatOllama(model="llama3", format="json")

messages = [
    HumanMessage(
        content="Please tell me about a person using the following JSON schema:"
    ),
    HumanMessage(content="{dumps}"),
    HumanMessage(
        content="Now, considering the schema, tell me about a person named John is a boy who is 35 years old and loves pizza."
    ),
]

prompt = ChatPromptTemplate.from_messages(messages)
dumps = json.dumps(json_schema, indent=2)

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"dumps": dumps}))