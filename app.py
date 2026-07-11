from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY


index_name = "medical-chatbot"

docsearch = None
retriever = None
rag_chain = None


chatModel = ChatCohere(
    model="command-r-plus-08-2024",
    temperature=0.4
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)


def load_chatbot():
    global docsearch, retriever, rag_chain

    if docsearch is None:

        print("Loading HuggingFace embeddings...")

        embeddings = download_hugging_face_embeddings()

        print("Connecting Pinecone...")

        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )


        retriever = docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )


        rag_chain = create_retrieval_chain(
            retriever,
            question_answer_chain
        )

        print("Chatbot loaded successfully")


    return rag_chain



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():

    msg = request.form["msg"]

    print("Question:", msg)


    chatbot = load_chatbot()


    response = chatbot.invoke(
        {
            "input": msg
        }
    )


    print("Response:", response["answer"])


    return str(response["answer"])




if __name__ == '__main__':

    app.run(
        host="0.0.0.0",
        port=8080
    )