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


PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY



index_name = "medical-chatbot"


docsearch = None
rag_chain = None



chatModel = ChatCohere(
    model="command-r-plus-08-2024",
    temperature=0.4
)



prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)



question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)




def load_chatbot():

    global docsearch, rag_chain


    if rag_chain is None:

        print("Loading embeddings...")

        embeddings = download_hugging_face_embeddings()


        print("Connecting Pinecone...")


        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )


        retriever = docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3
            }
        )


        rag_chain = create_retrieval_chain(
            retriever,
            question_answer_chain
        )


        print("Chatbot Ready")


    return rag_chain





@app.route("/")
def index():

    return render_template("chat.html")






@app.route("/get", methods=["POST"])
def chat():

    try:

        print("CHAT REQUEST RECEIVED")


        msg = request.form.get("msg")


        print("USER:", msg)


        if not msg:
            return "Please enter a message"


        chatbot = load_chatbot()



        response = chatbot.invoke(
            {
                "input": msg
            }
        )


        answer = response.get(
            "answer",
            "Sorry, I could not find an answer."
        )


        print("BOT:", answer)


        return answer



    except Exception as e:

        print("ERROR:", e)

        return "Something went wrong. Check server logs."






if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8080
    )