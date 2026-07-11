from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from src.prompt import system_prompt
import os


app = Flask(__name__)


# Load environment variables
load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY



# Global variables
index_name = "medical-chatbot"

docsearch = None
rag_chain = None



# Cohere model
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


        print("Chatbot loaded successfully")


    return rag_chain





@app.route("/")
def index():

    return render_template("chat.html")





@app.route("/get", methods=["POST"])
def chat():

    try:

        print("CHAT REQUEST STARTED")


        msg = request.form.get("msg")


        print("USER MESSAGE:", msg)


        if not msg:

            return "Please enter a message"



        chatbot = load_chatbot()


        print("Running chatbot...")


        response = chatbot.invoke(
            {
                "input": msg
            }
        )


        print("RAW RESPONSE:", response)


        answer = response.get(
            "answer",
            "Sorry, no answer found"
        )


        print("BOT ANSWER:", answer)


        return str(answer)



    except Exception as e:


        print("ERROR:", e)


        return "Server Error: " + str(e), 500





if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8080
    )