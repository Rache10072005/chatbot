from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():

    msg = request.form.get("msg")

    return "Hello, chatbot is working! You asked: " + msg


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080
    )