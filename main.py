from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"hello": "world"})

@app.route("/<query>")
def ai_endpoint(query):
    return jsonify({"result": f"AI response for {query}"})
