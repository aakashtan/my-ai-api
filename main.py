from flask import Flask
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)

# Get your API key securely from environment variables
HF_API_KEY = os.getenv("HF_API_KEY")
client = InferenceClient(api_key=HF_API_KEY)

@app.route('/')
def home():
    return "AI API is live!"

@app.route("/<query>")
def ai_endpoint(query):
    # Add short-answer condition
    query_with_condition = (
        f"{query} - reply under 200 characters total. "
        "Don't mention how many characters you used."
    )

    # Use a *fresh* conversation each time
    messages = [
        {"role": "assistant", "content": "Hello! I'm here to assist you."},
        {"role": "user", "content": query_with_condition}
    ]

    # Generate response
    stream = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=messages,
        temperature=0.5,
        max_tokens=512,
        top_p=0.7,
        stream=True
    )

    assistant_reply = ""
    for chunk in stream:
        if chunk.choices[0].delta.get("content"):
            assistant_reply += chunk.choices[0].delta["content"]

    # Return trimmed text (Nightbot needs plain text)
    return assistant_reply.strip()[:256]

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


