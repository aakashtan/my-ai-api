from flask import Flask
from urllib.parse import unquote_plus
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)

HF_API_KEY = os.getenv("HF_API_KEY")
client = InferenceClient(api_key=HF_API_KEY)

@app.route("/<path:query>")
def ai_endpoint(query):
    # Decode URL-encoded query
    query = unquote_plus(query)

    messages = [
        {"role": "assistant", "content": "Hello! I'm here to assist you."},
        {"role": "user", "content": f"{query} - reply under 200 characters."}
    ]

    try:
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

        return assistant_reply.strip()[:256]

    except Exception as e:
        # Return a friendly error message instead of 500
        return f"⚠️ Error: {str(e)}"
