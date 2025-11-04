from flask import Flask
from urllib.parse import unquote_plus
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)

# Load Hugging Face API key from environment variables (set this in Render)
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("Please set the HF_API_KEY environment variable in Render.")

client = InferenceClient(api_key=HF_API_KEY)

@app.route("/")
def home():
    return "AI API is live! Use /<your-query> to get a response."

@app.route("/<path:query>")
def ai_endpoint(query):
    # Decode URL-encoded multi-word query
    query = unquote_plus(query).strip()

    # Prepare conversation for the AI
    messages = [
        {"role": "assistant", "content": "Hello! I'm here to assist you."},
        {"role": "user", "content": f"{query} - reply under 200 characters. Don't mention character count."}
    ]

    try:
        # Generate AI response (stream=True works fine)
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

        # Return plain text, max 256 chars (Nightbot)
        return assistant_reply.strip()[:256]

    except Exception as e:
        # Catch errors and return friendly message
        return f"⚠️ Error: {str(e)}"

if __name__ == "__main__":
    # Render assigns its own port via $PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
