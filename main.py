from flask import Flask
from urllib.parse import unquote_plus
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)

# Load Hugging Face API key from environment variables
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("Please set the HF_API_KEY environment variable in Render.")

client = InferenceClient(api_key=HF_API_KEY)

@app.route("/")
def home():
    return "AI API is live! Use /<your-query> to get a response."

@app.route("/<path:query>")
def ai_endpoint(query):
    query = unquote_plus(query).strip()

    messages = [
        {"role": "assistant", "content": "Hello! I'm here to assist you."},
        {"role": "user", "content": f"{query} - reply under 200 characters. Don't mention character count."}
    ]

    try:
        # Non-streaming call for Nightbot compatibility
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=messages,
            temperature=0.5,
            max_tokens=256,  # Keep response short for Nightbot
            top_p=0.7,
            stream=False  # <- Key change to prevent timeouts
        )

        assistant_reply = response.choices[0].message.content

        if not assistant_reply:
            return "⚠️ AI returned no response. Try again!"

        return assistant_reply.strip()[:256]  # Trim to 256 chars for Nightbot

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
