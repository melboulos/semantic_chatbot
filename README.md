# 📚🔍 Semantic Search Chatbot

A Streamlit-based semantic chatbot that uses text embeddings, vector search in Couchbase, and AWS Bedrock for LLM fallback. It retrieves the most semantically similar questions from a vector index, checks a key-based cache, and generates answers when necessary — while displaying detailed response timings and hit sources.

---
## 🚀 Features

- 🔑 **Key cache lookup** for exact question matches.
- 🧬 **Text embeddings** generated via Amazon Bedrock Titan.
- 📊 **Vector search** in Couchbase FTS with top-3 nearest neighbors.
- 🧠 **LLM fallback** using Meta Llama 3 via AWS Bedrock.
- 📈 Detailed response timing and source breakdown.
- ✅ Automatic storage of new Q&A pairs with embeddings into Couchbase.
- 📊 Streamlit UI for interactive querying and result inspection.

---

## 📦 Tech Stack

- **Python 3.13**
- **Streamlit**
- **Couchbase Server 7.6+**
- **AWS Bedrock**
- **Boto3**
- **Requests**

---

## 🛠️ Setup Instructions

### 1️⃣ Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2️⃣ Configure Couchbase and Bedrock
Update the following values in app.py:


CB_CONN_STRING = "couchbase://localhost"
CB_BUCKET = "block_convo"
CB_USERNAME = "chatbot"
CB_PASSWORD = "chatbot123"
BEDROCK_REGION = "us-east-1"
EMBED_MODEL_ID = "amazon.titan-embed-text-v1"
LLM_MODEL_ID = "meta.llama3-70b-instruct-v1:0"

Ensure your Couchbase cluster has:

A block_convo bucket

A vector-capable FTS index named chat_vector_index

3️⃣ Run the App

streamlit run app.py

User question: how many veterans in VT

🧬 Embedding generated.

📊 Vector search returned 3 hits:
• Question: how many veterans in MA | Score: 0.871
• Question: how many veterans in NH | Score: 0.868
• Question: how many veterans in CT | Score: 0.766

✅ Retrieved doc for: how many veterans in MA

Answer: Approximately 375,000 veterans live in Massachusetts.

Source: Couchbase (vector search)
Response time: 0.42 sec
