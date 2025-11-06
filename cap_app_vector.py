import streamlit as st
import time
import json
import requests
import hashlib
from requests.auth import HTTPBasicAuth
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
import boto3
import json as json_lib

# === CONFIG ===
CB_CONN_STRING = "couchbases://cb<YourID>...cloud.couchbase.com"  # TLS
CB_BUCKET = "block_convo"
CB_USERNAME = "chatbot"
CB_PASSWORD = "Chatbot123!"
CB_TLS_CERT = "/Users/melboulos/Downloads/qvc-demo-root-certificate.txt"

# Full FTS URL (bucket.scope.index)
CB_FTS_URL = "https://cb<YourID>...cloud.couchbase.com:18094/api/index/block_convo._default.chat_vector_index/query"

BEDROCK_REGION = "us-east-1"
EMBED_MODEL_ID = "amazon.titan-embed-text-v1"
LLM_MODEL_ID = "meta.llama3-70b-instruct-v1:0"

# === BEDROCK CLIENT ===
bedrock_runtime = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

# === UTILS ===
def hash_question(question):
    return hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()

def get_embedding(text):
    body = {"inputText": text}
    try:
        response = bedrock_runtime.invoke_model(
            modelId=EMBED_MODEL_ID,
            body=json_lib.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        response_body = json_lib.loads(response["body"].read())
        return response_body.get("embedding")
    except Exception as e:
        st.error(f"Error getting embedding: {e}")
        return None

def generate_answer(question):
    prompt_text = f"Answer concisely:\n\n{question}\n\nAnswer:"
    body = {"prompt": prompt_text}
    try:
        response = bedrock_runtime.invoke_model(
            modelId=LLM_MODEL_ID,
            body=json_lib.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        response_body = json_lib.loads(response["body"].read())
        generated = response_body.get("generation") or (response_body.get("generations") or [{}])[0].get("text")
        return generated.strip() if generated else "No answer generated."
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return None

def vector_search_rest(embedding_vector, k=3, ef_search=8):
    query_json = {
        "fields": ["question", "answer", "_score"],
        "knn": [
            {
                "k": k,
                "field": "embedding_vector",
                "vector": embedding_vector
            }
        ],
        "control": {
            "ef_search": ef_search
        }
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            CB_FTS_URL,
            auth=HTTPBasicAuth(CB_USERNAME, CB_PASSWORD),
            headers=headers,
            data=json.dumps(query_json),
            verify=CB_TLS_CERT  # ✅ Use TLS cert for HTTPS
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Couchbase vector search error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calling vector search: {e}")
        return None

def connect_couchbase():
    cluster = Cluster(
        CB_CONN_STRING,  # couchbases:// for TLS
        ClusterOptions(PasswordAuthenticator(CB_USERNAME, CB_PASSWORD))
    )
    bucket = cluster.bucket(CB_BUCKET)
    return bucket.default_collection()

def store_qa(question, answer, embedding_vector):
    collection = connect_couchbase()
    hash_key = hash_question(question)
    doc_id = f"qa::{hash_key}"
    doc = {
        "type": "block_convo",
        "question": question,
        "answer": answer,
        "embedding_vector": embedding_vector
    }
    try:
        collection.upsert(doc_id, doc)
        st.write(f"✅ Stored new doc under ID: {doc_id}")
    except Exception as e:
        st.error(f"Error inserting document: {e}")

def fetch_doc_by_key(question):
    try:
        collection = connect_couchbase()
        doc_id = f"qa::{hash_question(question)}"
        return collection.get(doc_id).content_as[dict]
    except Exception:
        return None

# === STREAMLIT UI ===
st.title("Semantic Search Chatbot 📚🔍 (Capella TLS)")

user_question = st.text_input("Enter your question:")

if user_question:
    start_time = time.time()
    embedding = get_embedding(user_question)
    if not embedding:
        st.error("Failed to get embedding.")
        st.stop()
    st.write("🧬 Embedding generated.")

    # Vector search with ef_search = 32
    cb_results = vector_search_rest(embedding, k=3, ef_search=32)
    vector_time = time.time() - start_time
    hits_count = len(cb_results['hits']) if cb_results and 'hits' in cb_results else 0
    st.write(f"📊 Vector search returned {hits_count} hits")

    collection = connect_couchbase()  # Reuse collection

    top_hit_doc = None
    answer, source = None, None

    # Show top 3 question + score results
    if cb_results and "hits" in cb_results and cb_results["hits"]:
        for hit in cb_results["hits"]:
            score = hit.get("score", 0)
            try:
                doc = collection.get(hit['id']).content_as[dict]
                st.write(f"• Question: {doc.get('question')} | Score: {score:.3f}")
                if score >= 0.9 and not top_hit_doc: # the lower I set threshold the weaker my matches
                    top_hit_doc = doc
                    answer = doc.get("answer")
                    source = "Couchbase (vector search)"
            except Exception as e:
                st.error(f"Error fetching doc {hit['id']}: {e}")

    # Always do key cache lookup
    keycache_start = time.time()
    cached_doc = fetch_doc_by_key(user_question)
    keycache_time = time.time() - keycache_start

    if not top_hit_doc and cached_doc:
        st.write("🔑 Found doc in key cache.")
        top_hit_doc = cached_doc
        answer = cached_doc.get("answer")
        source = "Couchbase (key cache)"

    if not top_hit_doc:
        st.write("❌ No matching doc in vector or key cache.")
        answer = generate_answer(user_question)
        if answer:
            st.write("⚙️ Storing new doc…")
            store_qa(user_question, answer, embedding)
            source = "Bedrock LLM"
        else:
            st.error("❌ Failed to generate answer. No doc stored.")
            st.stop()

    total_time = time.time() - start_time

    st.markdown(f"**Answer:** {answer}")
    st.markdown(f"*Source:* {source}")
    st.markdown(f"*Vector search time:* {vector_time:.3f} sec")
    st.markdown(f"*Key cache lookup time:* {keycache_time:.3f} sec")
    st.markdown(f"*Total response time:* {total_time:.2f} sec")
