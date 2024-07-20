from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set NVIDIA API Key
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

# Set MongoDB credentials
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["test"]
collection = db["audios"]

embedding_model = NVIDIAEmbeddings(model="NV-Embed-QA")
llm = ChatNVIDIA(model="meta/llama2-70b")
chat = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1", temperature=0.5, max_tokens=1000, top_p=1.0)

def load_existing_metadata(dest_embed_dir):
    metadata_file = os.path.join(dest_embed_dir, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                # Handle empty or corrupted JSON file
                return []
    return []

def save_metadata(dest_embed_dir, metadata):
    metadata_file = os.path.join(dest_embed_dir, "metadata.json")
    with open(metadata_file, 'w') as file:
        json.dump(metadata, file)

def load_patient_data(user_id):
    """
    Load user data from MongoDB for a specific user.
    Args:
        user_id: The ID of the user.

    Returns:
        A list of document contents as strings with metadata.
    """
    user_records = collection.find({"userId": ObjectId(user_id)}).sort("uploadedDate", 1)  # Sort by uploadedDate timestamp
    documents = []

    for record in user_records:
        if 'transcripts' in record:
            content = record['transcripts']
            summary = record.get('summary', '')
            uploaded_date = record['uploadedDate'].strftime("%d-%m-%Y %H:%M:%S")

            content += f"\nSummary: {summary}\nDate and Time: {uploaded_date}"

            documents.append({
                "content": content,
                "metadata": {
                    "userId": str(record['userId']),  # Convert ObjectId to string
                    "uploadedDate": uploaded_date,    # Format datetime to "dd-mm-yyyy HH:MM:SS"
                    "summary": summary,
                    "transcripts": record['transcripts']
                }
            })
    return documents


def index_docs(documents, user_id):
    dest_embed_dir = f"./embed/{user_id}"
    if not os.path.exists(dest_embed_dir):
        os.makedirs(dest_embed_dir)

    index_path = os.path.join(dest_embed_dir, "index.faiss")
    embeddings = NVIDIAEmbeddings(model="NV-Embed-QA")
    
    existing_metadata = load_existing_metadata(dest_embed_dir)
    existing_metadata_set = set(tuple(meta.items()) for meta in existing_metadata)

    new_documents = []
    new_metadata = []
    for doc in documents:
        if tuple(doc["metadata"].items()) not in existing_metadata_set:
            new_documents.append(doc)
            new_metadata.append(doc["metadata"])

    if not new_documents:
        print("No new documents to embed.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, length_function=len)
    texts = text_splitter.create_documents([doc["content"] for doc in new_documents], metadatas=[doc["metadata"] for doc in new_documents])

    print("New documents to be embedded:")
    for text in texts:
        print(f"Content: {text.page_content}, Metadata: {text.metadata}")

    if os.path.exists(index_path):
        update = FAISS.load_local(folder_path=dest_embed_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
        print(f"Initial number of entries in the index: {update.index.ntotal}")
        update.add_texts([text.page_content for text in texts], metadatas=[text.metadata for text in texts])
        update.save_local(folder_path=dest_embed_dir)
        print(f"Number of entries in the index after adding new data: {update.index.ntotal}")
    else:
        docsearch = FAISS.from_texts([text.page_content for text in texts], embedding=embeddings, metadatas=[text.metadata for text in texts])
        docsearch.save_local(folder_path=dest_embed_dir)
        print(f"Created new index with {docsearch.index.ntotal} entries")

    existing_metadata.extend(new_metadata)
    save_metadata(dest_embed_dir, existing_metadata)

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    user_id = data.get('userId')

    if not user_id:
        return jsonify({"error": "userId is required"}), 400

    # Load user data from MongoDB
    user_data = load_patient_data(user_id)

    if not user_data:
        return jsonify({"error": "No data found for the user"}), 404

    # Process and index the user data
    index_docs(user_data, user_id)
    return jsonify({"message": "Training completed successfully"}), 200

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    patient_id = data.get('patientId')
    question = data.get('question')

    if not patient_id or not question:
        return jsonify({"error": "patientId and question are required"}), 400

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(chat, chain_type="stuff", prompt=QA_PROMPT)

    retriever = FAISS.load_local(folder_path=f"./embed/{patient_id}", embeddings=embedding_model, allow_dangerous_deserialization=True).as_retriever()
    qa = ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=doc_chain,
        memory=memory,
        question_generator=question_generator,
    )

    response = qa.invoke({"question": question})
    return jsonify({"answer": response.get("answer")}), 200

@app.route('/add_data', methods=['POST'])
def add_data():
    data = request.json

    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        collection.insert_one(data)
        return jsonify({"message": "Data added successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
