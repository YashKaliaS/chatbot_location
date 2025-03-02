from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import faiss
import os
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load a pre-trained embedding model (CPU-friendly)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Global variables to store places, their textual descriptions, and FAISS index
place_data = []      # Raw place data from Google Places API
documents = []       # List of text descriptions for FAISS indexing
index = None         # FAISS index instance


# Configure Gemini (Google Generative AI)
genai.configure(api_key=os.getenv('gemini_key'))
llm = genai.GenerativeModel('gemini-1.5-flash-8b')  # Use Gemini Pro model

# Google Places API key
API_KEY = os.getenv('GOOGLE_API_KEY')
print("Stage 1: Application Starting")

# --- Location & Places Fetching ---

def get_live_location():
    print("Stage 2: Fetching live location...")
    url = "https://www.googleapis.com/geolocation/v1/geolocate?key=" + API_KEY
    response = requests.post(url, json={})
    data = response.json()
    # print(data)
    return {"latitude": data["location"]["lat"], "longitude": data["location"]["lng"]}

    try:
        # Using static coordinates for now. Replace with a real API call if needed.
        return {
            "latitude": 19.1196,
            "longitude": 72.8834,
        }
    except Exception as e:
        print("Error fetching live location:", e)
        return None

def fetch_nearby_places(latitude, longitude, radius=1000, max_results=10):
    print("Stage 3: Fetching nearby places...")
    url = "https://places.googleapis.com/v1/places:searchNearby"
    # Expanded field mask to retrieve additional details:
    field_mask = (
        "places.displayName,places.id,places.formattedAddress,places.rating,places.userRatingCount,"
        "places.reviews,places.currentOpeningHours,places.regularOpeningHours,places.internationalPhoneNumber,"
        "places.websiteUri,places.priceLevel,places.priceRange,places.types,places.addressComponents,"
        "places.businessStatus,places.photos"
    )
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": field_mask,
    }
    payload = {
        "includedTypes": ["restaurant", "cafe", "hospital"],
        "maxResultCount": max_results,
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": latitude,
                    "longitude": longitude,
                },
                "radius": radius,
            }
        },
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        print("Stage 4: Successfully fetched places")
        places = response.json().get("places", [])
        print(f"Fetched {len(places)} places.")
        return places
    else:
        print("Failed to fetch nearby places:", response.status_code, response.text)
        return []

def build_faiss_index():
    global documents, index, place_data
    if not place_data:
        print("No place data available to build FAISS index.")
        return

    print("Stage 5: Building FAISS index from place data...")
    documents = []
    for p in place_data:
        # Extract fields safely
        display_name = p.get('displayName', {}).get('text', 'N/A')
        formatted_address = p.get('formattedAddress', 'No Address')
        rating = p.get('rating', 'N/A')
        user_rating_count = p.get('userRatingCount', 'N/A')
        current_hours = p.get('currentOpeningHours', 'N/A')
        regular_hours = p.get('regularOpeningHours', 'N/A')
        phone = p.get('internationalPhoneNumber', 'N/A')
        price_level = p.get('priceLevel', 'N/A')
        price_range = p.get('priceRange', 'N/A')
        website = p.get('websiteUri', 'N/A')
        types = ", ".join(p.get('types', [])) if p.get('types') else 'N/A'
        business_status = p.get('businessStatus', 'N/A')
        # For reviews, join texts after ensuring they are strings
        reviews = " ".join([str(rev.get('text', '')) for rev in p.get("reviews", [])])
        
        # Construct a document string that includes common details
        doc = (
            f"{display_name}. Address: {formatted_address}. Rating: {rating} "
            f"(from {user_rating_count} reviews). Opening Hours: Current - {current_hours}, Regular - {regular_hours}. "
            f"Phone: {phone}. Price: {price_level} ({price_range}). Website: {website}. "
            f"Types: {types}. Business Status: {business_status}. Reviews: {reviews}"
        )
        documents.append(doc)
    
    print("Documents for FAISS index:")
    for doc in documents:
        print(" -", doc)
    
    # Generate embeddings for each document
    doc_embeddings = embedding_model.encode(documents, convert_to_numpy=True)
    embedding_dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)  # Create FAISS index with L2 (Euclidean) distance
    index.add(doc_embeddings)
    print("FAISS index built successfully. Total documents indexed:", index.ntotal)

# --- API Endpoints ---

@app.route('/fetch-places', methods=['POST'])
def fetch_places():
    global place_data
    print("Stage 6: /fetch-places endpoint called.")
    location = get_live_location()
    if not location:
        return jsonify({"error": "Failed to fetch live location."}), 400

    latitude = location["latitude"]
    longitude = location["longitude"]
    print(f"Location received: Latitude={latitude}, Longitude={longitude}")
    
    places = fetch_nearby_places(latitude, longitude)
    if not places:
        return jsonify({"error": "No places found."}), 400

    place_data = places
    build_faiss_index()  # Build FAISS index after fetching places

    return jsonify({"message": f"Indexed {len(places)} places."})

def search_faiss(query, top_k=2):
    global index, documents
    print("\n🔎 Searching for query:", query)
    if index is None or len(documents) == 0:
        print("FAISS index is not built or documents list is empty.")
        return []
    
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    
    print("\n🔍 Top Relevant Documents:")
    for i, idx in enumerate(indices[0]):
        if idx < len(documents):
            print(f"   {i+1}. {documents[idx]} (Score: {distances[0][i]:.4f})")
        else:
            print(f"   {i+1}. Index {idx} out of range for documents.")
    
    return [documents[idx] for idx in indices[0] if idx < len(documents)]

def generate_response(query):
    location = get_live_location()
    if not location:
        return jsonify({"error": "Failed to fetch live location."}), 400

    latitude = location["latitude"]
    longitude = location["longitude"]
    print("\n🔎 Received Query:", query)
    
    # Always check FAISS for relevant documents
    print("🔍 Checking FAISS for relevant documents...")
    relevant_docs = search_faiss(query, top_k=2)
        
    if not relevant_docs:
        print("⚠️ FAISS index is not built or no relevant documents found. Falling back to plain query.")
        prompt = query
    else:
        print("\n📄 Retrieved Documents for Context:")
        for i, doc in enumerate(relevant_docs):
            print(f"   {i+1}. {doc}")
        context = "\n".join(relevant_docs)
        prompt = f"Context:\n{context}\n\nAnswer the query: {query} use context as much as possible(reviews and details about places are mentioned in the context) ,dont day this in the answer that you are using real time location service ,just use the context and dont mention about it"

    print("\n📜 Final Prompt Sent to Gemini:\n", prompt)

    try:
        model = genai.GenerativeModel("gemini-1.5-flash-8b")  
        response = model.generate_content(prompt)

        if not response or not response.text:
            print("⚠️ Gemini API returned an empty response.")
            return "The AI could not generate a response."
        
        print("\n✅ AI Response Generated Successfully!")
        return response.text.strip()

    except Exception as e:
        print(f"❌ Error during Gemini API call: {e}")
        return "An error occurred while generating the response."

@app.route('/query', methods=['POST'])
def query_endpoint():
    print("Stage 7: /query endpoint called.")
    try:
        req_data = request.get_json()
        if not req_data:
            print("No JSON data received in the request.")
            return jsonify({"error": "No JSON data received."}), 400
        
        query_text = req_data.get('query', '')
        if not query_text:
            print("Query text is empty.")
            return jsonify({"error": "Query text is required."}), 400
        
        print(f"Query received: {query_text}")
        response_text = generate_response(query_text)
        return jsonify({"response": response_text})
    except Exception as e:
        print("❌ Exception in /query endpoint:", e)
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
