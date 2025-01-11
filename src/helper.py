import os
from pdfminer.high_level import extract_text
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from google.cloud import aiplatform


load_dotenv()
Google_API_Key = os.getenv("Google_API_Key")
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") 

os.environ['GOOGLE_CLOUD_PROJECT'] = GCP_PROJECT_ID 


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_content = BytesIO(pdf.read())
        text += extract_text(pdf_content)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = VertexAIEmbeddings(
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),  # GCP Project ID
        location="us-central1",                     # Location of the model
        model_name="textembedding-gecko",           # Example Vertex AI embedding model
    )
    vector_store = FAISS.from_texts(text_chunks, embeddings=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    # Replace with appropriate Gemini API client
    from google.cloud import aiplatform

    # Initialize the Gemini API client
    client = aiplatform.gapic.PredictionServiceClient()

    # Define the endpoint and model name
    endpoint = "projects/your_gcp_project_id/locations/us-central1/endpoints/your_endpoint_id"  # Replace with your endpoint ID
    model_name = "projects/your_gcp_project_id/locations/us-central1/models/your_model_id"  # Replace with your model ID

    # Create a custom LLM using the Gemini API
    class GeminiLLM:
        def __init__(self):
            self.client = client
            self.endpoint = endpoint
            self.model_name = model_name

        def predict(self, prompt):
            instance = { "content": prompt }
            parameters = { "deployMode": "DEFAULT" }
            request = aiplatform.gapic.PredictRequest(
                endpoint=self.endpoint,
                instances=[instance],
                parameters=parameters,
            )
            response = self.client.predict(request=request)
            return response.predictions[0]["content"]

    llm = GeminiLLM()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain
