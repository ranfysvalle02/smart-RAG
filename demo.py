import os  
import logging  
import time  
from enum import Enum  
from typing import Optional, List, Dict, Any, Callable, Union  
  
from flask import Flask, request, jsonify  
from pydantic import BaseModel, Field  
import openai  
from openai import AzureOpenAI  

from pymongo import MongoClient  
from pymongo.errors import OperationFailure  
from pymongo.operations import SearchIndexModel  
  
# Initialize logging  
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)  
  
################################  
########### SETTINGS ###########  
################################  
# Load environment variables from .env file if it exists  
from dotenv import load_dotenv  
load_dotenv()  
# MongoDB configuration  
MONGO_URI = os.environ.get('MONGO_URI', "mongodb://localhost:27017/?directConnection=true")  
  
# Sample dataset database and collection names  
DATABASE_NAME = 'sample_mflix'  # Database containing the sample dataset  
COLLECTION_NAME = 'embedded_movies'  # Collection containing the movies data  
  
# Initialize Flask app  
app = Flask(__name__)  
  
################################  
########### APP SETTINGS ###########  
################################  
  
# LLM Type from environment variable  
class LLMType(str, Enum):  
    """LLM types"""  
    OPENAI = "OPENAI"  
    CREDAL = "CREDAL"  
  
class AppSettings(BaseModel):  
    """App settings configuration"""  
    LLM_TYPE: LLMType = Field(default_factory=lambda: os.environ.get("LLM_TYPE", "OPENAI"))  
  
# Initialize app settings  
app_settings = AppSettings()  
  
##########################  
########### DB ###########  
##########################  
  
class DBConfig(BaseModel):  
    """DB Config."""  
    mongo_uri: str = MONGO_URI  
  
class MovieDocument(BaseModel):  
    """Movie document model for database."""  
    _id: Any  
    title: str  
    plot: Optional[str]  
    plot_embedding: Optional[List[float]] = None  # Embedding for plot field  
  
# --------------------------------------------------------------------  
# Main CustomMongoClient  
# --------------------------------------------------------------------  
class CustomMongoClient(MongoClient):  
    def __init__(self, mongo_uri: str, get_embedding: Callable[[str], List[float]] = None, **kwargs):  
        super().__init__(mongo_uri, **kwargs)  
        self.get_embedding = get_embedding  
  
    def _create_search_index(  
        self,  
        database_name: str,  
        collection_name: str,  
        index_name: str,  
        distance_metric: str = "cosine",  
        embedding_field: str = "plot_embedding",  
        num_dimensions: int = 1536,  
    ) -> None:  
        """  
        Creates a search index on the specified collection if it does not already exist.  
        """  
        try:  
            collection = self[database_name][collection_name]  
  
            if self.index_exists(database_name, collection_name, index_name):  
                logger.info(f"Search index '{index_name}' already exists in collection '{collection_name}'.")  
                return  
  
            logger.info(f"Creating search index '{index_name}' for collection '{collection_name}'.")  
  
            # Create your index model, then create the search index  
            search_index_model = SearchIndexModel(  
                definition={  
                    "fields": [  
                        {  
                            "type": "vector",  
                            "numDimensions": num_dimensions,  
                            "path": embedding_field,  
                            "similarity": distance_metric  
                        },  
                        {  
                            "type": "string",  
                            "path": "genres",  
                            "analyzer": "keyword"  
                        },  
                        {  
                            "type": "string",  
                            "path": "type",  
                            "analyzer": "keyword"  
                        }  
                    ]  
                },  
                name=index_name,  
                type="vectorSearch"  
            )  
  
            result = collection.create_search_index(model=search_index_model)  
            logger.info(f"New search index named '{result}' is building.")  
  
            # Wait for initial sync to complete  
            logger.info("Polling to check if the index is ready. This may take up to a minute.")  
  
            predicate = lambda index: index.get("queryable") is True  
  
            while True:  
                indices = list(collection.list_search_indexes(result))  
                if len(indices) and predicate(indices[0]):  
                    break  
                time.sleep(5)  
            logger.info(f"Index '{result}' is ready for querying.")  
  
        except OperationFailure as e:  
            logger.error(f"Operation failed: {e.details.get('errmsg', str(e))}")  
            raise  
        except Exception as e:  
            logger.error(f"Failed to create search index '{index_name}': {e}")  
            raise  
  
    def index_exists(self, database_name: str, collection_name: str, index_name: str) -> bool:  
        """  
        Checks if a specific search index exists in the collection.  
        """  
        try:  
            collection = self[database_name][collection_name]  
            indexes = list(collection.list_search_indexes())  
            logger.debug(f"Retrieved indexes: {indexes}")  
  
            # Iterate through the indexes to check for a matching name  
            for index in indexes:  
                retrieved_name = index.get("name", "")  
                logger.debug(f"Checking index: {retrieved_name}")  
                if retrieved_name == index_name:  
                    logger.info(f"Found existing index '{index_name}'.")  
                    return True  
  
            logger.info(f"Index '{index_name}' does not exist in collection '{collection_name}'.")  
            return False  
  
        except OperationFailure as e:  
            logger.error(f"Operation failure while checking index existence: {e}")  
            return False  
        except Exception as e:  
            logger.error(f"Error checking search index existence for '{index_name}': {e}")  
            return False  
  
    def vector_search(  
        self,  
        query: Union[str, List[float]],  
        limit: int = 5,  
        database_name: str = "",  
        collection_name: str = "",  
        index_name: str = "",  
        embedding_field: str = "plot_embedding",  
        filters: Optional[Dict[str, Any]] = None,  
        distance_metric: str = "cosine",  # Ensure distance metric matches index  
        threshold: float = 0.5,  
    ) -> List[Dict]:  
        """  
        Performs a vector-based search using the specified search index.  
          - If `query` is a string, we call self.get_embedding(query).  
          - If `query` is a list/tuple, we assume it's already the embedding.  
        """  
        # Determine query embedding  
        if isinstance(query, str):  
            if self.get_embedding is None:  
                logger.error("No get_embedding function provided to generate query embedding.")  
                return []  
            query_embedding = self.get_embedding(query)  
        elif isinstance(query, (list, tuple)):  
            query_embedding = query  # assume user supplied embedding  
        else:  
            logger.error(f"Query type {type(query)} not supported for vector search.")  
            return []  
  
        if not self.index_exists(database_name, collection_name, index_name):  
            logger.error(f"Search index '{index_name}' does not exist.")  
            return []  
        if query_embedding is None:  
            logger.error(f"Failed to generate or receive embedding for query: {query}")  
            return []  
  
        try:  
            collection = self[database_name][collection_name]  
            query_vector = query_embedding  
  
            # Variables adjusted per your provided pipeline  
            search_index_name = index_name  
            path = embedding_field  
  
            # Build the pipeline  
            pipeline = [  
                {  
                    "$vectorSearch": {  
                        "index": search_index_name,  
                        "queryVector": query_vector,  
                        "limit": limit * 10,  # Increase numCandidates to fetch more results for uniqueness  
                        "numCandidates": limit * 10,  
                        "path": path,  
                    }  
                },  
                {"$set": {"score": {"$meta": "vectorSearchScore"}}},  
                {"$match": {"score": {"$gt": threshold}}},  
                {"$sort": {"score": -1}},  
                {"$project": {path: 0}},  
            ]  
  
            results = list(collection.aggregate(pipeline))  
            logger.info(f"Vector search completed. Found {len(results)} documents.")  
  
            # Optionally, collect unique results or further process results if needed  
            # For simplicity, limit to the requested number of documents  
            results = results[:limit]  
  
            return results  
        except Exception as e:  
            logger.error(f"Error during vector search: {e}")  
            return []  
  
# --------------------------------------------------------------------  
# Repository Class  
# --------------------------------------------------------------------  
class Repository:  
    """  
    Repository for interacting with the database.  
    """  
  
    def __init__(self, config: DBConfig, get_embedding: Callable[[str], List[float]] = None) -> None:  
        self._mongo_uri = config.mongo_uri  
        self._client = CustomMongoClient(self._mongo_uri, get_embedding=get_embedding)  
  
        self._db = self._client[DATABASE_NAME]  
        self._collection_name = COLLECTION_NAME  
        self._collection = self._db[self._collection_name]  
  
        logger.info(f"Connected to database: {self._db.name}")  
  
    def create_embedding_index(self):  
        """Create a vector search index on the 'plot_embedding' field."""  
        index_name = "vector_index"  
        self._client._create_search_index(  
            database_name=self._db.name,  
            collection_name=self._collection_name,  
            index_name=index_name,  
            distance_metric="cosine",  
            embedding_field='plot_embedding',  
            num_dimensions=1536,  # Update with the correct dimension of your embeddings  
        )  
  
    def vector_search(  
        self,  
        query: Union[str, List[float]],  
        limit: int = 5,  
    ) -> List[Dict]:  
        """Perform vector search on the movies collection."""  
        index_name = "vector_index"  
        results = self._client.vector_search(  
            query=query,  
            limit=limit,  
            database_name=self._db.name,  
            collection_name=self._collection_name,  
            index_name=index_name,  
            embedding_field='plot_embedding',  
            distance_metric="cosine",  # Ensure the distance metric matches the index  
            threshold=0.5,  # Add threshold parameter if needed  
        )  
        return results  
  
    def get_movie_by_id(self, movie_id: Any) -> Optional[Dict]:  
        """Retrieve a movie document by its _id."""  
        try:  
            movie = self._collection.find_one({'_id': movie_id}, {'plot_embedding': 0})  
            return movie  
        except Exception as e:  
            logger.error(f"Error retrieving movie by ID '{movie_id}': {e}")  
            return None  
  
##############################  
########### OPENAI ###########  
##############################  
  
# Embeddings settings from environment variables  
EMBEDDINGS_CHUNK_SIZE = int(os.environ.get('EMBEDDINGS_CHUNK_SIZE', '800'))  
EMBEDDINGS_CHUNK_OVERLAP = int(os.environ.get('EMBEDDINGS_CHUNK_OVERLAP', '50'))  
  
class OpenAISettings(BaseModel):  
    """OpenAI settings configuration"""  
  
    azure_openai_api_key: str = Field(default_factory=lambda: os.environ.get('AZURE_OPENAI_API_KEY', ''))  
    azure_openai_endpoint: str = Field(default_factory=lambda: os.environ.get('AZURE_OPENAI_ENDPOINT', ''))  
    azure_openai_api_version: str = Field(default_factory=lambda: os.environ.get('AZURE_OPENAI_API_VERSION', '2023-05-15'))  
    azure_openai_model: str = Field(default_factory=lambda: os.environ.get('AZURE_OPENAI_MODEL', ''))  
    azure_openai_embeddings_model: str = Field(default_factory=lambda: os.environ.get('AZURE_OPENAI_EMBEDDINGS_MODEL', ''))  
  
class ChatResponse(BaseModel):  
    """Chat response model"""  
    response: str = Field(default="")  
  
class OpenAIClient:  
    """Azure OpenAI Client"""  
  
    def __init__(self, settings: OpenAISettings) -> None:  
        self._endpoint = settings.azure_openai_endpoint.rstrip('/') + '/'  # Ensure it ends with '/'  
        self._api_key = settings.azure_openai_api_key  
        self._api_version = settings.azure_openai_api_version  
        self._chat_model = settings.azure_openai_model  
        self._embeddings_model = settings.azure_openai_embeddings_model  
  
        # Check if endpoint and API key are set  
        if not self._endpoint or not self._api_key:  
            logger.error("Azure OpenAI endpoint or API key is not set.")  
            raise ValueError("Azure OpenAI endpoint or API key is not set.")  
  
        # Initialize OpenAI client  
        openai.api_type = "azure"  
        openai.api_base = self._endpoint  # Should end with a forward slash  
        openai.api_version = self._api_version  
        openai.api_key = self._api_key  
        self._client = AzureOpenAI(  
            azure_endpoint=self._endpoint,  
            api_version=self._api_version,  
            api_key=self._api_key,  
        )  
    def get_embedding(self, text: str) -> list[float]:  
        """Get embeddings for the given text."""  
  
        try:  
            response = self._client.embeddings.create(  
                input=[text],  
                model="text-embedding-ada-002" ,  
            )  
            return response.data[0].embedding  
        except Exception as e:  
            print(f"Error getting embeddings: {e}")  
            raise  
    
  
    def generate_chat_response(  
        self,  
        query: str,  
        context: List[str] = None,  
        stream: bool = False,  
        temperature: float = 0.0,  
        force_ai_response: bool = False,  # If True, will always return AI response  
    ) -> dict:  
        """Get chat response for the given messages."""  
        try:  
            # Prepare context string for the prompt  
            if context:  
                # Join multiple context documents with a clear separator  
                joined_context_str = "\n---\n".join(context)  
            else:  
                joined_context_str = "No context provided."  
  
            if not force_ai_response:  
                messages = [  
                    {  
                        "role": "system",  
                        "content": f"""  
                            Answer the user's query based on the provided movie descriptions ONLY.  
                            Do not make up any part of your response.  
                            [Movie Descriptions]  
                            {joined_context_str}  
                            [/Movie Descriptions]  
                        """  
                    },  
                    {  
                        "role": "user",  
                        "content": f"{query}"  
                    }  
                ]  
            else:  
                messages = [  
                    {  
                        "role": "system",  
                        "content": "Answer the user's query to the best of your ability."  
                    },  
                    {  
                        "role": "user",  
                        "content": query  
                    }  
                ]  
            
            response = self._client.chat.completions.create(
                messages=messages,  
                temperature=temperature,  
                model="gpt-4o",
            )  
  
            chat_response_content = response.choices[0].message.content
            if chat_response_content is None:  
                logger.error("OpenAI chat completion returned None content.")  
                return {"error": "Failed to get response from AI model"}  
            # Since we are not enforcing a JSON response, we can return the content directly  
            chat_response = {"response": chat_response_content.strip()}  
            logger.info(f"Chat response: {chat_response}")  
            return chat_response  
        except Exception as e:  
            logger.error(f"Error getting chat response: {e}")  
            return {"error": str(e)}  
  
def get_openai_client(settings: OpenAISettings = None) -> OpenAIClient:  
    """Get a new instance of the OpenAI client."""  
    if settings is None:  
        settings = OpenAISettings()  
    return OpenAIClient(settings)  
  
###########################  
########### APP ###########  
###########################  
  
# Initialize OpenAI client  
openai_client = get_openai_client()  
  
# Initialize Repository  
repository = Repository(DBConfig(), get_embedding=openai_client.get_embedding)  
  
# Create the search index if it doesn't exist  
repository.create_embedding_index()  
  
def get_rag_response_internal(query, force_ai_response=False):  
    # Implement the RAG service logic  
    logger.info(f"Processing RAG response for query: {query}, force_ai_response: {force_ai_response}")  
  
    if force_ai_response:  
        # Return AI-generated response without context  
        response = openai_client.generate_chat_response(query, force_ai_response=True)  
        context_texts = []  # No context when forcing AI response  
    else:  
        # Perform vector search and get context documents  
        search_results = repository.vector_search(  
            query=query,  
            limit=3  # You can adjust the limit as needed  
        )  
        logger.info(f"{len(search_results)} pieces of context found, querying OpenAI for response")  
  
        # Retrieve the context texts  
        context_texts = []  
        for result in search_results:  
            movie = repository.get_movie_by_id(result['_id'])  
            if movie:  
                context_texts.append(f"Title: {movie['title']}\nDescription: {movie.get('plot')}")  
        if not context_texts:  
            logger.warning("No context texts found; will proceed without context.")  
        response = openai_client.generate_chat_response(query, context_texts)  
    return {"response": response.get("response", ""), "context": context_texts}  
  
# Flask routes and handlers  
  
@app.route('/api/health', methods=['GET'])  
def health_check():  
    """Health check endpoint."""  
    return jsonify({"status": "ok"}), 200  
  
@app.route('/api/query', methods=['GET', 'POST'])  
def rag_service():  
    """RAG service endpoint."""  
    if request.method == 'GET':  
        query = request.args.get('query')  
        force_ai_response = request.args.get('force_ai_response', 'false').lower() == 'true'  
    else:  
        data = request.json  
        query = data.get('query')  
        force_ai_response = data.get('force_ai_response', False)  
    if not query:  
        return jsonify({"error": "query parameter is required"}), 400  
    logger.info(f"Received query for RAG service: {query}, force_ai_response: {force_ai_response}")  
  
    response = get_rag_response_internal(query, force_ai_response)  
    return jsonify(response), 200  
  
################# RUN THE FLASK APP #################  
  
if __name__ == '__main__':  
    # Run the Flask app  
    app.run(host='0.0.0.0', port=5000, debug=True)  
