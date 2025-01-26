import debugpy
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
# routes for the API
from .routes import router_database

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="LanceDB API",
    description="API for LanceDB Viewer project",
    version="0.1.0"
)

app.include_router(router_database.router)

# CORS middleware configuration
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
print("Allowed origins: ", allowed_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Start debugpy listener
debugpy.listen(("0.0.0.0", 5678))
logging.debug("Debugpy listener started on 0.0.0.0:5678")

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)