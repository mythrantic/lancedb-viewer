import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
# routes for the API
from routes import router_database

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


# Tailwind HTML Template
html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LanceDB Viewer</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 text-gray-800">
    <div class="min-h-screen flex flex-col items-center justify-center">
        <header class="bg-blue-500 w-full py-6 shadow-lg">
            <h1 class="text-center text-white text-4xl font-bold">LanceDB Viewer</h1>
        </header>

        <main class="flex flex-col items-center mt-10">
            <h2 class="text-2xl font-semibold text-gray-700">Manage Your Data with Ease</h2>
            <p class="text-gray-500 mt-2 max-w-lg text-center">
                The LanceDB Viewer provides you with a powerful and intuitive interface to view, manage, and analyze your data. Start exploring today!
            </p>

            <div class="mt-6">
                <a href="/viewer" class="bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-lg shadow-md font-medium">
                    Launch Viewer
                </a>
            </div>
        </main>

        <footer class="mt-auto bg-gray-800 w-full py-4">
            <p class="text-center text-gray-400 text-sm">&copy; 2025 LanceDB Viewer | Built with ❤️ and Tailwind CSS</p>
        </footer>
    </div>
</body>
</html>
'''

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return HTMLResponse(content=html_template, status_code=200)

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)