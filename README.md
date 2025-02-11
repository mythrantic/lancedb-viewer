# LanceDB Viewer

## About

LanceDB Viewer is a user-friendly tool designed to interact with LanceDB, offering a visual interface for performing Create, Read, Update, and Delete (CRUD) operations. This project aims to simplify database management and enhance productivity.

## Features

-   **Visual Interface**: Easily browse and manage LanceDB databases.
-   **CRUD Support**: Perform Create, Read, Update, and Delete operations seamlessly.
-   **Developer-Friendly**: Designed for both end-users and contributors.

## Tech Stack

-   **Backend**: Python (FastAPI)
-   **Frontend**: SvelteKit
-   **Database**: LanceDB

---

## Getting Started

### Prerequisites

Ensure you have the following installed on your system:

-   **Node.js** (v16 or higher)
-   **Python** (v3.8 or higher)
-   **npm** (comes with Node.js)

---

### Running the Application

To start the application in development mode:

1.  Clone the repository:

    ```bash
    git clone https://github.com/valiantlynx/lancedb-viewer.git
    cd lancedb-viewer
    ```
2.  Start the backend server:

    ```bash
    python src/main.py
    ```
3.  Start the frontend development server:

    ```bash
    cd frontend
    npm install
    npm run dev
    ```
4.  Open your browser and navigate to [http://localhost:5173](http://localhost:5173).

---

## Contributing

If you'd like to contribute:

1.  Set up a development environment using Docker Compose:

    ```bash
    docker-compose up --build
    ```

    This will set up a complete development environment for testing and contributing.

    or do it however you like
3.  Make your changes and submit a Pull Request.

---

## License

This project is open source. Check the repository for license details.

---

## Contact

For questions or suggestions, please open an issue in the GitHub repository.
