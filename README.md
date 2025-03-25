# AI-in-production
This repo is to store the project submission for the AI in Production course by Red Dragon AI.

# Document Analysis Web Application

This project is a web application designed to perform document analysis using FastAPI for the backend and Gradio for the frontend. It allows users to upload research papers (PDFs) and receive an analysis, including a summary, key takeaways, abstract extraction, keywords identification, methodology breakdown, and references analysis.

## Project Structure

```
/project-root
├── app/                                     # Backend directory containing FastAPI app
│   ├── app.py                               # FastAPI app code
│   ├── Dockerfile                           # Dockerfile for backend
│   ├── requirements.txt                     # Backend dependencies
│   └── models/                              # LLM Models used
│      ├── summarization_create_onnx.py      # File to pull BART and create ONNX model
│      └── summarization_test_onnx           # File to run and test BART ONNX model
│   └── frontend/                            # Frontend directory containing Gradio app
│      ├── gradio_interface.py               # Gradio interface code
│      ├── requirements.txt                  # Frontend dependencies
│      └── Dockerfile                        # Dockerfile for frontend
├── archive/                                 # Research on Agents
├── docker-compose.yml                       # Docker Compose configuration
└── README.md                                # This file
```

## Requirements

- Python 3.9 or higher
- Docker (optional but recommended for containerization)

### Backend (FastAPI)
- **Dependencies**: FastAPI, Uvicorn, and other necessary Python packages.
- **Port**: 8000 (FastAPI backend)

### Frontend (Gradio)
- **Dependencies**: Gradio for building the web interface, Requests to communicate with the backend.
- **Port**: 7860 (Gradio frontend)

## Getting Started

### 1. Clone the Repository

First, clone the project to your local machine:

```bash
git clone https://github.com/your-username/document-analysis-webapp.git
cd document-analysis-webapp
```

### 2. Install Dependencies

If you are running the project **without Docker**, you can install the required dependencies for both backend and frontend manually:

#### Backend Dependencies:

1. Navigate to the `app` directory:
   ```bash
   cd app
   ```

2. Install the backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

#### Frontend Dependencies:

1. Navigate to the `frontend` directory:
   ```bash
   cd ../frontend
   ```

2. Install the frontend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Run the Backend

To run the backend using FastAPI and Uvicorn, navigate to the `app` directory and run the following:

```bash
cd app
uvicorn app.app:app --host 0.0.0.0 --port 8000
```

This will start the FastAPI backend on `http://127.0.0.1:8000`.

### 4. Run the Frontend

To run the frontend, navigate to the `frontend` directory and run the Gradio app:

```bash
cd ../frontend
python gradio_interface.py
```

This will start the Gradio frontend on `http://127.0.0.1:7860`.

### 5. Docker (Optional)

You can use Docker to run both the backend and frontend in containers. This is especially useful if you want to avoid dependency issues.

#### Build and Run with Docker Compose:

Make sure Docker and Docker Compose are installed. Then, in the root of the project directory, run the following:

```bash
docker-compose up --build
```

This will build the Docker images for both the backend and frontend and start the containers. The application will be available at:

- **Backend**: `http://127.0.0.1:8000`
- **Frontend**: `http://127.0.0.1:7860`

#### Stop the Docker Containers:

To stop the running containers, use:

```bash
docker-compose down
```

## API Documentation

### POST `/upload/`

- **Description**: Upload a PDF file and get an analysis of the document.
- **Request**:
  - `file` (required): The PDF file to be analyzed.

- **Response**:
  - `summary`: A summary of the document.
  - `key_takeaways`: Key takeaways from the document.
  - `abstract`: Extracted abstract of the document.
  - `keywords`: Keywords identified from the document.
  - `methodology`: Extracted methodology from the document.
  - `references`: Extracted references from the document.

Example request using `curl`:

```bash
curl -X 'POST'   'http://127.0.0.1:8000/upload/'   -H 'accept: application/json'   -H 'Content-Type: multipart/form-data'   -F 'file=@path/to/your/file.pdf'
```

## Notes

- Make sure the `requirements.txt` files are up to date in both the frontend and backend directories.
- If you're using Docker, make sure that both services (frontend and backend) are correctly linked in the `docker-compose.yml` file.
- You can modify the `Dockerfile` for each service to include additional dependencies if needed.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License

This project is licensed under the MIT License.
