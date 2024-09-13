# RAG chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot that can process up to 5 PDF files and generate answers based on their content.
The model used in the project is Microsoft's Phi-1.5 (see: https://huggingface.co/microsoft/phi-1_5), however, it can be flexibly changed when using the ```rag_pipeline``` function in the ```rag_chatbot.py``` script.

### Features
- Accepts up to 5 PDF files as input.
- Uses Retrieval-Augmented Generation (RAG) to generate answers based on PDF content.
- Streamlit web interface for a user-friendly chat experience.
- Supports conversational memory for ongoing context.

### Project setup
Follow the steps below to set up and run the project.

#### 1. Clone the repository in terminal
  - ``` git clone https://github.com/szbianka/rag_chatbot.git ```
#### 2. Change directory to folder
  - ```cd rag_chatbot```
#### 3. Run the setup script
  - Run the ```setup.sh``` script, which:
    - Creates a virtual environment
    - Installs the required dependencies from requirements.txt
  - ```bash setup.sh```
#### 4. Run the Application
  - activate in the virtual environment
    - by ``` source chatbot_venv/bin/activate ``` on Mac or in a bash terminal
    - or ``` chatbot_venv\Scripts\Activate.ps1``` on Windows
  - Launch the chatbot using Streamlit, which will start a local server where you can interact with the chatbot by running ```streamlit run app.py```

### How to Use
Upload up to 5 PDF files using the file uploader.
Ask questions related to the uploaded PDFs in the chat box.
The chatbot will retrieve relevant information from the PDFs and respond accordingly.
