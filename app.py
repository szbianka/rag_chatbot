### RAG UI ###

# Import libraries 
import streamlit as st
from rag_chatbot import rag_pipeline

# Set app's title
st.title("PDF Assistant :memo:")

# Initialize session state to store conversation history and user input
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""  # For clearing the input box later

# Placeholder for dropping multiple PDF files
uploaded_files = st.file_uploader("Upload up to 5 PDFs", type="pdf", accept_multiple_files=True)

# Process the uploaded PDFs and handle the user query
if uploaded_files is not None and len(uploaded_files) > 0:
    if len(uploaded_files) > 5:
        st.warning("You can only upload up to 5 PDFs.")
    else:
        with st.spinner('Processing the PDFs...'):
            pdf_paths = []
            for uploaded_file in uploaded_files:
                pdf_path = f"uploaded_{uploaded_file.name}"
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                pdf_paths.append(pdf_path)

            st.success("PDFs processed. You can now ask questions.")

# Define function to display chat history
def display_chat_history():
    for user_query, bot_response in st.session_state.chat_history:
        st.markdown(f"""
        <div style="background-color:#b3e4f8;">
        <strong>👩‍💻Human:</strong> {user_query}
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background-color:#faf7f6;">
        <strong>🤖 Bot:</strong> {bot_response}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("----")  # Divider between conversations

# Display chat history ( above the input box)
display_chat_history()

# Setting up chat input box ans send buttons
st.markdown("### :left_speech_bubble: Chat")
st.text("Ask a question about the document:")
# Create a form to group the input box and the button together
with st.form(key="input_form", clear_on_submit=True):
    # Place the input and the button next to each other
    cols = st.columns([8, 1])
    
    with cols[0]:
        query = st.text_input("Ask a question about the document:", value=st.session_state.user_input, key="input_box", label_visibility="collapsed")

    with cols[1]:
        submit_button = st.form_submit_button("Send")
    
    # Process the form submission (when either the button is clicked or 'Enter' is pressed)
    if submit_button and query:
        if len(uploaded_files) > 0:
            with st.spinner('Processing your query...'):
                response = rag_pipeline(pdf_paths, query)
                st.session_state.chat_history.append((query, response))
                
                # Clear the input field by resetting the session state
                st.session_state.user_input = ""  # Reset session state user_input
        else:
            st.warning("Please upload and process PDFs before asking questions.")

