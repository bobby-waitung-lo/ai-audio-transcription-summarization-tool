import streamlit as st
import whisper
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_huggingface import HuggingFacePipeline
import os

# Set page configuration for better accessibility
st.set_page_config(page_title="AI Audio Transcription and Summarization", layout="wide")

# Load models (initialize outside of functions to avoid reloading)
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt", device=-1)
    hf_pipeline = HuggingFacePipeline(pipeline=summarizer)
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in 30-200 words:\n\n{text}"
    )
    summary_chain = RunnableSequence(prompt | hf_pipeline)
    return whisper_model, summary_chain

whisper_model, summary_chain = load_models()

# Transcription function
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# Combined function to process audio
def process_audio(audio_path):
    transcription = transcribe_audio(audio_path)
    summary = summary_chain.invoke({"text": transcription})
    return transcription, summary

# Streamlit interface
st.title("AI-Powered Audio Transcription and Summarization")
st.markdown("Upload an audio file (MP3 or WAV) to transcribe and summarize it for media professionals.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"], help="Supported formats: MP3, WAV")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_audio_path = "temp_audio.mp3"
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Show a loading spinner while processing
    with st.spinner("Processing audio... This may take a moment."):
        try:
            # transcription, summary = process_audio(temp_audio_path)
            transcription, summary = process_audio("temp_audio.mp3")
            
            # Display results
            st.subheader("Transcription")
            st.text_area("Full Transcription", transcription, height=200, help="The complete text from the audio.")
            
            st.subheader("Summary")
            st.text_area("Summary", summary, height=100, help="A concise summary of the transcribed text.")
            
            # Download buttons for accessibility and usability
            st.download_button(
                label="Download Transcription",
                data=transcription,
                file_name="transcription.txt",
                mime="text/plain",
                help="Download the transcription as a text file."
            )
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain",
                help="Download the summary as a text file."
            )
            
            # Clean up temporary file
            os.remove(temp_audio_path)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try uploading a different file or check your setup.")
else:
    st.info("Please upload an audio file to begin.")

# Footer for context
st.markdown("---")
st.markdown("Built for media professionals to transcribe and summarize audio content efficiently. Powered by Whisper and BART.")