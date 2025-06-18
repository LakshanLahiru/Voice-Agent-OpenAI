from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
import streamlit as st

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

def get_answer(messages: list) -> str:
    """Get complete answer from OpenAI"""
    system_message = [{'role': 'system', 'content': "You are a helpful AI chatbot that answers questions asked by User. Keep your responses conversational and engaging."}]
    messages = system_message + messages
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1000,
        temperature=0.7
    )
    return response.choices[0].message.content

def get_answer_stream(messages: list) -> str:
    """Get streaming answer from OpenAI (simulated for now)"""
    # For now, we'll get the complete response and return it
    # In a real implementation, you would use the streaming API
    system_message = [{'role': 'system', 'content': "You are a helpful AI chatbot that answers questions asked by User. Keep your responses conversational and engaging."}]
    messages = system_message + messages
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

def get_answer_true_stream(messages: list):
    """Get true streaming answer from OpenAI (generator)"""
    system_message = [{'role': 'system', 'content': "You are a helpful AI chatbot that answers questions asked by User. Keep your responses conversational and engaging."}]
    messages = system_message + messages
    
    try:
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"I apologize, but I encountered an error: {str(e)}"

def speech_to_text(audio_data):
    """Convert speech to text using OpenAI Whisper"""
    try:
        with open(audio_data, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                response_format='text',
                file=audio_file
            )
        return transcript
    except Exception as e:
        st.error(f"Speech to text failed: {str(e)}")
        return None

def text_to_speech(input_text):
    """Convert text to speech using OpenAI TTS"""
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=input_text
        )
        web_file_path = 'temp_audio_play.mp3'
        with open(web_file_path, 'wb') as f:
            response.stream_to_file(web_file_path)
        return web_file_path
    except Exception as e:
        st.error(f"Text to speech failed: {str(e)}")
        return None

def autoplay_audio(file_path: str):
    """Autoplay audio in Streamlit"""
    try:
        if file_path and os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = f.read()
            b64 = base64.b64encode(data).decode('utf-8')
            md = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
            st.markdown(md, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Audio playback failed: {str(e)}")