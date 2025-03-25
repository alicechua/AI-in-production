import gradio as gr
import requests
import os

BACKEND_URL = os.getenv('BACKEND_URL', 'http://backend:8000/upload/')

def upload_file(file):
    with open(file.name, 'rb') as f:
        response = requests.post(BACKEND_URL, files={'file': f})
    
    result = response.json()
    
    return result.get('summary', 'Not available'), \
           result.get('key_takeaways', []), \
           result.get('abstract', 'Not available'), \
           result.get('keywords', []), \
           result.get('methodology', 'Not available'), \
           result.get('references', [])

iface = gr.Interface(
    fn=upload_file,
    inputs=gr.File(),
    outputs=[
        gr.Textbox(label="Summary", lines=5),
        gr.Textbox(label="Key Takeaways", lines=5),
        gr.Textbox(label="Abstract", lines=5),
        gr.Textbox(label="Keywords", lines=5),
        gr.Textbox(label="Methodology", lines=5),
        gr.Textbox(label="References", lines=5),
    ]
)

iface.launch()
