from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import pandas as pd
import requests
import pdfplumber
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = None
    filename = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            filename = file.filename
            try:
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                    data_str = df.head(20).to_string()
                elif file.filename.endswith('.pdf'):
                    with pdfplumber.open(filepath) as pdf:
                        pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
                        data_str = '\n'.join(pages)[:3000] if pages else 'No extractable text found in PDF.'
                else:
                    df = pd.read_excel(filepath)
                    data_str = df.head(20).to_string()
                # Use Hugging Face summarization API with a different public model
                api_url = "https://api-inference.huggingface.co/models/google/pegasus-xsum"
                headers = {"Accept": "application/json"}
                payload = {"inputs": f"Summarize this report for business stakeholders: {data_str}"}
                response = requests.post(api_url, headers=headers, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and 'summary_text' in result[0]:
                        summary = result[0]['summary_text']
                    elif isinstance(result, dict) and result.get("error"):
                        summary = f"Hugging Face API error: {result['error']}"
                    elif isinstance(result, dict) and result.get("estimated_time"):
                        summary = "Model is loading on Hugging Face. Please wait a few seconds and try again."
                    else:
                        summary = str(result)
                else:
                    summary = f"Hugging Face API error: {response.text}"
            except Exception as e:
                summary = f"Error processing file: {e}"
    return render_template('index.html', summary=summary, filename=filename)

@app.route('/download')
def download():
    summary = request.args.get('summary', '')
    return send_file(BytesIO(summary.encode()), as_attachment=True, download_name='summary.txt', mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)


