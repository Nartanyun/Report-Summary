from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import pandas as pd
import requests
from io import BytesIO
import pdfplumber
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    summary = None
    filename = None
    question = None
    if request.method == 'POST':
        file = request.files['file']
        question = request.form.get('question', '').strip()
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            filename = file.filename
            try:
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                elif file.filename.endswith('.pdf'):
                    with pdfplumber.open(filepath) as pdf:
                        pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
                        # For Tapas, we need a DataFrame, so PDF is not supported for Q&A
                        summary = 'PDF Q&A not supported. Please upload CSV or Excel.'
                        return render_template('index.html', summary=summary, answer=answer, filename=filename, question=question)
                else:
                    df = pd.read_excel(filepath)
                if question:
                    # Prepare Tapas API call
                    api_url = "https://api-inference.huggingface.co/models/google/tapas-large-finetuned-wtq"
                    hf_token = os.environ.get("HUGGINGFACE_API_TOKEN")
                    headers = {"Accept": "application/json"}
                    if hf_token:
                        headers["Authorization"] = f"Bearer {hf_token}"
                    # Convert DataFrame to records for Tapas
                    table = df.head(20).to_dict(orient='records')
                    payload = {"inputs": {"query": question, "table": table}}
                    response = requests.post(api_url, headers=headers, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, dict) and 'answer' in result:
                            answer = result['answer']
                        else:
                            answer = str(result)
                    else:
                        answer = f"Hugging Face API error: {response.text}"
                else:
                    summary = "Please enter a question for table Q&A."
            except Exception as e:
                summary = f"Error processing file: {e}"
    return render_template('index.html', summary=summary, answer=answer, filename=filename, question=question)

@app.route('/download')
def download():
    summary = request.args.get('summary', '')
    return send_file(BytesIO(summary.encode()), as_attachment=True, download_name='summary.txt', mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)



