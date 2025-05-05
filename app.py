from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from scorer import score_resume, extract_text_from_pdf

app = Flask(__name__)

# Configurations for uploading files
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Ensure the 'uploads' directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('index.html', score=None, error=None)


@app.route('/score', methods=['POST'])
def score():
    if request.method == 'POST':
        resume_file = request.files['resume']
        job_description = request.form['job_description']
        
        # Check if file is uploaded and valid
        if resume_file and allowed_file(resume_file.filename):
            filename = secure_filename(resume_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            resume_file.save(filepath)
            
            # Extract text from PDF
            resume_text = extract_text_from_pdf(filepath)
            
            # Calculate score
            score = score_resume(resume_text, job_description)
            score = round(score, 2)  # Ensure it's a float for comparison
            
            # Generate suggestions based on the score
            if score >= 70:
                suggestion = "Great job! Your resume is well-tailored to the job description."
            elif score >= 40:
                suggestion = "Good start, but consider improving the match by adjusting keywords and experience."
            else:
                suggestion = "Your resume doesn't match well with the job description. Focus on relevant skills, experience, and keywords."
            
            return render_template('index.html', score=score, suggestion=suggestion)
        else:
            return render_template('index.html', error="Invalid file format. Please upload a PDF.")
    
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
