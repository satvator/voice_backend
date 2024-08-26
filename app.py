from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from langdetect import detect
from googletrans import Translator
from collections import Counter
import re
import speech_recognition as sr
from pydub import AudioSegment
import os
import io
import spacy
import pytz
from datetime import datetime

app = Flask(__name__)
CORS(app)

# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///voice_analyzer.db'
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///voice_analyzer.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

recognizer = sr.Recognizer()
translator = Translator()

# Load spaCy models
nlp = spacy.load("en_core_web_sm")
nlp_md = spacy.load("en_core_web_md")

class Transcription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50))
    original_text = db.Column(db.Text)
    translated_text = db.Column(db.Text)
    language = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

class WordFrequency(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50))
    word = db.Column(db.String(50))
    frequency = db.Column(db.Integer)

class Phrase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50))
    phrase = db.Column(db.String(255))
    frequency = db.Column(db.Integer)

# Creating the database tables within the application context
with app.app_context():
    db.create_all()

@app.route('/', methods=['GET'])
def home():
    return "Hello from Voice Analyzer"

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.form
    user_id = data.get('user_id')
    audio_file = request.files.get('audio')

    if not audio_file:
        return jsonify({'status': 'error', 'message': 'No audio file provided.'}), 400

    # Converting audio to text
    try:
        # Convert audio to WAV format if necessary
        if audio_file.filename.endswith('.mp3'):
            audio = AudioSegment.from_mp3(audio_file)
            audio = audio.set_channels(1).set_frame_rate(16000)
            with io.BytesIO() as buffer:
                audio.export(buffer, format="wav")
                buffer.seek(0)
                audio_file = io.BytesIO(buffer.read())
        elif audio_file.mimetype != 'audio/wav':
            audio = AudioSegment.from_file(audio_file)
            wav_io = io.BytesIO()
            audio.export(wav_io, format='wav')
            wav_io.seek(0)
            audio_file = wav_io

        # Perform speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        
        original_text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return jsonify({'status': 'error', 'message': 'Could not understand audio.'}), 400
    except sr.RequestError:
        return jsonify({'status': 'error', 'message': 'Could not request results from Google Speech Recognition service.'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error processing audio file: {e}'}), 500

    # Detecting language
    try:
        detection=translator.detect(original_text)
        language=detection.lang
        if language != 'en':
            translated_text = translator.translate(original_text, src=language, dest='en').text
        else:
            translated_text = original_text
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error translating text: {e}'}), 500
    
    # Save transcription data to the database
    try:
        transcription = Transcription(user_id=user_id, original_text=original_text, translated_text=translated_text, language=language)
        db.session.add(transcription)
        db.session.commit()
        
        update_word_frequencies(user_id, translated_text)
        update_phrases(user_id, translated_text)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error saving transcription data: {e}'}), 500

    return jsonify({'status': 'success', 'translated_text': translated_text})

def update_word_frequencies(user_id, text):
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    
    for word, count in word_counts.items():
        existing = WordFrequency.query.filter_by(user_id=user_id, word=word).first()
        if existing:
            existing.frequency += count
        else:
            new_entry = WordFrequency(user_id=user_id, word=word, frequency=count)
            db.session.add(new_entry)
    db.session.commit()

def update_phrases(user_id, text):
    doc = nlp(text)
    phrases = []
    
    for np in doc.noun_chunks:
        phrase = np.text.lower().strip()
        if len(phrase.split()) <= 6:  # Limiting to 6-word phrases
            phrases.append(phrase)
    
    phrase_counts = Counter(phrases)
    
    for phrase, count in phrase_counts.items():
        existing = Phrase.query.filter_by(user_id=user_id, phrase=phrase).first()
        if existing:
            existing.frequency += count
        else:
            new_entry = Phrase(user_id=user_id, phrase=phrase, frequency=count)
            db.session.add(new_entry)
    db.session.commit()

@app.route('/transcribe_live', methods=['POST'])
def transcribe_live():
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    user_id = request.form.get('user_id')  # Ensure user_id is provided in the form data
    if not user_id:
        return jsonify({'status': 'error', 'message': 'No user_id provided'}), 400

    try:
        if audio_file.mimetype != 'audio/wav':
            audio = AudioSegment.from_file(audio_file)
            wav_io = io.BytesIO()
            audio.export(wav_io, format='wav')
            wav_io.seek(0)
            audio_file = wav_io

        recognizer = sr.Recognizer()
        audio_data = sr.AudioFile(audio_file)
        with audio_data as source:
            audio = recognizer.record(source)
        
        original_text = recognizer.recognize_google(audio)
        
        if not original_text.strip():
            return jsonify({'status': 'error', 'message': 'No speech detected in the audio file'}), 400
        
        detection = translator.detect(original_text)
        detected_language = detection.lang
        
        if detected_language != 'en':
            translated_text = translator.translate(original_text, src=detected_language, dest='en').text
        else:
            translated_text = original_text
        
        transcription = Transcription(user_id=user_id, original_text=original_text, translated_text=translated_text, language=detected_language)
        db.session.add(transcription)
        db.session.commit()
        
        update_word_frequencies(user_id, translated_text)
        update_phrases(user_id, translated_text)
        
        return jsonify({'status': 'success', 'transcription': translated_text, 'translated_text': translated_text})

    except sr.UnknownValueError:
        return jsonify({'status': 'error', 'message': 'Could not understand audio. The audio may be too unclear.'}), 400
    except sr.RequestError as e:
        return jsonify({'status': 'error', 'message': f'Could not request results from Google Speech Recognition service; {e}'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error processing audio file: {e}'}), 500

@app.route('/history/<user_id>', methods=['GET'])
def get_history(user_id):
    transcriptions = Transcription.query.filter_by(user_id=user_id).all()
    if not transcriptions:
        return jsonify({'error': 'No data found for this user.'}), 404
    
    ist = pytz.timezone('Asia/Kolkata')  # Define IST timezone
    result = []
    for t in transcriptions:
        # Convert timestamp to IST and format to only include date
        timestamp_ist = t.timestamp.astimezone(ist).strftime('%Y-%m-%d')
        result.append({
            'id': t.id,
            'original_text': t.original_text,
            'translated_text': t.translated_text,
            'language': t.language,
            'timestamp': timestamp_ist
        })
    
    return jsonify(result)

@app.route('/frequencies/<user_id>', methods=['GET'])
def get_word_frequencies(user_id):
    frequencies = WordFrequency.query.filter_by(user_id=user_id).all()
    if not frequencies:
        return jsonify({'error': 'No data found for this user.'}), 404
    return jsonify([{
        'word': f.word,
        'frequency': f.frequency
    } for f in frequencies])

@app.route('/phrase-frequencies/<user_id>', methods=['GET'])
def get_phrase_frequencies(user_id):
    phrases = Phrase.query.filter_by(user_id=user_id).all()
    if not phrases:
        return jsonify({'error': 'No data found for this user.'}), 404
    return jsonify([{
        'phrase': p.phrase,
        'frequency': p.frequency
    } for p in phrases])

@app.route('/comparison/<user_id>', methods=['GET'])
def compare_word_frequencies(user_id):
    user_frequencies = WordFrequency.query.filter_by(user_id=user_id).all()
    if not user_frequencies:
        return jsonify({'error': 'No data found for this user.'}), 404

    # Creating a dictionary for the user's word frequencies
    user_word_counts = {f.word: f.frequency for f in user_frequencies}

    # Getting all word frequencies across all users and aggregate them
    all_word_counts = Counter()
    all_user_frequencies = WordFrequency.query.all()
    for entry in all_user_frequencies:
        all_word_counts[entry.word] += entry.frequency

    # Preparing the comparison results
    comparison = {}
    for word, count in user_word_counts.items():
        comparison[word] = {
            'user_frequency': count,
            'all_users_frequency': all_word_counts.get(word, 0)
        }
    
    return jsonify(comparison)

def compute_similarity(text1, text2):
    doc1 = nlp_md(text1)
    doc2 = nlp_md(text2)
    return doc1.similarity(doc2)

@app.route('/compare_similarity/<user_id>', methods=['GET'])
def compare_similarity(user_id):
    current_user_transcriptions = Transcription.query.filter_by(user_id=user_id).all()
    all_users_transcriptions = Transcription.query.filter(Transcription.user_id != user_id).all()
    
    if not current_user_transcriptions:
        return jsonify({'error': 'No data found for this user.'}), 404

    similarity_scores = {}
    for other_transcription in all_users_transcriptions:
        for current_transcription in current_user_transcriptions:
            score = compute_similarity(current_transcription.translated_text, other_transcription.translated_text)
            if other_transcription.user_id not in similarity_scores:
                similarity_scores[other_transcription.user_id] = score
            else:
                similarity_scores[other_transcription.user_id] = max(similarity_scores[other_transcription.user_id], score)

    # Sort by similarity score and keep the top N most similar users (e.g., top 5)
    top_n = 5
    sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    return jsonify([{'user_id': user_id, 'score': score} for user_id, score in sorted_scores])

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'status': 'error', 'message': 'Internal server error.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('PORT', '0.0.0.0')
    app.run(host=host, port=port)
