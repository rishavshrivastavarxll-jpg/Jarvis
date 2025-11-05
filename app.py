# app.py
import os
import requests
import wikipedia
import subprocess
import speech_recognition as sr
import datetime
import random
import webbrowser
import io
import uuid
import tempfile
import logging
import importlib.util
from collections import deque
from flask import Flask, render_template, request, jsonify, send_file

from gtts import gTTS

# ---------------- Configuration ----------------
app = Flask(__name__)

# Replace with your OpenWeatherMap API key
WEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"

# Folder where your local videos are stored (update to your path)
MUS_DIR = r"D:\Video"

# Temp directory for storing uploaded / converted audio
TEMP_DIR = tempfile.gettempdir()
STOPFILE_PATH = os.path.join(TEMP_DIR, "jervis_stop.flag")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------------- Runtime controls & context ----------------
# Event-like flag to control an in-process listener (if you run one here)
import threading
listener_running = threading.Event()
listener_running.set()  # initially allow listener to run

# Short-term conversation context (in-memory)
CONTEXT_WINDOW = 8
conversation_context = deque(maxlen=CONTEXT_WINDOW)

def push_context(role, text):
    """Push a turn into short-term memory (role: 'user' or 'assistant')."""
    if not text:
        return
    conversation_context.append({"role": role, "text": text})

def get_context_text():
    """Return a compact textual version of recent conversation for optional use."""
    return " | ".join(f"{c['role']}: {c['text']}" for c in conversation_context)

# ---------------- Skill loader ----------------
SKILLS = []
SKILLS_DIR = "skills"

def load_skills(skills_dir=SKILLS_DIR):
    SKILLS.clear()
    if not os.path.isdir(skills_dir):
        logging.info("No skills directory found at %s", skills_dir)
        return
    for fn in sorted(os.listdir(skills_dir)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        path = os.path.join(skills_dir, fn)
        name = fn[:-3]
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception:
            logging.exception("Failed to import skill module %s", fn)
            continue

        # Require API: can_handle and handle
        if not (hasattr(mod, "can_handle") and callable(getattr(mod, "can_handle"))):
            logging.warning("Skill %s skipped: missing 'can_handle'", name)
            continue
        if not (hasattr(mod, "handle") and callable(getattr(mod, "handle"))):
            logging.warning("Skill %s skipped: missing 'handle'", name)
            continue

        SKILLS.append(mod)
        logging.info("Loaded skill: %s", name)

# Load skills at startup
load_skills()

# ---------------- Helpful constants ----------------
FALLBACK_RESPONSES = [
    "I'm not sure about that yet — but I can try to look it up if you want.",
    "Hmm, I don't have an answer for that right now.",
    "Good question — I don't know the answer yet. Want me to search the web for it?",
    "I couldn't find a match in my skills. I can try searching the web if you'd like.",
    "That one's outside my current abilities. I can attempt a web lookup if you say 'search web for ...'.",
    "I don't know the answer, but I can help you find it — try asking 'search web for [topic]'.",
    "I couldn't recognise that command. You can ask me things like 'search for [topic]' or say 'help'."
]

# ---------------- Helper functions ----------------

def wish():
    """Generates the initial greeting text."""
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        greeting = "Good Morning Rishav!"
    elif 12 <= hour < 18:
        greeting = "Good Afternoon Rishav!"
    else:
        greeting = "Good Evening Rishav!"
    return f"{greeting} I am your assistant. How may I help you?"

def get_weather_report(city):
    """Fetches real-time weather using OpenWeatherMap API."""
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={WEATHER_API_KEY}&q={requests.utils.requote_uri(city)}&units=metric"
    try:
        response = requests.get(complete_url, timeout=8)
        data = response.json()
        if data.get("cod") == 200:
            main = data['main']
            weather = data['weather'][0]
            temperature_c = round(main['temp'], 1)
            report = (
                f"The weather in {city.title()} is currently {weather['description']} "
                f"with a temperature of {temperature_c} degrees Celsius. Humidity is {main['humidity']} percent."
            )
            return report
        else:
            message = data.get("message", "")
            return f"Sorry, I couldn't find the weather for {city}. {message}"
    except requests.exceptions.RequestException:
        return "I had trouble connecting to the weather service. Please check your network connection."

def safe_join_after_keyword(query, keyword_list):
    """
    For 'weather in <city>' and similar: return the remainder after the first found keyword.
    This handles multi-word city names (e.g., 'new york').
    """
    query_lower = query.lower()
    for kw in keyword_list:
        if kw in query_lower:
            parts = query_lower.split(kw, 1)[1].strip()
            return parts
    return ""

def convert_to_wav_if_needed(input_path):
    """
    If input_path is already a wav file, returns it.
    Otherwise attempts to convert using ffmpeg (if available) to a wav file and returns new path.
    Raises RuntimeError with helpful message if conversion isn't possible.
    """
    _, ext = os.path.splitext(input_path)
    ext = ext.lower()
    if ext == ".wav":
        return input_path

    output_path = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}.wav")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", input_path, output_path
        ], check=True)
        if os.path.exists(output_path):
            return output_path
        else:
            raise RuntimeError("ffmpeg did not produce an output file.")
    except FileNotFoundError:
        raise RuntimeError(
            "Uploaded audio is not WAV and ffmpeg is not installed on the server. "
            "Install ffmpeg or upload a WAV file."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to convert audio: {e}")

def transcribe_audio_file(audio_path):
    """Transcribes a WAV (or convertible) audio file using SpeechRecognition's Google recognizer."""
    recognizer = sr.Recognizer()
    wav_path = None
    try:
        wav_path = convert_to_wav_if_needed(audio_path)
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="en-US")
            return text
    except sr.UnknownValueError:
        raise RuntimeError("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        raise RuntimeError(f"Could not request results from the speech service; {e}")
    finally:
        # If convert created a temporary wav watch for cleanup
        try:
            if wav_path and os.path.exists(wav_path) and wav_path != audio_path and os.path.dirname(wav_path) == TEMP_DIR:
                os.remove(wav_path)
        except Exception:
            pass

# ---------------- Core command processing ----------------

def process_command(query, context=None):
    """Parses the user's text command and executes the appropriate function."""
    if not query:
        return "No command received."

    original_query = query
    query = query.lower().strip()

    # 0. Skill dispatch: ask each loaded skill if it wants to handle this query
    for skill in SKILLS:
        try:
            if skill.can_handle(query):
                logging.info("Dispatching to skill: %s", getattr(skill, "__name__", str(skill)))
                # Allow skills to use context if they want (skill.handle can accept one arg or two)
                try:
                    return skill.handle(query, context) if skill.handle.__code__.co_argcount >= 2 else skill.handle(query)
                except TypeError:
                    return skill.handle(query)
        except Exception:
            logging.exception("Error in skill %s while handling query", getattr(skill, "__name__", str(skill)))
            return "Sorry, a skill errored while processing your request."

    # Built-in triggers
    WIKI_TRIGGERS = ['wikipedia', 'who is', 'what is', 'tell me about', 'search for']

    # 1. WIKIPEDIA SEARCH (fallback if no skill handled)
    if any(trigger in query for trigger in WIKI_TRIGGERS):
        search_term = query
        for trigger in WIKI_TRIGGERS:
            if trigger in search_term:
                search_term = search_term.replace(trigger, "").strip()
                break
        if not search_term or len(search_term) < 2:
            return "Please tell me what subject you would like me to search for."
        try:
            results = wikipedia.summary(search_term, sentences=3)
            return f"According to Wikipedia: {results}"
        except wikipedia.exceptions.PageError:
            return f"Sorry, I could not find a Wikipedia page for {search_term}."
        except wikipedia.exceptions.DisambiguationError as e:
            suggestion = e.options[0] if e.options else "a different query"
            return f"I found multiple results. Try searching for: {suggestion}"
        except Exception as e:
            logging.exception("Wikipedia error")
            return "An error occurred while searching Wikipedia."

    # 2. WEB BROWSER ACTIONS
    elif 'open youtube' in query:
        webbrowser.open("https://www.youtube.com/")
        return "Opening YouTube in your default browser."
    elif 'open google' in query:
        webbrowser.open("https://www.google.com/")
        return "Opening Google in your default browser."

    # 3. LOCAL FILE/APP ACTIONS (Video)
    elif 'video' in query:
        try:
            songs = os.listdir(MUS_DIR)
            video_files = [f for f in songs if not f.startswith('.') and os.path.isfile(os.path.join(MUS_DIR, f))]
            if video_files:
                chosen = random.choice(video_files)
                path_to_file = os.path.join(MUS_DIR, chosen)
                try:
                    os.startfile(path_to_file)  # Windows
                except Exception:
                    # Cross-platform fallback (may not work on Windows)
                    try:
                        subprocess.Popen(["xdg-open", path_to_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except Exception:
                        pass
                return f"Playing a random video from {MUS_DIR}."
            return f"No videos found in the directory: {MUS_DIR}"
        except FileNotFoundError:
            return f"The video directory '{MUS_DIR}' was not found. Please check the path."
        except Exception as e:
            logging.exception("Error playing local video")
            return f"An error occurred while trying to play the video: {e}"

    # 4. TIME
    elif 'time' in query:
        strTime = datetime.datetime.now().strftime("%I:%M:%S %p")
        return f"Sir, The Time is {strTime}"

    # 5. WEATHER REPORT
    elif 'weather' in query:
        city = safe_join_after_keyword(query, ['in', 'of'])
        if not city:
            parts = query.split()
            if len(parts) >= 2:
                city = parts[-1]
        if not city:
            return "Please specify the city for the weather report (e.g., 'weather in New York')."
        return get_weather_report(city)

    # 6. HELP / CAPABILITIES
    elif 'what can you do' in query or 'help' in query or 'commands' in query:
        return (
            "Here are some things I can do for you:\n"
            "1. Search Wikipedia — for example, say 'who is Elon Musk' or 'tell me about Python'.\n"
            "2. Open websites — try 'open YouTube' or 'open Google'.\n"
            "3. Play videos from your local folder.\n"
            "4. Tell the current time — say 'what time is it'.\n"
            "5. Give you weather reports — like 'weather in New York'.\n"
            "6. Respond politely when you say 'thank you' or 'bye'.\n"
            "Ask naturally and I'll do my best!"
        )

    # 7. EXIT/STOP COMMANDS
    elif 'thank you' in query or 'bye' in query or 'stop' in query:
        return "You're welcome! Goodbye, Rishav."

    # 8. FALLBACK / DEFAULT RESPONSE
    else:
        suggestion = "If you'd like, say: \"search for [topic]\" or \"search web for [topic]\"."
        return f"{random.choice(FALLBACK_RESPONSES)} {suggestion}"

# ---------------- Flask Routes ----------------

@app.route("/")
def index():
    initial_greeting = wish()
    return render_template("index.html", initial_greeting=initial_greeting)

@app.route("/generate_audio")
def generate_audio():
    """Generates speech audio using gTTS and streams it to the browser."""
    text = request.args.get('text', 'Hello, I am Jervis and audio is not working.')
    try:
        tts = gTTS(text=text, lang='en')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return send_file(
            mp3_fp,
            mimetype="audio/mp3",
            as_attachment=False,
            download_name="jervis_response.mp3"
        )
    except Exception:
        logging.exception("gTTS error")
        return jsonify({"error": "Text-to-speech failed."}), 500

@app.route("/command", methods=["POST"])
def handle_command():
    """Handles manual text or uploaded audio (FormData)."""
    user_command_text = ""
    jervis_response = ""

    # 1. MANUAL TEXT COMMAND (JSON)
    if request.is_json:
        data = request.get_json()
        user_command_text = (data.get("manual_command") or "").strip()
        if user_command_text:
            logging.info(f"Manual Command Received: {user_command_text}")
            push_context("user", user_command_text)
            jervis_response = process_command(user_command_text, context=get_context_text())
            push_context("assistant", jervis_response)
            return jsonify({
                "command": user_command_text,
                "response": jervis_response
            })
        return jsonify({"response": "No manual command text received."})

    # 2. VOICE COMMAND (FormData with file 'audio')
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"response": "No audio data received in the POST request."})

    # Save uploaded file to a unique temp file
    unique_name = f"{uuid.uuid4().hex}{os.path.splitext(audio_file.filename or '')[1]}"
    temp_input_path = os.path.join(TEMP_DIR, unique_name)
    try:
        audio_file.save(temp_input_path)
        logging.info(f"Saved uploaded audio to {temp_input_path}")

        try:
            # Try transcribing (this will attempt conversion if needed)
            user_command_text = transcribe_audio_file(temp_input_path)
            logging.info(f"Voice Command Transcribed: {user_command_text}")
            push_context("user", user_command_text)
            jervis_response = process_command(user_command_text, context=get_context_text())
            push_context("assistant", jervis_response)
        except RuntimeError as e:
            logging.info("Transcription/runtime error: %s", e)
            jervis_response = str(e)
        except Exception as e:
            logging.exception("Unexpected transcription error")
            jervis_response = f"An unexpected error occurred during transcription: {e}"
    finally:
        # cleanup uploaded file
        try:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
        except Exception:
            pass

    return jsonify({
        "command": user_command_text,
        "response": jervis_response
    })

# ---------------- Listener control endpoints ----------------

@app.route("/shutdown_listener", methods=["POST"])
def shutdown_listener():
    """
    Signal an in-process listener thread to stop.
    The listener should periodically check `listener_running.is_set()` and exit when it's cleared.
    """
    if listener_running.is_set():
        listener_running.clear()
        logging.info("Shutdown requested for in-process listener.")
        return jsonify({"status": "stopping", "message": "Listener stopping."})
    else:
        return jsonify({"status": "stopped", "message": "Listener already stopped."})

@app.route("/stop_external_listener", methods=["POST"])
def stop_external_listener():
    """
    Write a stop file that an external process can watch to shut down gracefully.
    """
    try:
        with open(STOPFILE_PATH, "w", encoding="utf-8") as f:
            f.write("stop\n")
        logging.info("Wrote stop file for external listener at %s", STOPFILE_PATH)
        return jsonify({"status": "stopping", "message": "Stop signal written."})
    except Exception as e:
        logging.exception("Failed to write stop file")
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------------- Run server ----------------
if __name__ == "__main__":
    # reload skills on startup (useful during development)
    load_skills()
    # NOTE: For production, set debug=False and use a production WSGI server
    app.run(host="0.0.0.0", port=5000, debug=True)
