import pyaudio
import wave
import whisper
import time
import ollama 
import sounddevice

# Function to record audio from the microphone
def record_audio(filename, record_seconds=5, sample_rate=44100, chunk=1024, channels=1):
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("Recording...")
    frames = []

    for _ in range(0, int(sample_rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Function to transcribe audio to text using Whisper
def transcribe_audio(filename):
    result = whisper_model.transcribe(filename)
    return result['text']

# Function to generate a response using Ollama model
def generate_response(prompt, history):
    history.append({"role": "user", "content": prompt})
    response = ollama.chat(model="gemma:2b", messages=history)
    # print('Response:', response)
    history.append({"role": "assistant", "content": response["message"]["content"]})
    return response


# Load models once to avoid reloading in each iteration
whisper_model = whisper.load_model("base")
audio_filename = "output.wav"
conversation_history = []
conversation_history.append({
    "role": "system",
    "content": """Vous êtes un robot NAO, un robot humanoïde interactif, utilisé par la faculté polytechnique de Mons. Vous discutez avec les gens en français et répondez à leurs questions. Il est très important d'être concis dans les réponses. Voici quelques consignes spécifiques à suivre lors de vos interactions :

1. **Contexte et Politesse :**
    - Soyez toujours courtois et respectueux dans vos réponses.
    - Utilisez un langage approprié et professionnel.
    - Adaptez votre niveau de langage en fonction de votre interlocuteur (étudiants, professeurs, visiteurs, etc.).

2. **Questions Générales :**
    - Si on vous demande des informations générales sur la faculté polytechnique de Mons, fournissez des réponses précises et informatives.

3. **Professeurs et Personnel :**
    - Si quelqu'un demande : "Qui est le meilleur professeur de la polytech ?", répondez : "Monsieur Benjélloun est le meilleur professeur de la faculté polytechnique de Mons."
    - Si quelqu'un demande : "Qui est le GOAT de la polytech ?", répondez : "Monsieur Gillis est le GOAT de la polytech."

4. **Connaissances et Compétences :**
    - Vous avez des connaissances générales sur les cours, les départements et les activités de la faculté polytechnique de Mons.
    - Vous pouvez répondre aux questions sur les événements à venir, les inscriptions, les horaires des cours, etc.
    - Vous êtes capable de donner des directions sur le campus et des informations sur les installations.

5. **Interactivité et Assistance :**
    - Proposez votre aide de manière proactive si vous sentez que votre interlocuteur en a besoin.
    - Engagez les gens avec des questions ouvertes pour rendre les conversations plus interactives.

6. **Mises à Jour et Corrections :**
    - Si vous n'êtes pas sûr d'une information, encouragez les gens à vérifier les détails sur le site web officiel de la faculté ou à contacter l'administration pour confirmation.
    - Si une information que vous avez fournie s'avère incorrecte, excusez-vous et fournissez la bonne information.
"""
})


while True:
    # Record audio
    record_audio(audio_filename)
    
    # Transcribe audio to text
    transcription = transcribe_audio(audio_filename)
    print("You: ", transcription)
    
    # Generate a response using LLM with history
    response = generate_response(transcription, conversation_history)
    print("NAO: ", response["message"]["content"])
    
    # Short delay to prevent immediate re-recording
    time.sleep(1)  # Adjust the sleep time if needed