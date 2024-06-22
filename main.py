import ollama 
import speech_recognition as sr
import sounddevice 


recognizer = sr.Recognizer()

def listenandrecognize():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        # text = recognizer.recognize_google(audio)
        text = recognizer.recognize_whisper(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
        return None


# Function to generate a response using Ollama model
def generate_response(prompt, history):
    history.append({"role": "user", "content": prompt})
    response = ollama.chat(model="gemma:2b", messages=history)
    # print('Response:', response)
    history.append({"role": "assistant", "content": response["message"]["content"]})
    return response

def load_system_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
system_prompt = load_system_prompt('system_prompt.txt')
conversation_history = []
conversation_history.append({
    "role": "system",
    "content": system_prompt
})

def main():
    while True:
        # Record audio
        user_input = listenandrecognize()

        # ajout des actions ici, par exemple si danse alors on fait danser le robot

        if user_input:
            # Generate a response using LLM with history
            response = generate_response(user_input, conversation_history)
            print("NAO: ", response["message"]["content"])
            # TTS response using connection to robot 

if __name__ == "__main__":
    main()