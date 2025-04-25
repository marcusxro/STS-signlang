import pyttsx3
import time

def test_tts_basic():
    """Test basic text-to-speech functionality"""
    print("Testing basic text-to-speech...")
    try:
        engine = pyttsx3.init()
        print("Engine initialized successfully")
        

        print("Saying 'Hello World'...")
        engine.say("Hello World")
        engine.runAndWait()
        print("Speech completed")
        
        return True
    except Exception as e:
        print(f"Error in basic test: {e}")
        return False

def test_tts_properties():
    """Test TTS properties and voices"""
    print("\nTesting TTS properties...")
    try:
        engine = pyttsx3.init()

        voices = engine.getProperty('voices')
        print(f"Available voices ({len(voices)}):")
        for i, voice in enumerate(voices):
            print(f"{i+1}. ID: {voice.id}")
            print(f"   Name: {voice.name}")
            print(f"   Languages: {voice.languages}")

        if len(voices) > 0:
            print("\nTesting voice switching...")
            engine.setProperty('voice', voices[0].id)
            engine.say(f"Testing voice {voices[0].name}")
            engine.runAndWait()
        

        print("\nTesting speech rate...")
        rates = [100, 150, 200]
        for rate in rates:
            engine.setProperty('rate', rate)
            engine.say(f"Rate set to {rate}")
            print(f" - Speaking at rate {rate}...")
            engine.runAndWait()
        
        return True
    except Exception as e:
        print(f"Error in properties test: {e}")
        return False

def test_tts_threading():
    """Test TTS in a threaded environment"""
    print("\nTesting threaded TTS...")
    try:
        engine = pyttsx3.init()
        
        def speak_in_thread(text):
            try:
                print(f"Thread saying: {text}")
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"Thread error: {e}")
        
        import threading
        threads = []
        phrases = ["Thread one speaking", "Thread two speaking", "Thread three speaking"]
        
        for i, phrase in enumerate(phrases):
            t = threading.Thread(target=speak_in_thread, args=(phrase,))
            threads.append(t)
            t.start()
            time.sleep(0.5)  

        for t in threads:
            t.join()
        
        print("All threads completed")
        return True
    except Exception as e:
        print(f"Error in threading test: {e}")
        return False

def main():
    print("Starting Text-to-Speech Tests\n")

    basic_result = test_tts_basic()
    props_result = test_tts_properties()
    thread_result = test_tts_threading()

    print("\nTest Summary:")
    print(f"- Basic Test: {'PASSED' if basic_result else 'FAILED'}")
    print(f"- Properties Test: {'PASSED' if props_result else 'FAILED'}")
    print(f"- Threading Test: {'PASSED' if thread_result else 'FAILED'}")
    
    if basic_result and props_result and thread_result:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. Check output for errors.")

if __name__ == "__main__":
    main()