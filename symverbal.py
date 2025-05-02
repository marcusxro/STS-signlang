import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
import threading
import pyttsx3
import queue
import speech_recognition as sr

class ASLTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SymVerbal - ASL Translator")
        self.root.geometry("1200x800")
        
        self.model = pickle.load(open('./model.p', 'rb'))['model']
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        
        self.engine = None
        self.tts_available = self.initialize_tts()
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.speech_recognition_active = False
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video capture")
        
        # GUI variables
        self.current_sign = ""
        self.translation_text = ""
        self.hearing_text = "" 
        self.stable_sign = ""
        self.last_sign_time = 0
        self.sign_stable_time = 1.5 
        self.camera_shot_effect = False
        self.effect_start_time = 0
        self.effect_duration = 0.5  
        self.last_added_sign = None
        self.last_added_time = 0
        self.speech_queue = queue.Queue()
        self.speaking = False
        self.listening = False
        self.current_speaker = None  
        

        self.frame_queue = queue.Queue(maxsize=1)
        
        self.create_widgets()
        
        self.running = True
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.daemon = True
        self.video_thread.start()
        
        self.speech_thread = threading.Thread(target=self.process_speech_queue)
        self.speech_thread.daemon = True
        self.speech_thread.start()
        
        self.speech_recognition_thread = threading.Thread(target=self.process_speech_recognition)
        self.speech_recognition_thread.daemon = True
        self.speech_recognition_thread.start()
        

        self.update_gui()
    
    def initialize_tts(self):
        """Initialize text-to-speech engine with error handling"""
        try:
            self.engine = pyttsx3.init()

            voices = self.engine.getProperty('voices')
            if voices:
                self.engine.setProperty('voice', voices[0].id) 
            self.engine.setProperty('rate', 150)

            self.engine.say(" ")
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"Failed to initialize TTS engine: {e}")
            return False
        
    def create_widgets(self):
      
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        

        self.camera_frame = ttk.LabelFrame(main_frame, text="Camera Input", padding="10")
        self.camera_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack()
        

        translation_frame = ttk.LabelFrame(main_frame, text="ASL Translation (You)", padding="10")
        translation_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.translation_display = tk.Text(
            translation_frame, 
            height=10, 
            width=40, 
            font=('Arial', 16), 
            wrap=tk.WORD,
            state='disabled',
            bg='#e6f7ff' 
        )
        self.translation_display.pack(fill=tk.BOTH, expand=True)
        
     
        hearing_frame = ttk.LabelFrame(main_frame, text="Voice Translation (Other Person)", padding="10")
        hearing_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        self.hearing_display = tk.Text(
            hearing_frame, 
            height=10, 
            width=40, 
            font=('Arial', 16), 
            wrap=tk.WORD,
            state='disabled',
            bg='#fff2e6'
        )
        self.hearing_display.pack(fill=tk.BOTH, expand=True)
        

        self.speaker_indicator = ttk.Label(
            main_frame, 
            text="Current Speaker: None", 
            font=('Arial', 12, 'bold'),
            foreground='gray'
        )
        self.speaker_indicator.grid(row=1, column=0, sticky="n", pady=10)
        

        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)
        
        self.speak_button = ttk.Button(
            controls_frame, 
            text="Speak Translation", 
            command=self.speak_translation,
            state='normal' if self.tts_available else 'disabled'
        )
        self.speak_button.pack(side=tk.LEFT, padx=5)
        
        self.listen_button = ttk.Button(
            controls_frame,
            text="Start Listening",
            command=self.toggle_listening
        )
        self.listen_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(
            controls_frame, 
            text="Clear All", 
            command=self.clear_all
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
    
        tts_status = " (TTS Available)" if self.tts_available else " (TTS Unavailable)"
        ttk.Label(controls_frame, text=tts_status).pack(side=tk.LEFT, padx=10)
        

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
    def toggle_listening(self):
        """Toggle speech recognition on/off"""
        self.speech_recognition_active = not self.speech_recognition_active
        if self.speech_recognition_active:
            self.listen_button.config(text="Stop Listening")
            self.current_speaker = 'voice'
            self.update_speaker_indicator()
        else:
            self.listen_button.config(text="Start Listening")
            self.current_speaker = None
            self.update_speaker_indicator()
    
    def update_speaker_indicator(self):
        """Update the speaker indicator based on who is currently speaking"""
        if self.current_speaker == 'asl':
            self.speaker_indicator.config(
                text="Current Speaker: ASL User (You)", 
                foreground='blue'
            )
        elif self.current_speaker == 'voice':
            self.speaker_indicator.config(
                text="Current Speaker: Voice User (Other)", 
                foreground='orange'
            )
        else:
            self.speaker_indicator.config(
                text="Current Speaker: None", 
                foreground='gray'
            )
    
    def process_video(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            processed_frame = self.process_frame(frame)
            
            try:
                self.frame_queue.put_nowait(processed_frame)
            except queue.Full:
                pass
                
    def process_frame(self, frame):
        H, W = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        data_aux = []
        x_ = []
        y_ = []
        
        current_sign = ""
        bbox = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
                
                # coordinates
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)
                
       
                min_x, min_y = min(x_), min(y_)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)
                
                # bounding box
                if x_ and y_:  
                    x1 = max(0, int(min(x_) * W) - 10)
                    y1 = max(0, int(min(y_) * H) - 10)
                    x2 = min(W, int(max(x_) * W) + 10)
                    y2 = min(H, int(max(y_) * H) + 10)
                    
                    if x1 < x2 and y1 < y2: 
                        bbox = (x1, y1, x2, y2)
                        
                    
                        if len(data_aux) == 42:
                            prediction = self.model.predict([np.asarray(data_aux)])
                            current_sign = prediction[0]
              
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                            cv2.putText(frame, current_sign, (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        

        self.update_sign(current_sign, bbox)
        

        if self.camera_shot_effect and bbox:
            elapsed = time.time() - self.effect_start_time
            if elapsed < self.effect_duration:
                # Flash effect - white rectangle
                alpha = 0.7 * (1 - elapsed/self.effect_duration)
                overlay = frame.copy()
                cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                             (255, 255, 255), -1)
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            else:
                self.camera_shot_effect = False
        
        return frame
    
    def update_sign(self, new_sign, bbox):
        current_time = time.time()

        if new_sign != self.current_sign:
            self.current_sign = new_sign
            self.last_sign_time = current_time
            self.stable_sign = ""
            return

        if (current_time - self.last_sign_time > self.sign_stable_time and new_sign):
            if new_sign == self.last_added_sign:
                if current_time - self.last_added_time > 2:  # 2 seconds delay
                    self.add_sign(new_sign, current_time)
            else:
                self.add_sign(new_sign, current_time)

    def add_sign(self, sign, current_time):
        char_to_add = " " if sign.lower() == "space" else sign
        self.translation_text += char_to_add
        self.camera_shot_effect = True
        self.effect_start_time = current_time
        self.last_added_sign = sign
        self.last_added_time = current_time
        self.current_speaker = 'asl'
        self.update_speaker_indicator()
        
      
        if self.tts_available:
            self.speech_queue.put(sign)
    
    def process_speech_queue(self):
        """Process speech requests from the queue"""
        while self.running:
            try:
                text = self.speech_queue.get(timeout=0.1)
                if text and self.engine:
                    try:
                        self.speaking = True
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except Exception as e:
                        print(f"Speech error: {e}")
                    finally:
                        self.speaking = False
                        self.speech_queue.task_done()
            except queue.Empty:
                continue
    
    def process_speech_recognition(self):
        """Process speech recognition in a separate thread"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            
        while self.running:
            if self.speech_recognition_active:
                try:
                    with self.microphone as source:
                        self.listening = True
                        audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=5)
                        self.listening = False
                        
                    try:
                        text = self.recognizer.recognize_google(audio)
                        self.hearing_text = text
                        self.current_speaker = 'voice'
                        self.update_speaker_indicator()
                        
                        # Auto-speak the recognized text if TTS is available
                        if self.tts_available:
                            self.speech_queue.put(text)
                            
                    except sr.UnknownValueError:
                        print("Could not understand audio")
                    except sr.RequestError as e:
                        print(f"Could not request results; {e}")
                        
                except Exception as e:
                    print(f"Speech recognition error: {e}")
                    self.listening = False
            else:
                time.sleep(0.1)
    
    def update_gui(self):
        try:
            frame = self.frame_queue.get_nowait()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
        except queue.Empty:
            pass
        

        self.translation_display.config(state='normal')
        self.translation_display.delete(1.0, tk.END)
        self.translation_display.insert(tk.END, self.translation_text)
        self.translation_display.config(state='disabled')
        

        self.hearing_display.config(state='normal')
        self.hearing_display.delete(1.0, tk.END)
        self.hearing_display.insert(tk.END, self.hearing_text)
        self.hearing_display.config(state='disabled')
        

        if self.listening:
            self.listen_button.config(text="Listening...")
        elif self.speech_recognition_active:
            self.listen_button.config(text="Stop Listening")
        else:
            self.listen_button.config(text="Start Listening")
        

        if self.running:
            self.root.after(30, self.update_gui)
    
    def speak_translation(self):
        if self.translation_text and self.tts_available:
            self.speech_queue.put(self.translation_text)
            self.current_speaker = 'asl'
            self.update_speaker_indicator()
    
    def clear_all(self):
        self.translation_text = ""
        self.hearing_text = ""
        self.stable_sign = ""
        self.current_sign = ""
        self.last_added_sign = None
        self.last_added_time = 0
        self.current_speaker = None
        self.update_speaker_indicator()
    
    def on_closing(self):
        self.running = False
        if self.video_thread.is_alive():
            self.video_thread.join()
        if self.speech_thread.is_alive():
            self.speech_thread.join()
        if self.speech_recognition_thread.is_alive():
            self.speech_recognition_thread.join()
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ASLTranslatorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
