import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import time
import threading
import pyttsx3
import queue
import speech_recognition as sr
import requests
import json
import os
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

    
class ASLTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SymVerbal - ASL Translator")

        try:
            self.root.iconbitmap(resource_path('icon.ico'))
        except Exception as e:
            print(f"Warning: Failed to load app icon: {e}")


        self.root.geometry("1400x800")
        self.root.configure(bg='#121212')
        
        model_path = resource_path('model.p')
        with open(model_path, 'rb') as f:
           self.model = pickle.load(f)['model']
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        self.engine = None
        self.tts_available = self.initialize_tts()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.speech_recognition_active = False
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video capture")
        
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
        self.ai_suggestions = []
        self.api_key = "ChristianDev-API-Key-002"
        self.frame_queue = queue.Queue(maxsize=1)
        self.suggestion_queue = queue.Queue()
        self.ai_assistant_active = False
        self.ai_chat_history = []
        
        self.setup_styles()
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
        self.ai_input.bind('<Return>', lambda event: self.ask_ai_question())
        self.update_gui()
    
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.style.configure('.', background='#121212', foreground='white')
        self.style.configure('TFrame', background='#121212')
        self.style.configure('TLabel', background='#121212', foreground='white')
        self.style.configure('TButton', background='#333333', foreground='white', 
                           borderwidth=1, focusthickness=3, focuscolor='#333333')
        self.style.map('TButton', background=[('active', '#444444')])
        self.style.configure('TEntry', fieldbackground='#222222', foreground='white', 
                            insertcolor='white', borderwidth=1)
        self.style.configure('TLabelFrame', background='#121212', foreground='#4a6fa5',
                           bordercolor='#333333', borderwidth=2)
        self.style.configure('TLabelFrame.Label', background='#121212', foreground='#4a6fa5')
        self.style.configure('TNotebook', background='#121212')
        self.style.configure('TNotebook.Tab', background='#333333', foreground='white',
                           padding=[10, 5], font=('Arial', 10, 'bold'))
        self.style.map('TNotebook.Tab', background=[('selected', '#4a6fa5')])
        
        self.style.configure('Accent.TButton', background='#4a6fa5', foreground='white',
                           font=('Arial', 10, 'bold'))
        self.style.map('Accent.TButton', background=[('active', '#5a7fb5')])
        
        self.style.configure('Suggestions.TButton', background='#333333', foreground='white',
                           font=('Arial', 10, 'bold'), padding=6)
        self.style.map('Suggestions.TButton', background=[('active', '#444444')])
        
        self.style.configure('Red.TButton', background='#d9534f', foreground='white')
        self.style.map('Red.TButton', background=[('active', '#e9635f')])
        
        self.style.configure('Green.TButton', background='#5cb85c', foreground='white')
        self.style.map('Green.TButton', background=[('active', '#6cc86c')])
        
        self.style.configure('Orange.TButton', background='#f0ad4e', foreground='white')
        self.style.map('Orange.TButton', background=[('active', '#ffbd5e')])
    
    def initialize_tts(self):
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
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.camera_frame = ttk.LabelFrame(left_panel, text="Camera Input", padding="10")
        self.camera_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack()
        
        translation_frame = ttk.LabelFrame(left_panel, text="ASL Translation", padding="10")
        translation_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.translation_display = tk.Text(
            translation_frame, 
            height=10, 
            width=40, 
            font=('Arial', 16), 
            wrap=tk.WORD,
            state='disabled',
            bg='#222222',
            fg='white',
            insertbackground='white',
            selectbackground='#4a6fa5',
            padx=10,
            pady=10,
            relief=tk.FLAT
        )
        self.translation_display.pack(fill=tk.BOTH, expand=True)
        
        self.get_suggestions_btn = ttk.Button(
            translation_frame,
            text="Get AI Suggestions",
            command=self.request_ai_suggestions,
            style='Suggestions.TButton'
        )
        self.get_suggestions_btn.pack(pady=5)
        
        self.suggestion_frame = ttk.Frame(translation_frame)
        self.suggestion_frame.pack(fill=tk.X, pady=(5, 0))
        
        hearing_frame = ttk.LabelFrame(left_panel, text="Voice Translation", padding="10")
        hearing_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        self.hearing_display = tk.Text(
            hearing_frame, 
            height=10, 
            width=40, 
            font=('Arial', 16), 
            wrap=tk.WORD,
            state='disabled',
            bg='#222222',
            fg='white',
            insertbackground='white',
            selectbackground='#4a6fa5',
            padx=10,
            pady=10,
            relief=tk.FLAT
        )
        self.hearing_display.pack(fill=tk.BOTH, expand=True)
        
        self.speaker_indicator = ttk.Label(
            left_panel, 
            text="Current Speaker: None", 
            font=('Arial', 12, 'bold'),
            foreground='#4a6fa5'
        )
        self.speaker_indicator.grid(row=1, column=0, sticky="n", pady=10)
        
        controls_frame = ttk.Frame(left_panel)
        controls_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)
        
        self.speak_button = ttk.Button(
            controls_frame, 
            text="Speak Translation", 
            command=self.speak_translation,
            style='Green.TButton',
            state='normal' if self.tts_available else 'disabled'
        )
        self.speak_button.pack(side=tk.LEFT, padx=5, ipadx=10)
        
        self.listen_button = ttk.Button(
            controls_frame,
            text="Start Listening",
            command=self.toggle_listening,
            style='Orange.TButton'
        )
        self.listen_button.pack(side=tk.LEFT, padx=5, ipadx=10)
        
        self.clear_button = ttk.Button(
            controls_frame, 
            text="Clear All", 
            command=self.clear_all,
            style='Red.TButton'
        )
        self.clear_button.pack(side=tk.LEFT, padx=5, ipadx=10)
        
        right_panel = ttk.LabelFrame(main_frame, text="SymVerbal AI", padding="10")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(
            right_panel,
            height=20,
            width=40,
            font=('Arial', 12),
            wrap=tk.WORD,
            state='disabled',
            bg='#222222',
            fg='white',
            insertbackground='white',
            selectbackground='#4a6fa5',
            padx=10,
            pady=10,
            relief=tk.FLAT
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        input_frame = ttk.Frame(right_panel)
        input_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.ai_input = ttk.Entry(
            input_frame,
            font=('Arial', 12)
        )
        self.ai_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.send_button = ttk.Button(
            input_frame,
            text="Send",
            command=self.ask_ai_question,
            style='Accent.TButton',
            width=5
        )
        self.send_button.pack(side=tk.RIGHT)
        
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        left_panel.columnconfigure(0, weight=1)
        left_panel.columnconfigure(1, weight=1)
        left_panel.rowconfigure(0, weight=1)
        left_panel.rowconfigure(1, weight=1)

    def request_ai_suggestions(self):
        if not self.translation_text:
            self.add_to_chat("SymVerbal AI", "No text available for suggestions")
            return
            
        self.clear_suggestions()
        self.get_suggestions_btn.config(text="Loading...", state='disabled')
        threading.Thread(target=self.get_ai_suggestions_thread, args=(self.translation_text,)).start()

    def get_ai_suggestions_thread(self, text):
        try:

            system_query = (
                "You are tasked with correcting text inputs. ",
                "Please provide exactly 3 different corrected versions of the following text. ",
                "Each version should have proper grammar, spelling, and clear wording. ",
                "Separate the three corrected versions using three vertical bars '|||'. ",
                "Important instructions: ",
                "- Only return the corrected text versions. ",
                "- Do NOT return any extra explanation, greetings, or formatting. ",
                "- Do NOT generate or return any images. ",
                "- If the text cannot be understood or corrected, simply return: could not understand.",
            )

            query = f"{system_query} {text}"
            encoded_query = requests.utils.quote(query)

            
            
            response = requests.get(
                f"https://apisystem.christiandev.xyz/api/gpt-3.5?apikey={self.api_key}&q={encoded_query}",
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status', False):
                    corrected_text = data['data']['response']
                    suggestions = [s.strip() for s in corrected_text.split('|||') if s.strip()]
                    if suggestions:
                        self.root.after(0, lambda: self.show_suggestions(suggestions))
                    else:
                        self.root.after(0, lambda: self.add_to_chat("SymVerbal AI", "No suggestions available"))
            else:
                self.root.after(0, lambda: self.add_to_chat("SymVerbal AI", f"API Error: {response.status_code}"))
        except Exception as e:
            print(f"Error getting AI suggestions: {e}")
            self.root.after(0, lambda: self.add_to_chat("SymVerbal AI", f"Error: {str(e)}"))
        finally:
            self.root.after(0, lambda: self.get_suggestions_btn.config(text="Get AI Suggestions", state='normal'))

    def clear_suggestions(self):
        for widget in self.suggestion_frame.winfo_children():
            widget.destroy()
        
    def show_suggestions(self, suggestions):
        self.clear_suggestions()
        header = tk.Label(self.suggestion_frame, 
                        text="AI Suggestions:", 
                        font=('Arial', 10, 'bold'),
                        bg='#222222',
                        fg='#4a6fa5')
        header.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        
        for i, suggestion in enumerate(suggestions[:3]):
            btn = tk.Button(
                self.suggestion_frame,
                text=suggestion,
                command=lambda s=suggestion: self.apply_suggestion(s),
                font=('Arial', 10),
                bg='#333333',
                fg='white',
                relief=tk.FLAT,
                activebackground='#4a6fa5',
                activeforeground='white',
                wraplength=400,
                padx=5,
                pady=2,
                anchor='w',
                justify='left'
            )
            btn.pack(side=tk.TOP, fill=tk.X, pady=2)

    def apply_suggestion(self, text):
        self.translation_text = text
        self.clear_suggestions()
        self.current_speaker = 'asl'
        self.update_speaker_indicator()
        self.update_display(self.translation_display, self.translation_text)
        if self.tts_available:
            self.speech_queue.put(text)
        
    def ask_ai_question(self, event=None):
        question = self.ai_input.get() 
        if question:
            self.add_to_chat("You", question)
            self.ai_input.delete(0, tk.END)
            threading.Thread(target=self.process_ai_question, args=(question,)).start()

    def process_ai_question(self, question):
        try:
            system_prompt = (
                "You are an SymVerbal AI for the SymVerbal application developed by Computer Science Students from Quezon City University (QCU). "
                "Your primary focus is on sign language, sign language recognition, communication support, and education. "
                "You must only answer questions related to sign language, communication, accessibility, or the SymVerbal platform. "
                "Forbid and firmly refuse any bullying, offensive behavior, or illegal activities. "
                "You were developed and trained using convolutional neural networks (CNNs) and advanced machine learning techniques, "
                "primarily built with Python, PyTorch, and related AI frameworks. "
                "Answer respectfully, clearly, and stay professional at all times. "
                "If a user asks something unrelated to your focus, politely guide them back to the main topic of sign language or SymVerbal. "
                "You are trained to handle diverse sign language systems, including American Sign Language (ASL), British Sign Language (BSL), and others, recognizing the cultural and linguistic uniqueness of each. "
                "You are familiar with speech-to-text technologies, computer vision for gesture recognition, and accessibility innovations. "
                "You must always encourage inclusive communication practices, and promote the rights and dignity of the Deaf and Hard of Hearing communities. "
                "If a question involves technical aspects, you may explain concepts such as machine learning, CNNs, data augmentation, real-time video processing, or gesture classification, but always link your explanation back to how it supports communication and accessibility. "
                "You operate under ethical guidelines: you must never assist in harmful activities, generate or spread misinformation, violate privacy, or discriminate. "
                "Maintain a friendly but informative tone at all times. "
            )

            query = f"{system_prompt} {question}"
            encoded_query = requests.utils.quote(query)

            response = requests.get(
                f"https://apisystem.christiandev.xyz/api/gpt-3.5?apikey={self.api_key}&q={encoded_query}",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status', False):
                    ai_response = data['data']['response']
                    self.process_ai_response(ai_response)
                    
        except Exception as e:
            self.add_to_chat("SymVerbal AI", f"Error: {str(e)}")

    def process_ai_response(self, response):
        if '<img ' in response.lower():
            try:
                start = response.lower().find('src="') + 5
                end = response.lower().find('"', start)
                img_url = response[start:end]
                text_part = response[:response.lower().find('<img')].strip()
                if text_part:
                    self.add_to_chat("SymVerbal AI", text_part)
                threading.Thread(target=self.load_and_display_image, args=(img_url,)).start()
                text_part = response[response.lower().find('>')+1:].strip()
                if text_part:
                    self.add_to_chat("SymVerbal AI", text_part)
            except Exception as e:
                print(f"Error processing image response: {e}")
                self.add_to_chat("SymVerbal AI", response)
        else:
            self.add_to_chat("SymVerbal AI", response)

    def load_and_display_image(self, img_url):
        try:
            response = requests.get(img_url, stream=True, timeout=10)
            if response.status_code == 200:
                img_data = response.content
                img = Image.open(io.BytesIO(img_data))
                max_size = (300, 300)
                img.thumbnail(max_size)
                photo_img = ImageTk.PhotoImage(img)
                self.root.after(0, lambda: self.display_image_in_chat(photo_img, img.size))
        except Exception as e:
            print(f"Error loading image: {e}")
            self.root.after(0, lambda: self.add_to_chat("SymVerbal AI", f"[Could not load image from {img_url}]"))

    def display_image_in_chat(self, photo_img, size):
        self.chat_display.config(state='normal')
        img_frame = tk.Frame(self.chat_display, bg='#222222')
        img_label = tk.Label(img_frame, image=photo_img, bg='#222222')
        img_label.image = photo_img
        img_label.pack()
        size_label = tk.Label(img_frame, 
                            text=f"Image size: {size[0]}×{size[1]}",
                            font=('Arial', 8),
                            bg='#222222',
                            fg='white')
        size_label.pack()
        self.chat_display.window_create(tk.END, window=img_frame)
        self.chat_display.insert(tk.END, "\n\n")
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
        if not hasattr(self, 'chat_images'):
            self.chat_images = []
        self.chat_images.append(photo_img)
    
    def add_to_chat(self, sender, message):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, f"{sender}: ", 'sender_tag')
        self.chat_display.insert(tk.END, f"{message}\n\n")
        if sender == "You":
            self.chat_display.tag_config('sender_tag', foreground='#4a6fa5', font=('Arial', 12, 'bold'))
        else:
            self.chat_display.tag_config('sender_tag', foreground='#5cb85c', font=('Arial', 12, 'bold'))
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
        
    def start_threads(self):
        threads = [
            threading.Thread(target=self.process_video),
            threading.Thread(target=self.process_speech_queue),
            threading.Thread(target=self.process_speech_recognition),
            threading.Thread(target=self.process_suggestions)
        ]
        for thread in threads:
            thread.daemon = True
            thread.start()

    def process_suggestions(self):
        while self.running:
            try:
                suggestions = self.suggestion_queue.get(timeout=0.1)
                self.ai_suggestions = suggestions
                self.show_suggestions(suggestions)
                self.suggestion_queue.task_done()
            except queue.Empty:
                continue
    
    def toggle_listening(self):
        self.speech_recognition_active = not self.speech_recognition_active
        if self.speech_recognition_active:
            self.listen_button.config(text="Stop Listening", style='Red.TButton')
            self.current_speaker = 'voice'
            self.update_speaker_indicator()
        else:
            self.listen_button.config(text="Start Listening", style='Orange.TButton')
            self.current_speaker = None
            self.update_speaker_indicator()
    
    def update_speaker_indicator(self):
        if self.current_speaker == 'asl':
            self.speaker_indicator.config(
                text="Current Speaker: ASL User (You)", 
                foreground='#4a6fa5'
            )
        elif self.current_speaker == 'voice':
            self.speaker_indicator.config(
                text="Current Speaker: Voice User (Other)", 
                foreground='#f0ad4e'
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
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )
                
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)
                
                min_x, min_y = min(x_), min(y_)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)
                
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
                if current_time - self.last_added_time > 2:
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
        
        self.update_display(self.translation_display, self.translation_text)
        self.update_display(self.hearing_display, self.hearing_text)
        
        if self.listening:
            self.listen_button.config(text="Listening...", style='Red.TButton')
        elif self.speech_recognition_active:
            self.listen_button.config(text="Stop Listening", style='Red.TButton')
        else:
            self.listen_button.config(text="Start Listening", style='Orange.TButton')
        
        if self.running:
            self.root.after(30, self.update_gui)
    
    def update_display(self, display, text):
        display.config(state='normal')
        display.delete(1.0, tk.END)
        display.insert(tk.END, text)
        display.config(state='disabled')
    
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
        if self.ai_assistant_thread.is_alive():
            self.ai_assistant_thread.join()
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ASLTranslatorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()