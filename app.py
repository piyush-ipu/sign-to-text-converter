from PIL import Image, ImageTk
import tkinter as tk
import cv2
import os
import numpy as np
from tensorflow.keras.models import model_from_json
import operator
from string import ascii_uppercase
from spellchecker import SpellChecker

class Application:
    def __init__(self):
        self.directory = 'model'
        self.spell_checker = SpellChecker()  # Initialize the SpellChecker
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None

        # Load base models
        self.loaded_model = self.load_model("model-bw.json", "model-bw.h5")
        self.loaded_model_dru = self.load_model("model-bw_dru.json", "model-bw_dru.h5")
        self.loaded_model_tkdi = self.load_model("model-bw_tkdi.json", "model-bw_tkdi.h5")
        self.loaded_model_smn = self.load_model("model-bw_smn.json", "model-bw_smn.h5")

        # Initialize counters
        self.ct = {'blank': 0}
        for i in ascii_uppercase:
            self.ct[i] = 0
        self.blank_flag = 0
        print("Loaded model from disk")

        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Sign Language to Text Converter")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x1100")

        # Image Panels
        self.panel = tk.Label(self.root)
        self.panel.place(x=140, y=10, width=640, height=500)

        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=500, y=95, width=310, height=310)

        # Labels
        self.T = tk.Label(self.root, text="Sign Language to Text", font=("Courier", 40, "bold"))
        self.T.place(x=31, y=17)

        self.T1 = tk.Label(self.root, text="Character :", font=("Courier", 40, "bold"))
        self.T1.place(x=10, y=500)

        self.panel3 = tk.Label(self.root)  # Current Symbol
        self.panel3.place(x=500, y=500)

        self.T2 = tk.Label(self.root, text="Word :", font=("Courier", 40, "bold"))
        self.T2.place(x=10, y=550)

        self.panel4 = tk.Label(self.root)  # Word
        self.panel4.place(x=220, y=550)

        self.T3 = tk.Label(self.root, text="Sentence :", font=("Courier", 40, "bold"))
        self.T3.place(x=10, y=600)

        self.panel5 = tk.Label(self.root) 
        self.panel5.place(x=350, y=600)

        # Buttons
        self.btcall = tk.Button(self.root, command=self.action_call, text="About", font=("Courier", 14))
        self.btcall.place(x=825, y=0)

        # Updated Button Commands to Use lambda for Suggestions
        self.bt1 = tk.Button(self.root, command=lambda: self.select_suggestion(0), font=("Courier", 20))
        self.bt1.place(x=26, y=890)

        self.bt2 = tk.Button(self.root, command=lambda: self.select_suggestion(1), font=("Courier", 20))
        self.bt2.place(x=325, y=890)

        self.bt3 = tk.Button(self.root, command=lambda: self.select_suggestion(2), font=("Courier", 20))
        self.bt3.place(x=625, y=890)

        self.bt4 = tk.Button(self.root, command=lambda: self.select_suggestion(3), font=("Courier", 20))
        self.bt4.place(x=125, y=950)

        self.bt5 = tk.Button(self.root, command=lambda: self.select_suggestion(4), font=("Courier", 20))
        self.bt5.place(x=425, y=950)

        # Initialize text variables
        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.photo = "Empty"

        # Bind Keys for User Interactions
        self.root.bind('<Return>', self.confirm_symbol)      
        self.root.bind('<BackSpace>', self.delete_last_word) 

        # Start video loop
        self.video_loop()

    def load_model(self, json_filename, weights_filename):
        json_path = os.path.join(self.directory, json_filename)
        weights_path = os.path.join(self.directory, weights_filename)
        with open(json_path, "r") as json_file:
            model_json = json_file.read()
        loaded_model = model_from_json(model_json)
        loaded_model.load_weights(weights_path)
        return loaded_model

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])
            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            # Process the ROI (Region of Interest)
            roi = cv2image[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            predicts = self.predict(res)  
            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)

            # Update Labels
            self.panel3.config(text=self.current_symbol, font=("Courier", 50))
            self.panel4.config(text=self.word, font=("Courier", 40))
            self.panel5.config(text=self.str, font=("Courier", 40))

            # Update Suggestions
            self.update_suggestions(self.spell_checker.candidates(self.word))

        self.root.after(30, self.video_loop)

    def update_suggestions(self, predicts):
        if isinstance(predicts, set):
            predicts = list(predicts)

        buttons = [self.bt1, self.bt2, self.bt3, self.bt4, self.bt5]
        for idx, button in enumerate(buttons):
            if len(predicts) > idx:
                button.config(text=predicts[idx])
            else:
                button.config(text="")

        # Display the updated text in the corresponding labels
        self.panel4.config(text=self.word, font=("Courier", 40))
        self.panel5.config(text=self.str, font=("Courier", 40))

    def predict(self, test_image):
        # Resize and reshape the image for prediction
        test_image_resized = cv2.resize(test_image, (128, 128))
        test_image_reshaped = test_image_resized.reshape(1, 128, 128, 1)

        # Predict using all models
        result = self.loaded_model.predict(test_image_reshaped)
        result_dru = self.loaded_model_dru.predict(test_image_reshaped)
        result_tkdi = self.loaded_model_tkdi.predict(test_image_reshaped)
        result_smn = self.loaded_model_smn.predict(test_image_reshaped)

        # Debugging
        print("result:", result)
        print("result_dru:", result_dru)
        print("result_tkdi:", result_tkdi)
        print("result_smn:", result_smn)

        # Aggregate predictions
        prediction = {'blank': result[0][0]}
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1

        # Sort predictions
        prediction = sorted(prediction.items(), key=lambda x: x[1], reverse=True)
        self.current_symbol = prediction[0][0]

        # Layer 2 Predictions
        if self.current_symbol in ['D', 'R', 'U']:
            if isinstance(result_dru, np.ndarray) and len(result_dru[0]) >= 3:
                prediction = {
                    'D': result_dru[0][0],
                    'R': result_dru[0][1],
                    'U': result_dru[0][2],
                }
                prediction = sorted(prediction.items(), key=lambda x: x[1], reverse=True)
                self.current_symbol = prediction[0][0]

        # Layer 3 Predictions
        if self.current_symbol in ['T', 'K', 'D', 'I']:
            if isinstance(result_tkdi, np.ndarray) and len(result_tkdi[0]) >= 4:
                prediction = {
                    'T': result_tkdi[0][0],
                    'K': result_tkdi[0][1],
                    'D': result_tkdi[0][2],
                    'I': result_tkdi[0][3],
                }
                prediction = sorted(prediction.items(), key=lambda x: x[1], reverse=True)
                self.current_symbol = prediction[0][0]

        # Layer 4 Predictions
        if self.current_symbol == 'S':
            if isinstance(result_smn, np.ndarray) and len(result_smn[0]) >= 2:
                prediction = {
                    'S': result_smn[0][0],
                    'N': result_smn[0][1],
                }
                prediction = sorted(prediction.items(), key=lambda x: x[1], reverse=True)
                self.current_symbol = prediction[0][0]

        # Check if current symbol is blank
        if self.current_symbol == 'blank':
            self.blank_flag += 1
            if self.blank_flag > 5:
                # Optionally, handle spaces here or keep it for automatic space insertion
                self.blank_flag = 0
        else:
            self.blank_flag = 0
            self.ct[self.current_symbol] += 1

            # Instead of appending, set a flag or update the GUI to indicate readiness
            if self.ct[self.current_symbol] > 5:
                # Example: Change label color to indicate readiness for confirmation
                self.panel3.config(fg="green")  # Highlight symbol in green
            else:
                self.panel3.config(fg="black")  # Default color

        # Update the string without appending
        self.str = self.word
        return set(self.word)

    def action_call(self):
        tk.messagebox.showinfo("About", "Sign Language to Text Converter v1.0")

    def select_suggestion(self, index):
        candidates = list(self.spell_checker.candidates(self.word))
        if index < len(candidates):
            selected_word = candidates[index]
            self.str = selected_word + " "  
            self.word = ""  
            self.panel4.config(text=self.word, font=("Courier", 40))
            self.panel5.config(text=self.str, font=("Courier", 40))
            print(f"Selected suggestion: {selected_word}")
        else:
            print(f"No suggestion available for index {index}")

    def delete_last_word(self, event):
        """
        Deletes the last word from the sentence or resets the current word.
        Triggered by pressing the Backspace key.
        """
        if self.word:
            # If there's a word being formed, reset it
            words = self.str.split()
            if words: 
                words.pop()   
                self.str = " ".join(words) + " "  

            else:
                self.str = "" #Clear the string if no words are left 
                self.panel5.config(text = self.str, font= ("Courier",40)) 
        
        # Update the combined string
        self.panel4.config(text=self.word, font=("Courier", 40))
        self.panel5.config(text=self.str, font=("Courier", 40))
        
        print("Last word deleted. Current sentence:", self.str)  

    def confirm_symbol(self, event):
        """
        Confirms the current predicted symbol and adds it to the current word and sentence.
        Triggered by pressing the Enter key.
        """
        if self.current_symbol not in ['blank', 'Empty'] and self.ct[self.current_symbol] > 5:
            # Append the current symbol to the current word
            self.word += self.current_symbol
            print(f"Symbol '{self.current_symbol}' added to the current word.")

            # Reset the symbol counter for the current symbol
            self.ct[self.current_symbol] = 0

            # Reset the symbol display color
            self.panel3.config(fg="black")

            # Update the combined string
            self.str = self.word

            # Update the labels in the GUI
            self.panel4.config(text=self.word, font=("Courier", 40))
            self.panel5.config(text=self.str, font=("Courier", 40))

            print(f"Current word: {self.word}")
            print(f"Sentence: {self.str}")
        else:
            print("No valid symbol to confirm.")

    def destructor(self):
        self.vs.release()
        cv2.destroyAllWindows()
        self.root.quit()

    def action_call(self) : #About section of the application
        
        self.root1 = tk.Toplevel(self.root)
        self.root1.title("About")
        self.root1.protocol('WM_DELETE_WINDOW', self.destructor1)
        self.root1.geometry("900x1100")
         
        self.tx = tk.Label(self.root1)
        self.tx.place(x = 330,y = 20)
        self.tx.config(text = "Efforts By", fg="red", font = ("Courier",30,"bold"))

        self.photo1 = tk.PhotoImage(file='Picture/Piyush_Kumar.png')
        self.w1 = tk.Label(self.root1, image = self.photo1)
        self.w1.place(x = 20, y = 105)
        self.tx6 = tk.Label(self.root1)
        self.tx6.place(x = 20,y = 250)
        self.tx6.config(text = "Piyush Kumar", font = ("Courier",15,"bold"))

        self.photo2 = tk.PhotoImage(file='Picture/Garima_Sharma.png')
        self.w2 = tk.Label(self.root1, image = self.photo2)
        self.w2.place(x = 200, y = 105)
        self.tx2 = tk.Label(self.root1)
        self.tx2.place(x = 200,y = 250)
        self.tx2.config(text = "Garima Sharma", font = ("Courier",15,"bold"))

        
        self.photo3 = tk.PhotoImage(file='Picture/Lakshay_Jaint.png')
        self.w3 = tk.Label(self.root1, image = self.photo3)
        self.w3.place(x = 380, y = 105)
        self.tx3 = tk.Label(self.root1)
        self.tx3.place(x = 380,y = 250)
        self.tx3.config(text = "Lakshay Jaint", font = ("Courier",15,"bold"))

        self.photo4 = tk.PhotoImage(file='Picture/Suraj_Pandey.png')
        self.w4 = tk.Label(self.root1, image = self.photo4)
        self.w4.place(x = 560, y = 105)
        self.tx4 = tk.Label(self.root1)
        self.tx4.place(x = 560,y = 250)
        self.tx4.config(text = "Suraj Pandey", font = ("Courier",15,"bold"))    

if __name__ == "__main__":
     app = Application()
     app.root.mainloop()