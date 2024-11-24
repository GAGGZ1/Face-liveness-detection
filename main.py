import os
import datetime
import pickle
import subprocess
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import face_recognition
import util  # Assuming util module is implemented and working

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.title("Face Recognition App")
        self.main_window.geometry("1200x520+350+100")

        # Buttons
        self.login_button_main_window = util.get_button(self.main_window, 'Login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)

        self.logout_button_main_window = util.get_button(self.main_window, 'Logout', 'red', self.logout)
        self.logout_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(
            self.main_window, 'Register New User', 'gray', self.register_new_user, fg='black'
        )
        self.register_new_user_button_main_window.place(x=750, y=400)

        # Webcam Label
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        # Webcam Initialization
        self.cap = cv2.VideoCapture(0)  # Use index 0 for the default camera
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Webcam could not be initialized. Check the camera connection.")
            self.main_window.destroy()
            return

        self.process_webcam()

        # Directory Setup
        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'

    def process_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            self.most_recent_capture_arr = frame
            img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
            self.most_recent_capture_pil = Image.fromarray(img_)
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
            self.webcam_label.imgtk = imgtk
            self.webcam_label.configure(image=imgtk)
        else:
            messagebox.showerror("Error", "Failed to read from webcam. Restart the application.")

        self.webcam_label.after(20, self.process_webcam)

    def login(self):
        if not hasattr(self, 'most_recent_capture_arr'):
            messagebox.showerror("Error", "No webcam feed detected. Ensure the camera is functioning.")
            return

        unknown_img_path = './.tmp.jpg'
        cv2.imwrite(unknown_img_path, self.most_recent_capture_arr)

        try:
            # Use face_recognition CLI to check for a match
            output = subprocess.check_output(['face_recognition', self.db_dir, unknown_img_path], text=True).strip()
            print("Face Recognition Output:", output)  # Debugging line

            # Extract name from the output
            name = output.split(',')[1][:-3]

            # Handle unknown cases
            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Error', 'Unknown user. Please register as a new user or try again.')
            else:
                util.msg_box('Welcome back!', f'Welcome, {name}.')
                with open(self.log_path, 'a') as f:
                    f.write(f'{name},{datetime.datetime.now()},in\n')

        except Exception as e:
            messagebox.showerror("Error", f"Face recognition failed: {str(e)}")
        finally:
            os.remove(unknown_img_path)


    def logout(self):
        util.msg_box('Logged Out', 'You have been successfully logged out.')
        self.main_window.destroy()

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.title("Register New User")
        self.register_new_user_window.geometry("1200x520+370+120")

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(
            self.register_new_user_window, 'Please, \ninput username:'
        )
        self.text_label_register_new_user.place(x=750, y=70)

        self.accept_button = util.get_button(self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
        self.accept_button.place(x=750, y=300)

        self.try_again_button = util.get_button(self.register_new_user_window, 'Try Again', 'red', self.try_again_register_new_user)
        self.try_again_button.place(x=750, y=400)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c").strip()
        if not name:
            messagebox.showerror("Error", "Username cannot be empty.")
            return

        embeddings = face_recognition.face_encodings(self.most_recent_capture_arr)
        if not embeddings:
            messagebox.showerror("Error", "No face detected. Try again.")
            return

        with open(os.path.join(self.db_dir, f'{name}.pickle'), 'wb') as file:
            pickle.dump(embeddings[0], file)

        util.msg_box('Success!', 'User was registered successfully!')
        self.register_new_user_window.destroy()

    def start(self):
        self.main_window.mainloop()

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


if __name__ == "__main__":
    app = App()
    app.start()
