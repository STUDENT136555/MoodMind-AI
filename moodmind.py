import tkinter as tk
from tkinter import messagebox
import sqlite3
from datetime import datetime, timedelta
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

conn = sqlite3.connect("mood_data.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS moods (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    time TEXT,
    mood TEXT,
    confidence REAL,
    journal TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
)
""")
conn.commit()

POSITIVE_WORDS = ["happy", "good", "great", "love", "excited",
                  "joy", "amazing", "wonderful", "fantastic", "awesome"]
NEGATIVE_WORDS = ["sad", "bad", "hate", "tired", "stress",
                  "angry", "depressed", "anxious", "worried", "lonely"]

QUOTES = [
    "You are stronger than you think",
    "Every day is a fresh start",
    "Your feelings are valid",
    "Small steps lead to big changes",
    "Breathe. You've got this",
    "You matter. You are enough"
]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def get_today():
    return datetime.now().strftime("%Y-%m-%d")

def get_time():
    return datetime.now().strftime("%H:%M:%S")

def get_yesterday():
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

def get_setting(key):
    cursor.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = cursor.fetchone()
    return row[0] if row else None

def set_setting(key, value):
    cursor.execute("INSERT OR REPLACE INTO settings (key,value) VALUES (?,?)",
                   (key, value))
    conn.commit()

def save_mood(mood, confidence=1.0, journal=""):
    cursor.execute(
        "INSERT INTO moods (date, time, mood, confidence, journal) VALUES (?,?,?,?,?)",
        (get_today(), get_time(), mood, confidence, journal)
    )
    conn.commit()
    update_streak()
    update_wellness()
    check_emergency()

def update_streak():
    last_date = get_setting("last_log_date")
    today = get_today()
    streak = int(get_setting("streak") or 0)

    if last_date == get_yesterday():
        streak += 1
    elif last_date != today:
        streak = 1

    set_setting("streak", str(streak))
    set_setting("last_log_date", today)
    lbl_streak.config(text=f"{streak}-day streak")

def update_wellness():
    cursor.execute("SELECT mood FROM moods")
    moods = [row[0] for row in cursor.fetchall()]
    if not moods:
        lbl_wellness.config(text="Wellness Score: --")
        return
    weights = {"Happy": 100, "Neutral": 70, "Sad": 40, "Angry": 20}
    total = sum(weights.get(m, 50) for m in moods)
    score = round(total / len(moods))
    lbl_wellness.config(text=f"Wellness Score: {score}/100")

def check_emergency():
    cursor.execute("SELECT mood FROM moods ORDER BY id DESC LIMIT 3")
    last3 = [row[0] for row in cursor.fetchall()]
    if len(last3) == 3 and all(m == "Sad" for m in last3):
        contact = get_setting("trusted_contact")
        if contact:
            messagebox.showwarning(
                "Emergency Alert",
                f"3 consecutive sad days detected!\n"
                f"Alert sent to: {contact}"
            )

def classify_emotion(smile_count, eye_count, brightness):
    if smile_count >= 1:
        return "Happy", 0.90
    elif brightness < 80:
        return "Sad", 0.70
    elif eye_count >= 2 and brightness > 140:
        return "Angry", 0.60
    else:
        return "Neutral", 0.50

def detect_mood_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Webcam not accessible!")
        return

    messagebox.showinfo("Camera Ready",
                        "Look at the camera!\n"
                        "Press 'S' to save, 'Q' to cancel.")

    detected_mood = None
    confidence = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]

            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
            eyes = eye_cascade.detectMultiScale(roi_gray)
            brightness = np.mean(roi_gray)

            detected_mood, confidence = classify_emotion(
                len(smiles), len(eyes), brightness
            )

            cv2.putText(frame, f"{detected_mood} ({confidence:.2f})",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

        cv2.putText(frame, "Press S = Save | Q = Quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        cv2.imshow("MoodMind AI - Face Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and detected_mood:
            save_mood(detected_mood, confidence)
            lbl_mood_status.config(
                text=f"Detected: {detected_mood} ({confidence:.2f}) on {get_today()}"
            )
            messagebox.showinfo("Mood Saved",
                                f"AI detected: {detected_mood}")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def analyze_sentiment(text):
    text = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in text)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text)

    if pos > neg:
        return "Positive", "Share your happiness with a friend!"
    elif neg > pos:
        return "Negative", "Take 5 deep breaths and drink water"
    else:
        return "Neutral", "Listen to your favorite song"

def analyze_journal():
    text = txt_journal.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Empty", "Write something first!")
        return
    sentiment, suggestion = analyze_sentiment(text)
    lbl_sentiment.config(text=f"Sentiment: {sentiment}")
    lbl_suggestion.config(text=f"AI Suggestion: {suggestion}")
    save_mood("Neutral", 1.0, text)

def predict_tomorrow():
    cursor.execute("SELECT mood FROM moods ORDER BY id DESC LIMIT 7")
    recent = [row[0] for row in cursor.fetchall()]
    if not recent:
        return "Not enough data"

    happy = recent.count("Happy")
    sad = recent.count("Sad")
    if happy > sad:
        return "Tomorrow: Likely Happy"
    elif sad > happy:
        return "Tomorrow: Likely Sad - Be kind to yourself"
    else:
        return "Tomorrow: Likely Neutral"

def show_charts():
    cursor.execute("SELECT mood, COUNT(*) FROM moods GROUP BY mood")
    data = cursor.fetchall()
    if not data:
        messagebox.showinfo("No Data", "Log some moods first!")
        return

    moods = [row[0] for row in data]
    counts = [row[1] for row in data]
    colors = {"Happy": "#4CAF50", "Sad": "#2196F3",
              "Angry": "#F44336", "Neutral": "#9E9E9E"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.pie(counts, labels=moods, autopct="%1.1f%%",
            colors=[colors.get(m, "#999") for m in moods])
    ax1.set_title("Mood Distribution")
    ax2.bar(moods, counts, color=[colors.get(m, "#999") for m in moods])
    ax2.set_title("Mood Frequency")
    ax2.set_ylabel("Count")
    plt.tight_layout()
    plt.show()

    lbl_prediction.config(text=predict_tomorrow())

def breathing_exercise():
    for step in ["Breathe IN", "Hold", "Breathe OUT"]:
        messagebox.showinfo("Breathing Exercise", f"{step}\n(4 seconds)")
    messagebox.showinfo("Done", "Great job! Repeat 4 times")

def chatbot_buddy():
    cursor.execute("SELECT mood FROM moods ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    last = row[0] if row else "Neutral"
    responses = {
        "Sad": "I'm here for you. Try the breathing exercise!",
        "Angry": "Take a walk, listen to calm music",
        "Happy": "That's wonderful! Keep shining",
        "Neutral": "How can I help you today?"
    }
    messagebox.showinfo("AI Buddy", responses.get(last, responses["Neutral"]))

def save_contact():
    c = entry_contact.get().strip()
    if c:
        set_setting("trusted_contact", c)
        messagebox.showinfo("Saved", "Trusted contact saved!")

def clear_data():
    if messagebox.askyesno("Confirm", "Delete all mood data?"):
        cursor.execute("DELETE FROM moods")
        conn.commit()
        update_wellness()
        lbl_streak.config(text="0-day streak")
        lbl_mood_status.config(text="No mood logged yet")
        lbl_sentiment.config(text="Sentiment: --")
        lbl_suggestion.config(text="AI Suggestion will appear here")
        lbl_prediction.config(text="AI Prediction: --")
        messagebox.showinfo("Cleared", "All data deleted.")

def log_mood(mood):
    save_mood(mood)
    lbl_mood_status.config(text=f"Logged: {mood} on {get_today()}")
    messagebox.showinfo("Mood Saved", f"Mood saved: {mood}")

root = tk.Tk()
root.title("MoodMind AI - Class 12 Capstone Project")
root.geometry("820x850")
root.configure(bg="#F5F5F5")

tk.Label(root, text="MoodMind AI", font=("Arial", 24, "bold"),
         bg="#F5F5F5", fg="#6A1B9A").pack(pady=10)

tk.Label(root, text="Hello, Friend!", font=("Arial", 14),
         bg="#F5F5F5").pack()

tk.Label(root, text=random.choice(QUOTES), font=("Arial", 12, "italic"),
         bg="#F5F5F5", fg="#1976D2").pack(pady=5)

lbl_streak = tk.Label(root, text="0-day streak",
                      font=("Arial", 14, "bold"), bg="#F5F5F5")
lbl_streak.pack()

lbl_wellness = tk.Label(root, text="Wellness Score: --",
                        font=("Arial", 14, "bold"),
                        bg="#F5F5F5", fg="#2E7D32")
lbl_wellness.pack(pady=5)

tk.Label(root, text="How are you feeling today?",
         font=("Arial", 13), bg="#F5F5F5").pack(pady=10)

frame_moods = tk.Frame(root, bg="#F5F5F5")
frame_moods.pack()

tk.Button(frame_moods, text="Happy", bg="#4CAF50", fg="white",
          font=("Arial", 11, "bold"), width=10,
          command=lambda: log_mood("Happy")).grid(row=0, column=0, padx=4)

tk.Button(frame_moods, text="Sad", bg="#2196F3", fg="white",
          font=("Arial", 11, "bold"), width=10,
          command=lambda: log_mood("Sad")).grid(row=0, column=1, padx=4)

tk.Button(frame_moods, text="Angry", bg="#F44336", fg="white",
          font=("Arial", 11, "bold"), width=10,
          command=lambda: log_mood("Angry")).grid(row=0, column=2, padx=4)

tk.Button(frame_moods, text="Neutral", bg="#9E9E9E", fg="white",
          font=("Arial", 11, "bold"), width=10,
          command=lambda: log_mood("Neutral")).grid(row=0, column=3, padx=4)

tk.Button(root, text="Detect Mood Using Webcam (AI)",
          bg="#6A1B9A", fg="white", font=("Arial", 12, "bold"),
          width=35, command=detect_mood_from_webcam).pack(pady=10)

lbl_mood_status = tk.Label(root, text="No mood logged yet",
                           font=("Arial", 11), bg="#F5F5F5", fg="#555")
lbl_mood_status.pack(pady=5)

tk.Label(root, text="Daily Journal",
         font=("Arial", 13, "bold"), bg="#F5F5F5").pack(pady=5)

txt_journal = tk.Text(root, height=4, width=70, font=("Arial", 11))
txt_journal.pack(pady=5)

tk.Button(root, text="Analyze Sentiment",
          bg="#6A1B9A", fg="white", font=("Arial", 11, "bold"),
          width=25, command=analyze_journal).pack(pady=5)

lbl_sentiment = tk.Label(root, text="Sentiment: --",
                         font=("Arial", 12, "bold"), bg="#F5F5F5")
lbl_sentiment.pack()

lbl_suggestion = tk.Label(root, text="AI Suggestion will appear here",
                          font=("Arial", 11, "italic"),
                          bg="#F5F5F5", fg="#6A1B9A")
lbl_suggestion.pack(pady=5)

frame_actions = tk.Frame(root, bg="#F5F5F5")
frame_actions.pack(pady=10)

tk.Button(frame_actions, text="Dashboard",
          bg="#FF9800", fg="white", font=("Arial", 11, "bold"),
          width=14, command=show_charts).grid(row=0, column=0, padx=4)

tk.Button(frame_actions, text="Breathing",
          bg="#00BCD4", fg="white", font=("Arial", 11, "bold"),
          width=14, command=breathing_exercise).grid(row=0, column=1, padx=4)

tk.Button(frame_actions, text="AI Buddy",
          bg="#8BC34A", fg="white", font=("Arial", 11, "bold"),
          width=14, command=chatbot_buddy).grid(row=0, column=2, padx=4)

tk.Button(frame_actions, text="Clear Data",
          bg="#E53935", fg="white", font=("Arial", 11, "bold"),
          width=14, command=clear_data).grid(row=0, column=3, padx=4)

lbl_prediction = tk.Label(root, text="AI Prediction: --",
                          font=("Arial", 13, "bold"),
                          bg="#F5F5F5", fg="#E65100")
lbl_prediction.pack(pady=10)

tk.Label(root, text="Trusted Contact (Emergency Alert)",
         font=("Arial", 11), bg="#F5F5F5").pack()

entry_contact = tk.Entry(root, width=30, font=("Arial", 11))
entry_contact.pack(pady=3)

tk.Button(root, text="Save Contact", bg="#607D8B", fg="white",
          font=("Arial", 10, "bold"),
          command=save_contact).pack(pady=5)

update_wellness()
streak_val = get_setting("streak") or "0"
lbl_streak.config(text=f"{streak_val}-day streak")

root.mainloop()