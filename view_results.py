import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

def view_results_from_csv(tab_widget, csv_path):
    for widget in tab_widget.winfo_children():
        widget.destroy()

    # ========== Scrollable canvas setup ==========
    canvas = tk.Canvas(tab_widget)
    scrollbar = ttk.Scrollbar(tab_widget, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    def resize_canvas(event):
        canvas.itemconfig(window_id, width=event.width)

    canvas.bind("<Configure>", resize_canvas)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # ========== Data loading and processing ==========
    df = pd.read_csv(csv_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['DrowsinessAlert'] = df['DrowsinessAlert'].astype(str) == 'Yes'
    df['YawnAlert'] = df['YawnAlert'].astype(str) == 'Yes'

    emotion_score_map = {
        'Happy': 2, 'Surprise': 1, 'Neutral': 1,
        'Sad': -1, 'Angry': -2, 'Fear': -2, 'Disgust': -3
    }

    def compute_focus_score(row):
        if row['DrowsinessAlert']:
            return -3
        elif row['YawnAlert']:
            return -2
        else:
            return emotion_score_map.get(row['Emotion'], 0)

    df['FocusScore'] = df.apply(compute_focus_score, axis=1)
    total_score = df['FocusScore'].sum()
    avg_score = df['FocusScore'].mean()

    def on_resize(event):
        canvas.itemconfig("all", width=event.width)

        canvas.bind("<Configure>", on_resize)

    # ========== Function to add each chart ==========
    def add_plot(fig, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True, pady=20)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()

        widget = canvas.get_tk_widget()
        widget.pack(fill='both', expand=True)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['DrowsinessAlert'] = df['DrowsinessAlert'].astype(str) == 'Yes'
    df['YawnAlert'] = df['YawnAlert'].astype(str) == 'Yes'

    duration = (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds()

    resample_interval = None
    if duration > 1800:
        resample_interval = '30S'
    elif duration > 900:
        resample_interval = '10S'
    elif duration > 300:
        resample_interval = '5S'

    def mode_agg(series):
        try:
            return series.mode()[0]
        except IndexError:
            return series.iloc[0]

    df = df.set_index('Timestamp')

    if resample_interval:
        df_resampled = df.resample(resample_interval).agg({
            'Emotion': mode_agg,
            'DrowsinessAlert': 'max',
            'YawnAlert': 'max',
            'FocusScore': 'mean'
        }).dropna().reset_index()
        df = df_resampled.reset_index(drop=True)
    else:
        df = df.reset_index()

    df = df.sort_values('Timestamp')
    df = df.drop_duplicates(subset='Timestamp', keep='first')

    # 1. Cảm xúc theo thời gian
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df['Timestamp'], df['Emotion'], marker='o', linestyle='-', alpha=0.7, markersize=2)
    ax1.set_title("Cảm xúc theo thời gian")
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True)
    fig1.tight_layout()
    add_plot(fig1, scrollable_frame)

    # 2. Tần suất cảm xúc
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    df['Emotion'].value_counts().plot(kind='bar', ax=ax2, color='skyblue')
    ax2.set_title("Tần suất cảm xúc")
    ax2.set_ylabel("Số lần")
    fig2.tight_layout()
    add_plot(fig2, scrollable_frame)

    # 3. Cảnh báo buồn ngủ/ngáp
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(df['Timestamp'], df['DrowsinessAlert'], label='Buồn ngủ', color='red')
    ax3.plot(df['Timestamp'], df['YawnAlert'], label='Ngáp', color='orange')
    ax3.set_title("Cảnh báo buồn ngủ / ngáp")
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Không', 'Có'])
    fig3.tight_layout()
    add_plot(fig3, scrollable_frame)

    # 4. Điểm tập trung
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(df['Timestamp'], df['FocusScore'], color='green', marker='o', markersize=2)
    ax4.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax4.set_title(f"Điểm tập trung (Tổng: {total_score}, TB: {avg_score:.2f})")
    ax4.tick_params(axis='x', rotation=45)
    fig4.tight_layout()
    add_plot(fig4, scrollable_frame)
