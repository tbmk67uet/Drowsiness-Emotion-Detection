import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def view_results_from_csv(tab_widget, csv_path):
    for widget in tab_widget.winfo_children():
        widget.destroy()

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

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle("📊 Kết quả phân tích cảm xúc & độ tập trung", fontsize=14)

    # 1. Cảm xúc theo thời gian
    axs[0, 0].plot(df['Timestamp'], df['Emotion'], marker='o', linestyle='-', alpha=0.7)
    axs[0, 0].set_title("Cảm xúc theo thời gian")
    axs[0, 0].tick_params(axis='x', rotation=45)
    axs[0, 0].grid(True)

    # 2. Tần suất cảm xúc
    df['Emotion'].value_counts().plot(kind='bar', ax=axs[0, 1], color='skyblue')
    axs[0, 1].set_title("Tần suất cảm xúc")
    axs[0, 1].set_ylabel("Số lần")

    # 3. Cảnh báo buồn ngủ/ngáp
    axs[1, 0].plot(df['Timestamp'], df['DrowsinessAlert'], label='Buồn ngủ', color='red')
    axs[1, 0].plot(df['Timestamp'], df['YawnAlert'], label='Ngáp', color='orange')
    axs[1, 0].set_title("Cảnh báo buồn ngủ / ngáp")
    axs[1, 0].legend()
    axs[1, 0].tick_params(axis='x', rotation=45)
    axs[1, 0].set_yticks([0, 1])
    axs[1, 0].set_yticklabels(['Không', 'Có'])

    # 4. Điểm tập trung
    axs[1, 1].plot(df['Timestamp'], df['FocusScore'], color='green', marker='o')
    axs[1, 1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axs[1, 1].set_title(f"Điểm tập trung (Tổng: {total_score}, TB: {avg_score:.2f})")
    axs[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Gắn figure vào tkinter tab
    canvas = FigureCanvasTkAgg(fig, master=tab_widget)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
