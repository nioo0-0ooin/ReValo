from os import path
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification
from PIL import Image
import torch
import serial
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

processor = AutoImageProcessor.from_pretrained(
    "Ecobot"
)

model = AutoModelForImageClassification.from_pretrained(
    "Ecobot"
)

# =========================================================
# REVALO CONFIG
# =========================================================

RECOVERY_VALUE = {
    "bio": 2,
    "paper_cardboard": 3,
    "plastic_glass_metal": 5,
}

URBAN_MINING_POTENTIAL = {
    "bio": "Low",
    "paper_cardboard": "Medium",
    "plastic_glass_metal": "High",
}

CATEGORY_DISPLAY_NAME = {
    "bio": "Bio Waste",
    "paper_cardboard": "Paper / Cardboard",
    "plastic_glass_metal": "Plastic / Glass / Metal",
}

# Session-only analytics (in-memory, resets each run — no database)
session_stats = {
    "items_processed": 0,
    "bio": 0,
    "paper_cardboard": 0,
    "plastic_glass_metal": 0,
    "total_value": 0,
    "confidences": [],
}

# =========================================================
# GUI CONFIG
# =========================================================

WINDOW_NAME = "ReValo - Waste to Wealth"
WIDTH, HEIGHT = 1000, 650
FONT = cv2.FONT_HERSHEY_SIMPLEX

COLOR_BG = (30, 30, 30)
COLOR_PANEL = (45, 45, 45)
COLOR_ACCENT = (60, 179, 113)   # green
COLOR_ACCENT2 = (244, 133, 66)  # accent (BGR)
COLOR_TEXT = (255, 255, 255)
COLOR_MUTED = (180, 180, 180)
COLOR_WARN = (0, 100, 220)

current_buttons = []   # [(rect, action), ...] rebuilt every frame
pending_action = None  # set by mouse callback, consumed by the loop


def mouse_callback(event, x, y, flags, param):
    global pending_action
    if event == cv2.EVENT_LBUTTONDOWN:
        for rect, action in current_buttons:
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                pending_action = action
                break


def pop_pending_action():
    global pending_action
    a = pending_action
    pending_action = None
    return a


def window_closed():
    """True if the user closed the window with the OS 'X' button."""
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


# =========================================================
# DRAWING HELPERS
# =========================================================

def new_canvas():
    return np.full((HEIGHT, WIDTH, 3), COLOR_BG, dtype=np.uint8)


def add_button(canvas, rect, label, action, color=COLOR_ACCENT):
    global current_buttons
    x1, y1, x2, y2 = rect
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1)
    tsize = cv2.getTextSize(label, FONT, 0.6, 2)[0]
    tx = x1 + (x2 - x1 - tsize[0]) // 2
    ty = y1 + (y2 - y1 + tsize[1]) // 2
    cv2.putText(canvas, label, (tx, ty), FONT, 0.6, COLOR_TEXT, 2)
    current_buttons.append((rect, action))


def fit_image_to_box(cv_img, box_w, box_h):
    """Resize preserving aspect ratio, letterboxed onto a black box."""
    h, w = cv_img.shape[:2]
    scale = min(box_w / w, box_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(cv_img, (new_w, new_h))
    boxed = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    y_off = (box_h - new_h) // 2
    x_off = (box_w - new_w) // 2
    boxed[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return boxed


def draw_report_panel(canvas, x, y, w, h, prediction=None, confidence=None):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_ACCENT, 2)
    cv2.putText(canvas, "RESOURCE REPORT", (x + 15, y + 30), FONT, 0.6, COLOR_ACCENT, 2)

    if prediction is None:
        cv2.putText(canvas, "Awaiting classification...", (x + 15, y + 65), FONT, 0.5, COLOR_MUTED, 1)
        return

    display_name = CATEGORY_DISPLAY_NAME.get(prediction, prediction)
    value = RECOVERY_VALUE.get(prediction, 0)
    potential = URBAN_MINING_POTENTIAL.get(prediction, "Unknown")

    lines = [
        f"Detected : {display_name}",
        f"Category : {prediction}",
        f"Confidence : {round(confidence * 100, 2)}%",
        f"Recovery Value : Rs. {value}",
        f"Mining Potential : {potential}",
        "Status : Resource Recovered",
    ]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (x + 15, y + 65 + i * 28), FONT, 0.52, COLOR_TEXT, 1)


def most_common_category():
    counts = {
        "bio": session_stats["bio"],
        "paper_cardboard": session_stats["paper_cardboard"],
        "plastic_glass_metal": session_stats["plastic_glass_metal"],
    }
    if max(counts.values()) == 0:
        return "None yet"
    return max(counts, key=counts.get)


def compute_recovery_score():
    items = session_stats["items_processed"]
    if items == 0:
        return 0
    avg_confidence = sum(session_stats["confidences"]) / len(session_stats["confidences"])
    throughput_score = min(items, 10) * 3
    confidence_score = avg_confidence * 70
    return min(round(throughput_score + confidence_score), 100)


def update_analytics(prediction, confidence):
    session_stats["items_processed"] += 1
    if prediction in ("bio", "paper_cardboard", "plastic_glass_metal"):
        session_stats[prediction] += 1
        session_stats["total_value"] += RECOVERY_VALUE.get(prediction, 0)
    session_stats["confidences"].append(confidence)


def draw_mini_dashboard(canvas, x, y, w, h):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_ACCENT2, 2)
    cv2.putText(canvas, "LIVE ANALYTICS", (x + 15, y + 28), FONT, 0.58, COLOR_ACCENT2, 2)
    cv2.putText(canvas, f"Items Processed: {session_stats['items_processed']}", (x + 15, y + 58), FONT, 0.48, COLOR_TEXT, 1)
    cv2.putText(canvas, f"Total Value: Rs. {session_stats['total_value']}", (x + 15, y + 84), FONT, 0.48, COLOR_TEXT, 1)
    cv2.putText(canvas, f"Recovery Score: {compute_recovery_score()}/100", (x + 15, y + 110), FONT, 0.48, COLOR_TEXT, 1)

    bar_x = x + 15
    bar_y = y + 135
    cats = [("Bio", session_stats['bio']), ("Paper", session_stats['paper_cardboard']), ("Plastic", session_stats['plastic_glass_metal'])]
    max_count = max(1, *[c for _, c in cats])
    bar_max_w = w - 130
    for i, (label, count) in enumerate(cats):
        yy = bar_y + i * 28
        if yy + 20 > y + h - 5:
            break
        cv2.putText(canvas, label, (bar_x, yy + 15), FONT, 0.45, COLOR_TEXT, 1)
        bw = int((count / max_count) * bar_max_w)
        cv2.rectangle(canvas, (bar_x + 60, yy), (bar_x + 60 + bw, yy + 18), COLOR_ACCENT, -1)
        cv2.putText(canvas, str(count), (bar_x + 65 + bw, yy + 15), FONT, 0.45, COLOR_TEXT, 1)


def draw_category_bar_chart(canvas, x, y, w, h, title="Category Breakdown"):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_ACCENT2, 2)
    cv2.putText(canvas, title, (x + 15, y + 28), FONT, 0.58, COLOR_ACCENT2, 2)

    cats = [
        ("Bio", session_stats['bio']),
        ("Paper/Cardboard", session_stats['paper_cardboard']),
        ("Plastic/Glass/Metal", session_stats['plastic_glass_metal']),
    ]
    max_count = max(1, *[c for _, c in cats])
    bar_area_h = h - 90
    n = len(cats)
    slot_w = (w - 40) // n
    bar_w = min(90, slot_w - 30)
    base_y = y + h - 30

    for i, (label, count) in enumerate(cats):
        bx = x + 30 + i * slot_w + (slot_w - bar_w) // 2
        bh = int((count / max_count) * bar_area_h)
        cv2.rectangle(canvas, (bx, base_y - bh), (bx + bar_w, base_y), COLOR_ACCENT, -1)
        cv2.putText(canvas, str(count), (bx + bar_w // 2 - 8, base_y - bh - 10), FONT, 0.55, COLOR_TEXT, 2)
        cv2.putText(canvas, label, (x + 30 + i * slot_w, base_y + 22), FONT, 0.45, COLOR_TEXT, 1)


# =========================================================
# SCREENS
# =========================================================

def render_menu():
    global current_buttons
    current_buttons = []
    canvas = new_canvas()

    cv2.putText(canvas, "ReValo", (WIDTH // 2 - 100, 110), FONT, 2.0, COLOR_ACCENT, 4)
    cv2.putText(canvas, "Waste-to-Wealth Resource Management & Urban Mining Network",
                (WIDTH // 2 - 225, 150), FONT, 0.55, COLOR_TEXT, 1)
    cv2.putText(canvas, "Status: Ready", (WIDTH // 2 - 65, 180), FONT, 0.55, COLOR_ACCENT, 1)

    btn_w, btn_h, gap = 320, 60, 18
    start_y = 230
    items = [
        ("1. Live Detection", "live", COLOR_ACCENT),
        ("2. Image Analysis", "image", COLOR_ACCENT),
        ("3. Analytics Dashboard", "dashboard", COLOR_ACCENT),
        ("4. Demo Mode", "demo", COLOR_ACCENT),
        ("5. Exit", "exit", COLOR_WARN),
    ]
    for i, (label, action, color) in enumerate(items):
        x1 = WIDTH // 2 - btn_w // 2
        y1 = start_y + i * (btn_h + gap)
        add_button(canvas, (x1, y1, x1 + btn_w, y1 + btn_h), label, action, color)

    cv2.putText(canvas, "Click a button, or press 1-5 on the keyboard",
                (WIDTH // 2 - 160, HEIGHT - 25), FONT, 0.5, COLOR_MUTED, 1)
    return canvas


def run_live_detection_loop():
    global current_buttons
    cap = cv2.VideoCapture(0)
    last_prediction, last_confidence = None, None
    result = "menu"
    try:
        while True:
            if window_closed():
                result = "exit"
                break

            ret, frame = cap.read()
            if not ret:
                break

            disp_frame = fit_image_to_box(frame, 600, 450)

            canvas = new_canvas()
            current_buttons = []
            cv2.putText(canvas, "LIVE DETECTION", (20, 35), FONT, 0.75, COLOR_ACCENT, 2)
            canvas[60:60 + 450, 20:20 + 600] = disp_frame

            draw_report_panel(canvas, 640, 60, 340, 260, last_prediction, last_confidence)
            draw_mini_dashboard(canvas, 640, 330, 340, 180)

            add_button(canvas, (20, 525, 220, 575), "Classify (C)", "classify")
            add_button(canvas, (240, 525, 440, 575), "Back to Menu (M)", "menu", COLOR_ACCENT2)
            add_button(canvas, (460, 525, 600, 575), "Quit (Q)", "exit", COLOR_WARN)

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            action = pop_pending_action()

            if key == ord('c') or action == "classify":
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                prediction, confidence = classify_image(image)
                send_to_bin(prediction)
                update_analytics(prediction, confidence)
                last_prediction, last_confidence = prediction, confidence
            elif key == ord('m') or action == "menu":
                result = "menu"
                break
            elif key == ord('q') or action == "exit":
                result = "exit"
                break
    finally:
        cap.release()
    return result


def run_image_mode():
    global current_buttons

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image for ReValo Image Analysis",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()

    if not file_path:
        return "menu"

    image = Image.open(file_path)
    prediction, confidence = classify_image(image)
    send_to_bin(prediction)
    update_analytics(prediction, confidence)

    cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    boxed_img = fit_image_to_box(cv_img, 600, 450)

    while True:
        if window_closed():
            return "exit"

        canvas = new_canvas()
        current_buttons = []
        cv2.putText(canvas, "IMAGE ANALYSIS", (20, 35), FONT, 0.75, COLOR_ACCENT, 2)
        canvas[60:60 + 450, 20:20 + 600] = boxed_img

        draw_report_panel(canvas, 640, 60, 340, 260, prediction, confidence)
        draw_mini_dashboard(canvas, 640, 330, 340, 180)

        add_button(canvas, (240, 525, 440, 575), "Back to Menu (M)", "menu", COLOR_ACCENT2)
        add_button(canvas, (460, 525, 600, 575), "Quit (Q)", "exit", COLOR_WARN)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(30) & 0xFF
        action = pop_pending_action()

        if key == ord('m') or action == "menu":
            return "menu"
        elif key == ord('q') or action == "exit":
            return "exit"


def run_dashboard_loop():
    global current_buttons
    while True:
        if window_closed():
            return "exit"

        canvas = new_canvas()
        current_buttons = []
        cv2.putText(canvas, "ANALYTICS DASHBOARD", (20, 40), FONT, 0.85, COLOR_ACCENT, 2)

        stats = [
            ("Items Processed", session_stats['items_processed']),
            ("Total Value (Rs.)", session_stats['total_value']),
            ("Recovery Score", f"{compute_recovery_score()}/100"),
        ]
        card_w, card_h = 300, 100
        for i, (label, val) in enumerate(stats):
            x1 = 20 + i * (card_w + 20)
            y1 = 70
            cv2.rectangle(canvas, (x1, y1), (x1 + card_w, y1 + card_h), COLOR_PANEL, -1)
            cv2.rectangle(canvas, (x1, y1), (x1 + card_w, y1 + card_h), COLOR_ACCENT, 2)
            cv2.putText(canvas, str(val), (x1 + 20, y1 + 50), FONT, 1.0, COLOR_TEXT, 2)
            cv2.putText(canvas, label, (x1 + 20, y1 + 80), FONT, 0.48, COLOR_MUTED, 1)

        draw_category_bar_chart(canvas, 20, 210, 620, 300)

        cv2.putText(canvas, f"Most Common Waste: {most_common_category()}", (660, 240), FONT, 0.55, COLOR_TEXT, 1)
        common = most_common_category()
        if common != "None yet":
            potential = URBAN_MINING_POTENTIAL.get(common, "Unknown")
            cv2.putText(canvas, f"Urban Mining Potential: {potential}", (660, 275), FONT, 0.55, COLOR_TEXT, 1)

        add_button(canvas, (20, 550, 220, 600), "Back to Menu (M)", "menu", COLOR_ACCENT2)
        add_button(canvas, (240, 550, 440, 600), "Quit (Q)", "exit", COLOR_WARN)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(30) & 0xFF
        action = pop_pending_action()

        if key == ord('m') or action == "menu":
            return "menu"
        elif key == ord('q') or action == "exit":
            return "exit"


def run_demo_loop():
    """
    Expo Demonstration Mode: live feed + full analytics side by side,
    updating on every classification. Same detection pipeline and
    hardware calls as Live Detection — purely a bigger visual layout.
    """
    global current_buttons
    cap = cv2.VideoCapture(0)
    last_prediction, last_confidence = None, None
    result = "menu"
    try:
        while True:
            if window_closed():
                result = "exit"
                break

            ret, frame = cap.read()
            if not ret:
                break

            disp_frame = fit_image_to_box(frame, 480, 360)

            canvas = new_canvas()
            current_buttons = []
            cv2.putText(canvas, "EXPO DEMONSTRATION MODE", (20, 30), FONT, 0.75, COLOR_ACCENT, 2)
            canvas[50:50 + 360, 20:20 + 480] = disp_frame

            draw_report_panel(canvas, 520, 50, 460, 200, last_prediction, last_confidence)
            draw_category_bar_chart(canvas, 520, 260, 460, 240, title="Live Analytics")

            add_button(canvas, (20, 430, 220, 480), "Classify (C)", "classify")
            add_button(canvas, (240, 430, 440, 480), "Back to Menu (M)", "menu", COLOR_ACCENT2)
            add_button(canvas, (20, 500, 220, 550), "Quit (Q)", "exit", COLOR_WARN)

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            action = pop_pending_action()

            if key == ord('c') or action == "classify":
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                prediction, confidence = classify_image(image)
                send_to_bin(prediction)
                update_analytics(prediction, confidence)
                last_prediction, last_confidence = prediction, confidence
            elif key == ord('m') or action == "menu":
                result = "menu"
                break
            elif key == ord('q') or action == "exit":
                result = "exit"
                break
    finally:
        cap.release()
    return result


# =========================================================
# CORE MODEL / HARDWARE LOGIC 
# =========================================================

def classify_image(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    predicted_class = outputs.logits.argmax(-1).item()

    print("Prediction:", model.config.id2label[predicted_class])
    print(outputs.logits)

    prediction = model.config.id2label[predicted_class]
    prediction = prediction.lower()

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence, predicted_class = torch.max(probs, dim=-1)

    print("Prediction:", model.config.id2label[predicted_class.item()])
    print("Confidence:", round(confidence.item() * 100, 2), "%")

    if confidence < 0.5:
        print("The model is not confident about the prediction.")

    return prediction, confidence.item()


def send_to_bin(prediction):
    print("Sending prediction to Arduino...")
    arduino = serial.Serial('COM3', 9600)
    time.sleep(5)
    print("Raw:", repr(prediction))
    if prediction == "bio":
        print("sent 1")
        arduino.write(b'1')
    elif prediction == "paper_cardboard":
        print("sent 2")
        arduino.write(b'2')
    elif prediction == "plastic_glass_metal":
        print("sent 3")
        arduino.write(b'3')


# =========================================================
# MAIN APP LOOP
# =========================================================

def main():
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    current_screen = "menu"
    while True:
        if current_screen == "menu":
            canvas = render_menu()
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(30) & 0xFF
            action = pop_pending_action()

            if window_closed():
                break
            elif key == ord('q'):
                break
            elif key == ord('1') or action == "live":
                current_screen = "live"
            elif key == ord('2') or action == "image":
                current_screen = "image"
            elif key == ord('3') or action == "dashboard":
                current_screen = "dashboard"
            elif key == ord('4') or action == "demo":
                current_screen = "demo"
            elif key == ord('5') or action == "exit":
                break

        elif current_screen == "live":
            current_screen = run_live_detection_loop()
        elif current_screen == "image":
            current_screen = run_image_mode()
        elif current_screen == "dashboard":
            current_screen = run_dashboard_loop()
        elif current_screen == "demo":
            current_screen = run_demo_loop()
        elif current_screen == "exit":
            break

    cv2.destroyAllWindows()
    print("\nThank you for visiting ReValo.")
    print("Today's Waste = Tomorrow's Resource Reserve.")


if __name__ == "__main__":
    main()
