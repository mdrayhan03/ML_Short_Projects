from kivy.lang import Builder
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import StringProperty, BooleanProperty
from plyer import filechooser
from kivy.clock import Clock
import numpy as np
import joblib, os
from PIL import Image
from kivy.metrics import dp

Builder.load_string('''
<Main>:
    canvas.before:
        Color:
            rgba: 0.95, 0.95, 0.95, 1
        Rectangle:
            pos: self.pos
            size: self.size

    Image:
        source: root.img_src
        size_hint: 0.5, 0.5
        pos_hint: {"center_x": 0.5, "top": 0.98}
        allow_stretch: True
        keep_ratio: True

    Button:
        text: "Load Image"
        size_hint: 0.4, None
        height: dp(45)
        pos_hint: {"center_x": 0.5, "y": 0.3}
        background_color: (0.3, 0.6, 0.9, 1)
        color: (1, 1, 1, 1)
        font_size: dp(16)
        on_press: root.load_image()

    Label:
        text: "Prediction of your image:"
        size_hint: 1, None
        height: dp(30)
        pos_hint: {"center_x": 0.5, "y": 0.25}
        color: (0, 0, 0, 1)
        font_size: dp(16)
        bold: True

    Label:
        id: prediction_label
        text: root.prediction
        size_hint: 1, None
        height: dp(30)
        pos_hint: {"center_x": 0.5, "y": 0.16}
        color: (0.2, 0.2, 0.2, 1)
        font_size: dp(18)

    BoxLayout:
        size_hint: 0.6, None
        height: dp(30)
        spacing: dp(20)
        pos_hint: {"center_x": 0.5, "y": 0.06}

        opacity: 1 if root.show_feedback_buttons else 0
        disabled: not root.show_feedback_buttons

        Button:
            text: "Right"
            background_color: 0, 0.7, 0.2, 1  # green
            color: 1, 1, 1, 1
            font_size: dp(16)
            on_press: root.print_feedback("Right")

        Button:
            text: "Wrong"
            background_color: 0.8, 0, 0, 1  # red
            color: 1, 1, 1, 1
            font_size: dp(16)
            on_press: root.print_feedback("Wrong")
''')

# Load model
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'cat_dog_randomforest_model.pkl')
load_model = joblib.load(model_path)

def predict_image(file_path):
    img = Image.open(file_path)
    img = img.resize((100, 100))
    img = img.convert('L')
    img = np.array(img)
    img = img.flatten().reshape(1, -1)
    predict = load_model.predict(img)
    return "Cat" if predict == 0 else "Dog"

class Main(FloatLayout):
    prediction = StringProperty("No prediction yet.")
    img_src = StringProperty("")
    show_feedback_buttons = BooleanProperty(False)
    _current_file = ""

    def load_image(self):
        file_path = filechooser.open_file(
            title="Select an Image",
            filters=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
        )
        if file_path:
            self.img_src = file_path[0]
            self.prediction = "Loading..."
            self.show_feedback_buttons = False  # hide buttons on new load
            self._current_file = file_path[0]
            Clock.schedule_once(self._predict_after_delay, 3)
        else:
            self.prediction = "No file selected."
            self.show_feedback_buttons = False

    def _predict_after_delay(self, dt):
        self.prediction = predict_image(self._current_file)
        self.show_feedback_buttons = True  # show buttons after prediction

    def print_feedback(self, feedback):
        print(feedback)

class PredictionApp(App):
    def build(self):
        return Main()

if __name__ == "__main__":
    PredictionApp().run()