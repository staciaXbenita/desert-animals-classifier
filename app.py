from fastai.vision.all import load_learner
import gradio as gr

learn = load_learner("model.pkl")

categories = ("Fennec Fox", "Prairie Dog", "Sand Cat", "Meerkat")

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))


demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", height=192, width=192),
    outputs=gr.Label(num_top_classes=len(categories)),
    # outputs = gr.outputs.label()
    examples=["test_desert_animals_classifier/fennec_fox.jpg", "test_desert_animals_classifier/meerkat.jpg", "test_desert_animals_classifier/prariedog.jpg", "test_desert_animals_classifier/sand_cat.jpg"],
    title="Desert Animal Classifier",
)

if __name__ == "__main__":
    demo.launch()