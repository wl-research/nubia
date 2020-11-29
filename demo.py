from nubia import Nubia
import gradio

nubia = Nubia()


def predict(inp_1, inp_2):
    features = nubia.score(inp_1, inp_2, get_features=True)
    labels = {k: v for k, v in features["features"].items()}
    return {"nubia_score": features["nubia_score"]}, labels

title = "NUBIA"
description = "NeUral Based Interchangeability Assessor. A SoTA evaluation metric for text generation."
inputs = [gradio.inputs.Textbox(label="First Text"), gradio.inputs.Textbox(label="Second Text")]
outputs = [gradio.outputs.Label(label="Interchangeability Score"), gradio.outputs.JSON(label="All Features")]
examples = [
    ["This car is expensive! I can't buy it.", "That automobile costs a fortune! Purchasing it? Impossible!"],
    ["This car is expensive! I can't buy it.", "That automobile costs a good amount. Purchasing it? Totally feasible!"],
    ["The dinner was delicious.", "The dinner did not taste good."]
]
iface = gradio.Interface(fn=predict, inputs=inputs, outputs=outputs, capture_session=True, examples=examples,
                         title=title, description=description, allow_flagging=False)

iface.launch()
