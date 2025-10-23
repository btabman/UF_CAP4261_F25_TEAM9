import gradio as gr

def greet(name):
    return f"Hello, {name}!"

def add(x, y):
    return x + y

greet_ui = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text",
    title="Greeting App",
    flagging_mode="manual",         
    flagging_options=["Wrong", "Other"]
)

add_ui = gr.Interface(
    fn=add,
    inputs=[gr.Number(), gr.Number()],
    outputs="number",
    title="Adder App",
    flagging_mode="manual",  
    flagging_options=["Bad result", "Unexpected"]
)

demo = gr.TabbedInterface(
    [greet_ui, add_ui],
    tab_names=["Greet", "Add"]
)

demo.launch()
