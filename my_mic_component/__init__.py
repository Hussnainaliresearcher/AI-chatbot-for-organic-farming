import os
import streamlit.components.v1 as components

# Declare your custom component
_component_func = components.declare_component(
    "my_mic_component",
    path=os.path.join(os.path.dirname(__file__), "frontend")
)

def mic_input(key=None):
    """Returns text from mic/chat input"""
    return _component_func(key=key, default="")
