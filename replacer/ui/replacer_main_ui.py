import gradio as gr
from modules import infotext_utils
from replacer.extensions import replacer_extensions
from replacer.ui.tools_ui import AttrDict
from replacer.tools import Pause
from replacer.ui.replacer_tab_ui import getTabUI


try:
    from modules.ui_components import ResizeHandleRow
except:
    ResizeHandleRow = gr.Row



class ReplacerMainUI:
    def __init__(self, isDedicatedPage: bool):
        self.replacerTabUI = None
        self.replacerVideoTabUI = None
        self.components = AttrDict()
        self.init_tab(isDedicatedPage)

    def init_tab(self, isDedicatedPage: bool):
        comp = AttrDict()
        self.replacerTabUI = getTabUI(comp, isDedicatedPage)
        self.components = comp


    def getReplacerTabUI(self):
        return self.replacerTabUI


replacerMainUI: ReplacerMainUI = None
replacerMainUI_dedicated: ReplacerMainUI = None

registered_param_bindings_main_ui = []

def initMainUI(*args):
    global replacerMainUI, replacerMainUI_dedicated, registered_param_bindings_main_ui
    lenBefore = len(infotext_utils.registered_param_bindings)
    try:
        replacer_extensions.initAllScripts()
        replacerMainUI = ReplacerMainUI(isDedicatedPage=False)
        replacerMainUI_dedicated = ReplacerMainUI(isDedicatedPage=True)
    finally:
        replacer_extensions.restoreTemporaryChangedThings()

    registered_param_bindings_main_ui = infotext_utils.registered_param_bindings[lenBefore:]


def reinitMainUIAfterUICreated():
    replacer_extensions.reinitAllScriptsAfterUICreated()
    infotext_utils.registered_param_bindings += registered_param_bindings_main_ui

