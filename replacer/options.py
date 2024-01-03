import os
from pathlib import Path
import gradio as gr
from modules import shared
from modules import script_callbacks

EXT_NAME = None
try:
    extNameFilePath = str(Path(__file__).parent.parent.absolute()) + '/ExtensionName.txt'
    if os.path.isfile(extNameFilePath):
        with open(extNameFilePath, 'r') as f:
            EXT_NAME = f.readline().strip()
except :
    EXT_NAME = None

if EXT_NAME is None or EXT_NAME == "":
    EXT_NAME = os.environ.get("SD_WEBUI_REPLACER_EXTENTION_NAME", "Replacer").strip()

EXT_NAME_LOWER = EXT_NAME.lower().replace(' ', '_')

defaultOutputDirectory = f"outputs/{EXT_NAME_LOWER}"


def getSaveDir():
    return shared.opts.data.get(EXT_NAME_LOWER + "_save_dir", defaultOutputDirectory)


def needAutoUnloadModels():
    return shared.cmd_opts.lowvram or shared.cmd_opts.medvram


detectionPromptExamples_defaults = [
            "background",
            "hairstyle",
            "t-shirt",
        ]

positvePromptExamples_defaults = [
            "waterfall",
            "photo of blonde girl",
            "photo of girl with red t-shirt",
        ]

negativePromptExamples_defaults = [
    "cartoon, painting, illustration, (worst quality, low quality, normal quality:2)",
    "poor quality, low quality,  low res",
]

hiresFixPositivePromptSuffixExamples_defaults = [
    "<lora:lcm-lora-sdv1-5:1>",
    " ",
]




def getDetectionPromptExamples():
    res : str = shared.opts.data.get(EXT_NAME_LOWER + "_detection_prompt_examples", "")
    if res == "":
        return detectionPromptExamples_defaults
    else:
        return res.split("\n")
    
def getPositivePromptExamples():
    res : str = shared.opts.data.get(EXT_NAME_LOWER + "_positive_prompt_examples", "")
    if res == "":
        return positvePromptExamples_defaults
    else:
        return res.split("\n")
    
def getNegativePromptExamples():
    res : str = shared.opts.data.get(EXT_NAME_LOWER + "_negative_prompt_examples", "")
    if res == "":
        return negativePromptExamples_defaults
    else:
        return res.split("\n")

def getHiresFixPositivePromptSuffixExamples():
    res : str = shared.opts.data.get(EXT_NAME_LOWER + "_hf_positive_prompt_suffix_examples", "")
    if res == "":
        return hiresFixPositivePromptSuffixExamples_defaults
    else:
        return res.split("\n")


def useFirstPositivePromptFromExamples():
    res : bool = shared.opts.data.get(EXT_NAME_LOWER + "_use_first_positive_prompt_from_examples", True)
    return res

def useFirstNegativePromptFromExamples():
    res : bool = shared.opts.data.get(EXT_NAME_LOWER + "_use_first_negative_prompt_from_examples", True)
    return res

def needHideSegmentAnythingAccordions():
    res : bool = shared.opts.data.get(EXT_NAME_LOWER + "_hide_segment_anything_accordions", False)
    return res



def on_ui_settings():
    section = (EXT_NAME_LOWER, EXT_NAME)

    shared.opts.add_option(
        EXT_NAME_LOWER + "_use_first_positive_prompt_from_examples",
        shared.OptionInfo(
            True,
            "Use first positive pormpt form examples, if field is empty",
            gr.Checkbox,
            section=section,
        ).needs_reload_ui()
    )

    shared.opts.add_option(
        EXT_NAME_LOWER + "_use_first_negative_prompt_from_examples",
        shared.OptionInfo(
            True,
            "Use first negative pormpt form examples, if field is empty",
            gr.Checkbox,
            section=section,
        ).needs_reload_ui()
    )

    shared.opts.add_option(
        EXT_NAME_LOWER + "_hide_segment_anything_accordions",
        shared.OptionInfo(
            False,
            f"Hide Segment Anything accordions at txt2img and img2img tabs. Useful if you installed it only for {EXT_NAME}",
            gr.Checkbox,
            section=section,
        ).needs_reload_ui()
    )


    shared.opts.add_option(
        EXT_NAME_LOWER + "_detection_prompt_examples",
        shared.OptionInfo(
            "",
            "Override Detection prompt examples",
            gr.Textbox,
            {
                "lines" : 2,
                "placeholder" : "\n".join(detectionPromptExamples_defaults),
            },
            section=section,
        ).needs_reload_ui()
    )

    shared.opts.add_option(
        EXT_NAME_LOWER + "_positive_prompt_examples",
        shared.OptionInfo(
            "",
            "Override Positive prompt examples",
            gr.Textbox,
            {
                "lines" : 2,
                "placeholder" : "\n".join(positvePromptExamples_defaults),
            },
            section=section,
        ).needs_reload_ui()
    )

    shared.opts.add_option(
        EXT_NAME_LOWER + "_negative_prompt_examples",
        shared.OptionInfo(
            "",
            "Override Negative prompt examples",
            gr.Textbox,
            {
                "lines" : 2,
                "placeholder" : "\n".join(negativePromptExamples_defaults),
            },
            section=section,
        ).needs_reload_ui()
    )

    shared.opts.add_option(
        EXT_NAME_LOWER + "_hf_positive_prompt_suffix_examples",
        shared.OptionInfo(
            "",
            "Override HiresFix suffix for positive prompt examples",
            gr.Textbox,
            {
                "lines" : 2,
                "placeholder" : "\n".join(hiresFixPositivePromptSuffixExamples_defaults),
            },
            section=section,
        ).needs_reload_ui()
    )

    shared.opts.add_option(
        EXT_NAME_LOWER + "_save_dir",
        shared.OptionInfo(
            defaultOutputDirectory,
            f"{EXT_NAME} save directory",
            gr.Textbox,
            {
                "visible": not shared.cmd_opts.hide_ui_dir_config,
            },
            section=section,
        )
    )


script_callbacks.on_ui_settings(on_ui_settings)
