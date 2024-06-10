import os
from pathlib import Path
import gradio as gr
from modules import shared

EXT_NAME = None
EXT_ROOT_DIRECTORY = str(Path(__file__).parent.parent.absolute())
try:
    extNameFilePath = os.path.join(EXT_ROOT_DIRECTORY, 'ExtensionName.txt')
    if os.path.isfile(extNameFilePath):
        with open(extNameFilePath, 'r') as f:
            EXT_NAME = f.readline().strip()
except :
    EXT_NAME = None

if EXT_NAME is None or EXT_NAME == "":
    EXT_NAME = os.environ.get("SD_WEBUI_REPLACER_EXTENSION_NAME", "Replacer").strip()

EXT_NAME_LOWER = EXT_NAME.lower().replace(' ', '_')

defaultOutputDirectory = os.path.join('output', EXT_NAME_LOWER)


def getSaveDir():
    return shared.opts.data.get(EXT_NAME_LOWER + "_save_dir", defaultOutputDirectory)


def getDedicatedPagePath():
    return f'/{EXT_NAME_LOWER}-dedicated'


def needAutoUnloadModels():
    opt = shared.opts.data.get(EXT_NAME_LOWER + "_always_unload_models", 'Automatic')

    if opt == 'Enabled':
        return True
    if opt == 'Disabled':
        return False
    if opt == 'Only SDXL':
        return shared.sd_model.is_sdxl

    return shared.cmd_opts.lowvram or shared.cmd_opts.medvram or (shared.sd_model.is_sdxl and shared.cmd_opts.medvram_sdxl)


def doNotShowUnloadButton():
    opt = shared.opts.data.get(EXT_NAME_LOWER + "_always_unload_models", 'Automatic')

    if opt == 'Enabled':
        return True
    if opt == 'Disabled':
        return False
    if opt == 'Only SDXL':
        return False

    return shared.cmd_opts.lowvram or shared.cmd_opts.medvram



def useCpuForDetection():
    opt = shared.opts.data.get(EXT_NAME_LOWER + "_use_cpu_for_detection", False)
    return opt

def useFastDilation():
    opt = shared.opts.data.get(EXT_NAME_LOWER + "_fast_dilation", True)
    return opt


defaultMaskColor = '#84FF9A'

def getMaskColorStr():
    opt = shared.opts.data.get(EXT_NAME_LOWER + "_mask_color", defaultMaskColor)
    return opt


detectionPromptExamples_defaults = [
            "background",
            "hairstyle",
            "t-shirt",
        ]

avoidancePromptExamples_defaults = [
            "",
            "face",
            "hairstyle",
            "clothes",
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
    "<lora:sdxl_lightning_2step_lora:1>",
    " ",
]



def getDetectionPromptExamples():
    res : str = shared.opts.data.get(EXT_NAME_LOWER + "_detection_prompt_examples", "")
    if res == "":
        return detectionPromptExamples_defaults
    else:
        return res.split("\n")
    
def getAvoidancePromptExamples():
    res : str = shared.opts.data.get(EXT_NAME_LOWER + "_avoidance_prompt_examples", "")
    if res == "":
        return avoidancePromptExamples_defaults
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

def needHideAnimateDiffAccordions():
    res : bool = shared.opts.data.get(EXT_NAME_LOWER + "_hide_animatediff_accordions", False)
    return res

def needHideReplacerScript():
    res : bool = shared.opts.data.get(EXT_NAME_LOWER + "_hide_replacer_script", False)
    return res


def getDetectionPromptExamplesNumber():
    res : int = shared.opts.data.get(EXT_NAME_LOWER + "_examples_per_page_for_detection_prompt", 10)
    return res

def getAvoidancePromptExamplesNumber():
    res : int = shared.opts.data.get(EXT_NAME_LOWER + "_examples_per_page_for_avoidance_prompt", 10)
    return res

def getPositivePromptExamplesNumber():
    res : int = shared.opts.data.get(EXT_NAME_LOWER + "_examples_per_page_for_positive_prompt", 10)
    return res

def getNegativePromptExamplesNumber():
    res : int = shared.opts.data.get(EXT_NAME_LOWER + "_examples_per_page_for_negative_prompt", 10)
    return res


if not hasattr(shared.OptionInfo, 'needs_reload_ui'): # webui 1.5.0
    shared.OptionInfo.needs_reload_ui = lambda self: self.info('requires Reload UI')
    shared.OptionInfo.needs_restart = lambda self: self.info('requires restart')


section = (EXT_NAME_LOWER, EXT_NAME)
preloaded_options = {
    EXT_NAME_LOWER + "_default_extra_includes": shared.OptionInfo(
        ["script"],
        "Defaults for Extra include in gallery",
        gr.CheckboxGroup,
        {
            'choices' : ["mask", "box", "cutted", "preview", "script"],
        },
        section=section,
    ).needs_reload_ui(),
}
shared.options_templates.update(shared.options_section(section, preloaded_options))


def on_ui_settings():
    global section

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
            f"Hide Segment Anything accordions in txt2img and img2img tabs. Useful if you installed it only for {EXT_NAME}",
            gr.Checkbox,
            section=section,
        ).needs_reload_ui()
    )

    shared.opts.add_option(
        EXT_NAME_LOWER + "_hide_animatediff_accordions",
        shared.OptionInfo(
            False,
            f"Hide AnimateDiff accordions in txt2img and img2img tabs. Useful if you installed it only for {EXT_NAME}",
            gr.Checkbox,
            section=section,
        ).needs_reload_ui()
    )

    shared.opts.add_option(
        EXT_NAME_LOWER + "_hide_replacer_script",
        shared.OptionInfo(
            False,
            f"Hide {EXT_NAME} accordions in txt2img and img2img tabs",
            gr.Checkbox,
            section=section,
        ).needs_reload_ui()
    )


    shared.opts.add_option(
        EXT_NAME_LOWER + "_always_unload_models",
        shared.OptionInfo(
            'Automatic',
            "Always unload detection models after generation",
            gr.Radio,
            {
                'choices' : ['Automatic', 'Enabled', 'Only SDXL', 'Disabled'],
            },
            section=section,
        ).info("Significally increases detection time but reduces vram usage. "\
               "Automatic means enable only for --lowvram and --medvram mode. "\
            )
    )


    shared.opts.add_option(
        EXT_NAME_LOWER + "_use_cpu_for_detection",
        shared.OptionInfo(
            False,
            "Use CPU for detection (SAM + Dino). For AMD Radeon and Intel ARC or if you don't have enought vram",
            gr.Checkbox,
            section=section,
        ).needs_restart()
    )


    shared.opts.add_option(
        EXT_NAME_LOWER + "_fast_dilation",
        shared.OptionInfo(
            True,
            "Use fast mask dilation (Squarer)",
            gr.Checkbox,
            section=section,
        ).needs_restart()
    )


    shared.opts.add_option(
        EXT_NAME_LOWER + "_mask_color",
        shared.OptionInfo(
            defaultMaskColor,
            "Color for mask in preview (fast dilation) and default for Custom mask and Avoidance mask",
            gr.ColorPicker,
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
        EXT_NAME_LOWER + "_avoidance_prompt_examples",
        shared.OptionInfo(
            "",
            "Override Avoidance prompt examples",
            gr.Textbox,
            {
                "lines" : 2,
                "placeholder" : "\n".join(avoidancePromptExamples_defaults),
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
        EXT_NAME_LOWER + "_examples_per_page_for_detection_prompt",
        shared.OptionInfo(
            10,
            "Override number of examples per pages for detection prompt",
            gr.Number,
            section=section,
        ).needs_reload_ui().info('Set 0 to hide')
    )

    shared.opts.add_option(
        EXT_NAME_LOWER + "_examples_per_page_for_avoidance_prompt",
        shared.OptionInfo(
            10,
            "Override number of examples per pages for avoidance prompt",
            gr.Number,
            section=section,
        ).needs_reload_ui().info('Set 0 to hide')
    )

    shared.opts.add_option(
        EXT_NAME_LOWER + "_examples_per_page_for_positive_prompt",
        shared.OptionInfo(
            10,
            "Override number of examples per pages for positive prompt",
            gr.Number,
            section=section,
        ).needs_reload_ui().info('Set 0 to hide')
    )

    shared.opts.add_option(
        EXT_NAME_LOWER + "_examples_per_page_for_negative_prompt",
        shared.OptionInfo(
            10,
            "Override number of examples per pages for negative prompt",
            gr.Number,
            section=section,
        ).needs_reload_ui().info('Set 0 to hide')
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

