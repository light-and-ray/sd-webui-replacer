import gradio as gr
import modules
from modules import shared, sd_samplers
from modules.ui_common import create_refresh_button
from replacer.options import getHiresFixPositivePromptSuffixExamples, doNotShowUnloadButton
from replacer.extensions import replacer_extensions
from replacer.ui.tools_ui import IS_WEBUI_1_5, IS_WEBUI_1_9, AttrDict



def getHiresFixCheckpoints():
    if IS_WEBUI_1_5:
        return ["Use same checkpoint"] + modules.sd_models.checkpoint_tiles()
    else:
        return ["Use same checkpoint"] + modules.sd_models.checkpoint_tiles(use_short=False)



def makeHiresFixOptions(comp: AttrDict):
    with gr.Accordion("HiresFix options", open=False):
        with gr.Tabs():
            with gr.Tab('General'):
                with gr.Row():
                    comp.hf_upscaler = gr.Dropdown(
                        value="ESRGAN_4x",
                        choices=[x.name for x in shared.sd_upscalers],
                        label="Upscaler",
                    )

                    comp.hf_steps = gr.Slider(
                        label='Hires steps',
                        value=4,
                        step=1,
                        minimum=0,
                        maximum=150,
                        elem_id="replacer_hf_steps"
                    )

                with gr.Row():
                    comp.hf_denoise = gr.Slider(
                        label='Hires Denoising',
                        value=0.35,
                        step=0.01,
                        minimum=0.0,
                        maximum=1.0,
                        elem_id="replacer_hf_denoise",
                    )

                with gr.Row():
                    comp.hf_size_limit = gr.Slider(
                        label='Limit render size',
                        value=1800,
                        step=1,
                        minimum=700,
                        maximum=10000,
                        elem_id="replacer_hf_size_limit",
                    )

                    comp.hf_above_limit_upscaler = gr.Dropdown(
                        value="Lanczos",
                        choices=[x.name for x in shared.sd_upscalers],
                        label="Above limit upscaler",
                    )

                with gr.Row():
                    comp.hf_extra_mask_expand = gr.Slider(
                        label='Extra mask expand',
                        value=5,
                        step=1,
                        minimum=0,
                        maximum=200,
                        elem_id="replacer_hf_extra_mask_expand",
                    )

                    comp.hf_extra_inpaint_padding = gr.Slider(label='Extra inpaint padding',
                        value=250, elem_id="replacer_hf_extra_inpaint_padding",
                        minimum=0, maximum=3000, step=1)

                    comp.hf_extra_mask_blur = gr.Slider(label='Extra mask blur',
                        value=2, elem_id="replacer_hf_extra_mask_blur",
                        minimum=0, maximum=150, step=1)

                with gr.Row():
                    comp.hf_randomize_seed = gr.Checkbox(
                        label='Randomize seed for hires fix',
                        value=True,
                        elem_id="replacer_hf_randomize_seed",
                    )

            with gr.Tab('Advanced'):
                with gr.Row():
                    comp.hf_sampler = gr.Dropdown(
                        label='Hires sampling method',
                        elem_id="replacer_hf_sampler",
                        choices=["Use same sampler"] + sd_samplers.visible_sampler_names(),
                        value="Use same sampler"
                    )
                    if IS_WEBUI_1_9:
                        from modules import sd_schedulers
                        comp.hf_scheduler = gr.Dropdown(
                            label='Hires schedule type',
                            elem_id="replacer_hf_scheduler",
                            choices=["Use same scheduler"] + [x.label for x in sd_schedulers.schedulers],
                            value="Use same scheduler"
                        )
                    else:
                        comp.hf_scheduler = gr.Textbox("", visible=False)

                    comp.hf_cfg_scale = gr.Slider(
                        label='Hires CFG Scale',
                        value=1.0,
                        step=0.5,
                        minimum=1.0,
                        maximum=30.0,
                        elem_id="replacer_hf_cfg_scale"
                    )

                with gr.Row():
                    comp.hf_unload_detection_models = gr.Checkbox(
                        label='Unload detection models before hires fix',
                        value=True,
                        elem_id="replacer_hf_unload_detection_models",
                    )
                    if doNotShowUnloadButton():
                        comp.hf_unload_detection_models.visible = False

                with gr.Row():
                    placeholder = None
                    placeholder = getHiresFixPositivePromptSuffixExamples()[0]

                    comp.hfPositivePromptSuffix = gr.Textbox(
                        label="Suffix for positive prompt",
                        show_label=True,
                        lines=1,
                        elem_classes=["hfPositivePromptSuffix"],
                        placeholder=placeholder,
                        elem_id="replacer_hfPositivePromptSuffix",
                    )

                    gr.Examples(
                        examples=getHiresFixPositivePromptSuffixExamples(),
                        inputs=comp.hfPositivePromptSuffix,
                        label="",
                        elem_id="replacer_hfPositivePromptSuffix_examples",
                    )

                with gr.Row():
                    comp.hf_positivePrompt = gr.Textbox(label="Override positive prompt",
                            show_label=True,
                            lines=1,
                            elem_classes=["positivePrompt"],
                            placeholder='leave empty to use the same prompt',
                            elem_id="replacer_hf_positivePrompt")

                    comp.hf_negativePrompt = gr.Textbox(label="Override negative prompt",
                            show_label=True,
                            lines=1,
                            elem_classes=["negativePrompt"],
                            placeholder='leave empty to use the same prompt',
                            elem_id="replacer_hf_negativePrompt")

                with gr.Row():
                    comp.hf_sd_model_checkpoint = gr.Dropdown(label='Hires checkpoint',
                        elem_id="replacer_hf_sd_model_checkpoint",
                        choices=getHiresFixCheckpoints(), value="Use same checkpoint")
                    create_refresh_button(comp.hf_sd_model_checkpoint, modules.sd_models.list_models,
                        lambda: {"choices": getHiresFixCheckpoints()}, "replacer_hf_sd_model_checkpoint")

                    comp.hf_disable_cn = gr.Checkbox(
                        label='Disable ControlNet while hires fix',
                        value=True,
                        elem_id="replacer_hf_disable_cn",
                    )
                    if not replacer_extensions.controlnet.SCRIPT:
                        comp.hf_disable_cn.visible = False

                with gr.Row():
                    comp.hf_soft_inpaint = gr.Radio(label='Soft inpainting for hires fix',
                        choices=['Same', 'Enable', 'Disable'],
                        value='Same', type="value", elem_id="replacer_hf_soft_inpaint")
                    if not replacer_extensions.soft_inpainting.SCRIPT:
                        comp.hf_soft_inpaint.visible = False

