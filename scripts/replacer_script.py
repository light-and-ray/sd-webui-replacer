import copy
import gradio as gr
from modules import scripts, scripts_postprocessing, errors, ui_settings, shared
from modules.processing import Processed, StableDiffusionProcessingTxt2Img
from replacer.options import EXT_NAME, needHideReplacerScript
from replacer.ui import replacer_tab_ui
from replacer.generation_args import GenerationArgs, HiresFixArgs, DUMMY_ANIMATEDIFF_ARGS
from replacer.extensions import replacer_extensions
from replacer.tools import prepareMask
from replacer.generate import generate
from replacer.ui.tools_ui import prepareExpectedUIBehavior

if hasattr(scripts_postprocessing.ScriptPostprocessing, 'process_firstpass'):  # webui >= 1.7
    from modules.ui_components import InputAccordion
else:
    InputAccordion = None



class ReplacerScript(scripts.Script):
    def __init__(self):
        self.gArgs: GenerationArgs = None
        self.extra_includes: list = None
        self.enable: bool = None
        self.save_originals: bool = None
        self.force_override_sd_model: bool = None
        self.force_sd_model_checkpoint: str = None
        self.save_samples: bool = None
        self.override_seed: bool = None
        self.append_positive_prompt: bool = None

    def title(self):
        return EXT_NAME

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        try:
            tabName = 'img2img' if is_img2img else 'txt2img'
            with (
                InputAccordion(False, label=EXT_NAME) if InputAccordion
                else gr.Accordion(EXT_NAME, open=False)
                as enable
            ):
                if not InputAccordion:
                    with gr.Row():
                        enable = gr.Checkbox(False, label="Enable")
                with gr.Row():
                    gr.Markdown(f'This script takes all {EXT_NAME} settings from its tab')
                with gr.Row():
                    save_originals = gr.Checkbox(True, label="Save originals", elem_id=f'replacer_{tabName}_save_originals')
                    follow_txt2img_hires_fix = gr.Checkbox(True, label="Follow txt2img hires fix",
                        elem_id=f'replacer_{tabName}_follow_txt2img_hires_fix',visible=not is_img2img)
                with gr.Row():
                    override_seed = gr.Checkbox(True, label=f"Use {tabName} seed",
                        elem_id=f'replacer_{tabName}_override_seed', tooltip=f"If false, seed in {EXT_NAME} tab is used")
                    append_positive_prompt = gr.Checkbox(False, label=f"Append positive prompt", elem_id=f'replacer_{tabName}_append_positive_prompt',
                        tooltip=f"If true, positive prompt from generation will be appended at the beginning of {EXT_NAME} tab's positive prompt")
                with gr.Row():
                    force_override_sd_model = gr.Checkbox(label='Force override stable diffusion model',
                        value=True, elem_id=f"replacer_{tabName}_force_override_sd_model",
                        info='Be sure you use inpainting model here')
                    force_sd_model_checkpoint = ui_settings.create_setting_component('sd_model_checkpoint')

            comp = replacer_tab_ui.replacerMainUI.components

            main_tab_inputs = [
                comp.detectionPrompt,
                comp.avoidancePrompt,
                comp.positivePrompt,
                comp.negativePrompt,
                comp.upscaler_for_img2img,
                comp.seed,
                comp.sampler,
                comp.scheduler,
                comp.steps,
                comp.box_threshold,
                comp.mask_expand,
                comp.mask_blur,
                comp.max_resolution_on_detection,
                comp.sam_model_name,
                comp.dino_model_name,
                comp.cfg_scale,
                comp.denoise,
                comp.inpaint_padding,
                comp.inpainting_fill,
                comp.width,
                comp.height,
                comp.batch_count,
                comp.batch_size,
                comp.inpainting_mask_invert,
                comp.extra_includes,
                comp.fix_steps,
                comp.override_sd_model,
                comp.sd_model_checkpoint,
                comp.mask_num,
                comp.avoid_mask_mode,
                comp.avoidance_mask,
                comp.only_custom_mask,
                comp.custom_mask_mode,
                comp.custom_mask,
                comp.use_inpaint_diff,
                comp.inpaint_diff_mask_view,
                comp.clip_skip,
                comp.pass_into_hires_fix_automatically,
                comp.save_before_hires_fix,
                comp.do_not_use_mask,
                comp.rotation_fix,
                comp.variation_seed,
                comp.variation_strength,

                comp.hf_upscaler,
                comp.hf_steps,
                comp.hf_sampler,
                comp.hf_scheduler,
                comp.hf_denoise,
                comp.hf_cfg_scale,
                comp.hfPositivePromptSuffix,
                comp.hf_size_limit,
                comp.hf_above_limit_upscaler,
                comp.hf_unload_detection_models,
                comp.hf_disable_cn,
                comp.hf_extra_mask_expand,
                comp.hf_positivePrompt,
                comp.hf_negativePrompt,
                comp.hf_sd_model_checkpoint,
                comp.hf_extra_inpaint_padding,
                comp.hf_extra_mask_blur,
                comp.hf_randomize_seed,
                comp.hf_soft_inpaint,
            ] + comp.cn_inputs \
            + comp.soft_inpaint_inputs

            for i in range(len(main_tab_inputs)):
                main_tab_inputs[i] = copy.copy(main_tab_inputs[i])
                main_tab_inputs[i].do_not_save_to_config = True

            inputs = [
                enable,
                save_originals,
                force_override_sd_model,
                force_sd_model_checkpoint,
                follow_txt2img_hires_fix,
                override_seed,
                append_positive_prompt,
            ]

            return inputs + main_tab_inputs

        except:
            return []

    def before_process(self, p: StableDiffusionProcessingTxt2Img,
        enable,
        save_originals,
        force_override_sd_model,
        force_sd_model_checkpoint,
        follow_txt2img_hires_fix,
        override_seed,
        append_positive_prompt,

        detectionPrompt,
        avoidancePrompt,
        positivePrompt,
        negativePrompt,
        upscaler_for_img2img,
        seed,
        sampler,
        scheduler,
        steps,
        box_threshold,
        mask_expand,
        mask_blur,
        max_resolution_on_detection,
        sam_model_name,
        dino_model_name,
        cfg_scale,
        denoise,
        inpaint_padding,
        inpainting_fill,
        width,
        height,
        batch_count,
        batch_size,
        inpainting_mask_invert,
        extra_includes,
        fix_steps,
        override_sd_model,
        sd_model_checkpoint,
        mask_num,
        avoid_mask_mode,
        avoidance_mask,
        only_custom_mask,
        custom_mask_mode,
        custom_mask,
        use_inpaint_diff,
        inpaint_diff_mask_view,
        clip_skip,
        pass_into_hires_fix_automatically,
        save_before_hires_fix,
        do_not_use_mask,
        rotation_fix,
        variation_seed,
        variation_strength,

        hf_upscaler,
        hf_steps,
        hf_sampler,
        hf_scheduler,
        hf_denoise,
        hf_cfg_scale,
        hfPositivePromptSuffix,
        hf_size_limit,
        hf_above_limit_upscaler,
        hf_unload_detection_models,
        hf_disable_cn,
        hf_extra_mask_expand,
        hf_positivePrompt,
        hf_negativePrompt,
        hf_sd_model_checkpoint,
        hf_extra_inpaint_padding,
        hf_extra_mask_blur,
        hf_randomize_seed,
        hf_soft_inpaint,

        *scripts_args,
    ):
        self.enable = enable
        if not self.enable:
            return

        p.do_not_save_grid = True
        self.save_samples = getattr(p, 'save_samples', lambda: True)()
        if not save_originals:
            p.do_not_save_samples = True

        self.save_originals = save_originals
        self.extra_includes = extra_includes
        self.force_override_sd_model = force_override_sd_model
        self.force_sd_model_checkpoint = force_sd_model_checkpoint
        self.follow_txt2img_hires_fix = follow_txt2img_hires_fix
        self.override_seed = override_seed
        self.append_positive_prompt = append_positive_prompt


        cn_args, soft_inpaint_args = replacer_extensions.prepareScriptsArgs(scripts_args)

        hires_fix_args = HiresFixArgs(
            upscaler = hf_upscaler,
            steps = hf_steps,
            sampler = hf_sampler,
            scheduler = hf_scheduler,
            denoise = hf_denoise,
            cfg_scale = hf_cfg_scale,
            positive_prompt_suffix = hfPositivePromptSuffix,
            size_limit = hf_size_limit,
            above_limit_upscaler = hf_above_limit_upscaler,
            unload_detection_models = hf_unload_detection_models,
            disable_cn = hf_disable_cn,
            extra_mask_expand = hf_extra_mask_expand,
            positive_prompt = hf_positivePrompt,
            negative_prompt = hf_negativePrompt,
            sd_model_checkpoint = hf_sd_model_checkpoint,
            extra_inpaint_padding = hf_extra_inpaint_padding,
            extra_mask_blur = hf_extra_mask_blur,
            randomize_seed = hf_randomize_seed,
            soft_inpaint = hf_soft_inpaint,
        )

        self.gArgs = GenerationArgs(
            positivePrompt=positivePrompt,
            negativePrompt=negativePrompt,
            detectionPrompt=detectionPrompt,
            avoidancePrompt=avoidancePrompt,
            upscalerForImg2Img=upscaler_for_img2img,
            seed=seed,
            samModel=sam_model_name,
            grdinoModel=dino_model_name,
            boxThreshold=box_threshold,
            maskExpand=mask_expand,
            maxResolutionOnDetection=max_resolution_on_detection,

            steps=steps,
            sampler_name=sampler,
            scheduler=scheduler,
            mask_blur=mask_blur,
            inpainting_fill=inpainting_fill,
            batch_count=batch_count,
            batch_size=batch_size,
            cfg_scale=cfg_scale,
            denoising_strength=denoise,
            height=height,
            width=width,
            inpaint_full_res_padding=inpaint_padding,
            img2img_fix_steps=fix_steps,
            inpainting_mask_invert=inpainting_mask_invert,

            images=None,
            override_sd_model=override_sd_model,
            sd_model_checkpoint=sd_model_checkpoint,
            mask_num=mask_num,
            avoidance_mask=prepareMask(avoid_mask_mode, avoidance_mask),
            only_custom_mask=only_custom_mask,
            custom_mask=prepareMask(custom_mask_mode, custom_mask),
            use_inpaint_diff=use_inpaint_diff and inpaint_diff_mask_view is not None and \
                replacer_extensions.inpaint_difference.Globals is not None and \
                replacer_extensions.inpaint_difference.Globals.generated_mask is not None,
            clip_skip=clip_skip,
            pass_into_hires_fix_automatically=pass_into_hires_fix_automatically,
            save_before_hires_fix=save_before_hires_fix,
            previous_frame_into_controlnet=[],
            do_not_use_mask=do_not_use_mask,
            animatediff_args=DUMMY_ANIMATEDIFF_ARGS,
            rotation_fix=rotation_fix,
            variation_seed=variation_seed,
            variation_strength=variation_strength,

            hires_fix_args=hires_fix_args,
            cn_args=cn_args,
            soft_inpaint_args=soft_inpaint_args,
            )
        prepareExpectedUIBehavior(self.gArgs)



    def postprocess(self, p: StableDiffusionProcessingTxt2Img, processed: Processed, *args):
        if not self.enable:
            return
        self.gArgs.images = [x.convert('RGBA') for x in processed.images[:len(processed.all_seeds)]]
        if self.force_override_sd_model:
            self.gArgs.override_sd_model = True
            self.gArgs.sd_model_checkpoint = self.force_sd_model_checkpoint
        if self.follow_txt2img_hires_fix and hasattr(p, 'enable_hr'):
            self.gArgs.pass_into_hires_fix_automatically = p.enable_hr
        if self.override_seed:
            self.gArgs.seed = processed.all_seeds[0]
        if self.append_positive_prompt:
            self.gArgs.positivePrompt = processed.prompt + ", " + self.gArgs.positivePrompt

        saveDir = p.outpath_samples if self.save_samples else None
        saveToSubdirs = p.override_settings.get('save_to_dirs', None)
        if saveToSubdirs is None: saveToSubdirs = shared.opts.save_to_dirs

        try:
            processedReplacer, allExtraImages = generate(self.gArgs, saveDir, saveToSubdirs, False, self.extra_includes)
        except Exception as e:
            print(f"[{EXT_NAME}] Exception: {e}")
            return
        if processedReplacer is None:
            print(f"[{EXT_NAME}] No one image was processed")
            return

        if self.save_originals:
            processed.images.extend(processedReplacer.images + allExtraImages)
            processed.infotexts += processedReplacer.infotexts
        else:
            processed.images = processed.images[len(processed.all_seeds):]
            processed.infotexts = processed.infotexts[len(processed.all_seeds):]
            processed.images = processedReplacer.images + processed.images + allExtraImages
            processed.infotexts = processedReplacer.infotexts[:len(processedReplacer.images)] + processed.infotexts + processedReplacer.infotexts[len(processedReplacer.images):]


if needHideReplacerScript():
    del ReplacerScript
