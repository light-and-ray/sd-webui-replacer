
class GenerationArgs:
    def __init__(self,
            positvePrompt,
            negativePrompt,
            detectionPrompt,
            image,
            mask,
            upscalerForImg2Img,
            seed,
            samModel,
            grdinoModel,
            boxThreshold,
            maskExpand,
            steps,
            sampler_name,
            mask_blur,
            inpainting_fill,
            n_iter,
            batch_size,
            cfg_scale,
            denoising_strength,
            height,
            width,
            inpaint_full_res_padding,
            img2img_fix_steps,
        ):
        self.positvePrompt = positvePrompt
        self.negativePrompt = negativePrompt
        self.detectionPrompt = detectionPrompt
        self.image = image
        self.mask = mask
        self.upscalerForImg2Img = upscalerForImg2Img
        self.seed = seed
        self.samModel = samModel
        self.grdinoModel = grdinoModel
        self.boxThreshold = boxThreshold
        self.maskExpand = maskExpand
        self.steps = steps
        self.sampler_name = sampler_name
        self.mask_blur = mask_blur
        self.inpainting_fill = inpainting_fill
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.cfg_scale = cfg_scale
        self.denoising_strength = denoising_strength
        self.height = height
        self.width = width
        self.inpaint_full_res_padding = inpaint_full_res_padding
        self.img2img_fix_steps = img2img_fix_steps

    