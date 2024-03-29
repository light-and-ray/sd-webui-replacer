
# Usage
## General
You just need to upload your image, enter 3 prompts, and click "Run". You can override prompts examples in Settings with your commonly using prompts. Don't forget to select inpaint checkpoint

Be sure you are using inpainting model

By default if a prompt is empty, it uses first prompt from examples. You can disable this behavior in settings for positive and negative prompts. Detection prompt can not be empty

You can detect few objects, just using comma `,`


## Advanced options

### Generation
![](/docs/images/advanced_options_generation.jpg)

- _"Do exactly the number of steps the slider specifies"_: actual steps num is steps from slider * denoising straight
- _"width"_, _"height"_: internal resolution on generation. 512 for sd1, 1024 for sdxl. If you increase, it will produce mutations for high denoising straight
- _"Upscaler for img2Img"_: which method will be used to fix the generated image inside the original image. It can be used instead hires fix. DAT upscalers are good. For example this is a good one: https://openmodeldb.info/models/4x-FaceUpDAT

### Detection
![](/docs/images/advanced_options_detection.jpg)

- _"Mask num"_: SegmentAnything generates 3 masks for 1 image. By default, it's selected randomly by seed, but you can override it. See which mask num was used in generation info

### Inpainting
![](/docs/images/advanced_options_inpainting.jpg)

- _"Padding"_: How much context around a mask will be passed into a generation. You can see it in the  live preview
- _"Denoising"_: 0.0 - original image (sampler Euler a), 1.0 - completely new image. If you use a low denoising level, you need to use `Original` in masked content
- _"Lama cleaner"_: remove the masked object and then pass into inpainting. From extension: https://github.com/light-and-ray/sd-webui-lama-cleaner-masked-content
- _"Soft inpainting"_: can be used instead of inpainting models, or with a small `mask expand` to not change inpainting area too much. E.g. change color. You need to set high `mask blur` for it!
- _"Mask mode"_: useful to invert selection (Replace everything except something). You need to set a negative `mask expand` for it.

### Avoidance
![](/docs/images/advanced_options_avoidance.jpg)

- You can draw mask or/and type prompt, which will be excluded from the mask

### Custom mask
![](/docs/images/advanced_options_custom_mask.jpg)

- If you want to use this extension for regular inpainting by drown mask, to take advantage of HiresFix, batch processing or controlnet inpaining, which are not able in img2img/inpaint tab of webui
- Or it can be appended to generated mask if `Do not use detection prompt if use custom mask` is disabled. Opposite of avoidance mask

### Inpaing Diff
![](/docs/images/advanced_options_inpaint_diff.jpg)

[Inpaint Difference](https://github.com/John-WL/sd-webui-inpaint-difference) extension.

## HiresFix
You can select the blured image in the gallery, then press "Apply HiresFix ✨" button. Or you can enable `Pass into hires fix automatically`

Default settings are designed for using lcm lora for fast upscale. It requires lcm lora I mentioned, cfg scale 1.0 and sampling steps 4. There is no difference in quality for my opinion

Despite in txt2img for lcm lora DPM++ samplers produces awful results, while hires fix it produces a way better result. So I recommend "Use the same sampler" option

Note: hires fix is designed for single-user server

### Options - General
![](/docs/images/hiresfix_options_general.jpg)
- _"Extra inpaint padding"_: higher are recommended because generation size will never be higher then the original image

### Options - Advanced
![](/docs/images/hiresfix_options_advanced.jpg)
- _"Unload detection models before hires fix"_: I recommend you to disable it if you have a lot of vram. It will give significant negative impact on batches + `pass into hires fix automatically`

## Video inpainting

You can use Replacer to inpaint video with a regular stable diffusion inpaint method. It is very inconsistent, but in a few cases it can produce good enough results

![](/docs/images/replacer_video.jpg)

To increase consistency, it's better to inpaint clear objects on video with good quality and enough. Your prompts need to produce consistent results.

To suppress flickering you can generate in little fps (e.g. 10), then interpolate (x2) it with ai interpolation algorithm (e.g [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) or [frame interpolation in deforum sd-webui extension](https://github.com/deforum-art/sd-webui-deforum/wiki/Upscaling-and-Frame-Interpolation))

You can also use [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) or [lama-cleaner](https://github.com/light-and-ray/sd-webui-lama-cleaner-masked-content) with (low denosing) extensions to increase consistency, if it fits to your scenario

Also a good can be to use `Pass the previous frame into ControlNet` with _IP-Adapter_, _Reference_, _Shuffle_, _T2IA-Color_, _T2IA-Style_


## Dedicated page
Dedicated page (replacer tab only) is available on url `/replacer-dedicated`

## ControlNet
![](/docs/images/controlnet.jpg)
[ControlNet extension](https://github.com/Mikubill/sd-webui-controlnet) is also avaliavle here. [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) is also supported

## Replacer script in txt2img/img2img tabs
![](/docs/images/replacer_script.jpg)

You can use it to pass generated images into replacer immediately


## Extention name
Replacer" name of this extension, you can provide it inside `ExtensionName.txt` in root of extension directory.

Or you can override it using the environment variable `SD_WEBUI_REPLACER_EXTENTION_NAME`

For exaple: Linux
```sh
export SD_WEBUI_REPLACER_EXTENTION_NAME="Fast Inpaint"
```

Or Windows in your `.bat` file:
```bat
set SD_WEBUI_REPLACER_EXTENTION_NAME="Fast Inpaint"
```
