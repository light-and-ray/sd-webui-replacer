
# Useful Tips!
## How to change default values of advanced options and hires fix options?

![](/docs/images/defaults.jpg)

You need to reload the web page, then set your desirable settings. Then go to the "Defaults" section in the "Settings" tab. Click "View changes", check is it ok, then click "Apply" and "Reload UI"

## How to get an inpainting model?

I recommend you to using [EpicPhotoGasm - Z - Inpainting](https://civitai.com/models/132632?modelVersionId=201346) model for realism. If you've already have your favorite model, but it doesn't have inpainting model, you can make it in "Checkpoint Merger" tab:
1. Select your target model as "model B"
2. Select [sd-v1-5-inpainting](https://huggingface.co/webui/stable-diffusion-inpainting/blob/main/sd-v1-5-inpainting.safetensors) as "model A"
3. Select `sd_v1-5-pruned-emaonly` as "model C"
4. Set `Custom Name` the same as your target model name (`.inpainting` suffix will be added automatically)
5. Set `Multiplier (M)` to 1.0
6. Select `Interpolation Method` to "add difference", and "Save as float16"
7. For sdxl I recommend you to select fixed sdxl vae in baked vae dropdown
8. Merge

![](images/inpaint_merge.jpg)


## How to change maximum values of sliders?

In file `ui-config.json` in root of webui you can edit maximum and minimum values of sliders

