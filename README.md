# Replacer

Replacer is an extension for [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui). The goal of this extension is to automate objects masking by detection prompt, using [sd-webui-segment-anything](https://github.com/continue-revolution/sd-webui-segment-anything), and img2img inpainting in one easy to use tab. It also useful for batch inpaint, and inpaint in video with AnimateDiff


![](/docs/images/main_screenshot.jpg)

You also can draw your mask instead of or in addition to detection, and take advantage of convenient HiresFix option, and ControlNet inpainting with preserving original image resolution and aspect ratio

> If you find this project useful, please star it on GitHub!

## Installation
1. Install [sd-webui-segment-anything](https://github.com/continue-revolution/sd-webui-segment-anything) extension. If it bothers you, you can hide it in the Replacer's settings. Go to tab `Extension` -> `Available` -> click `Load from` and search _"sd-webui-segment-anything"_
2. Download model [sam_hq_vit_l.pth](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth), or others from the list bellow, and put it into `extensions/sd-webui-segment-anything/models/sam`
3. For faster hires fix, download [lcm-lora-sdv1-5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/blob/main/pytorch_lora_weights.safetensors), rename it into `lcm-lora-sdv1-5.safetensors`, put into `models/Lora`. Or if you have already lcm lora, then change hires suffix in the extension options
4. Install this extension. Go to tab `Extension` -> `Available` -> click `Load from` and search _"Replacer"_. Be sure your sd-webui version is >= 1.5.0. For AMD and Intel GPUs, and maybe something else, you need to enable CPU for detection in Replacer's settings. Go to `Settings` -> `Replacer` and enable it
5. Reload UI

If you don't want to use Video feature, that's all for you. Further steps are for Video:

1. Install [sd-webui-animatediff](https://github.com/continue-revolution/sd-webui-animatediff) and [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) extensions. You should also use `Extension` -> `Available` tab and find them there
2. Download [mm_sd15_v3.safetensors](https://huggingface.co/conrevo/AnimateDiff-A1111/resolve/main/motion_module/mm_sd15_v3.safetensors) animatediff's motion model, and put it into `extensions/sd-webui-animatediff/model` directory
3. Download [control_v11p_sd15_inpaint_fp16.safetensors](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors) controlnet's model and put it into `models/ControlNet` directory
4. I strongly recommend you to download [mm_sd15_v3_sparsectrl_rgb.safetensors](https://huggingface.co/conrevo/AnimateDiff-A1111/resolve/main/control/mm_sd15_v3_sparsectrl_rgb.safetensors) and [mm_sd15_v3_sparsectrl_scribble.safetensors](https://huggingface.co/conrevo/AnimateDiff-A1111/resolve/main/control/mm_sd15_v3_sparsectrl_scribble.safetensors) controlnet's models. Put them also into `models/ControlNet` directory. Then you can select SparseCtrl module in ControlNet extension. The rgb one requires "none" preprocessor


##### SAM models list:

SAM-HQ are the best for me. Choose it depending on your vram. Sum this model size with dino model size (694MB-938MB)

<blockquote>

1. [SAM](https://github.com/facebookresearch/segment-anything) from Meta AI.
    - [2.56GB sam_vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
    - [1.25GB sam_vit_l](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
    - [375MB sam_vit_b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

2. [SAM-HQ](https://github.com/SysCV/sam-hq) from SysCV.
    - [2.57GB sam_hq_vit_h](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth)
    - [1.25GB sam_hq_vit_l](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth)
    - [379MB sam_hq_vit_b](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth)

3. [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) from Kyung Hee University.
    - [39MB mobile_sam](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt)

</blockquote>

_FastSAM_ and _Matting-Anything_ aren't currently supported



## How does it work?

First, grounding dino models detect objects you provided in the detection prompt. Then segment anything model generates contours of them. And then extension chooses randomly 1 of 3 generated masks, and inpaints it with regular inpainting method in a1111 webui

When you press the "Apply hires fix" button, the extension regenerates the image with exactly the same settings, excluding upscaler_for_img2img. Then it applies inpainting with "Original" masked content mode and lower denoising but higher resolution.



## Supported extensons:

<blockquote>

1. [Lama cleaner as masked content](https://github.com/light-and-ray/sd-webui-lama-cleaner-masked-content)

1. [Image Comparison](https://github.com/Haoming02/sd-webui-image-comparison)

1. [ControlNet](https://github.com/Mikubill/sd-webui-controlnet)

1. [AnimateDiff](https://github.com/continue-revolution/sd-webui-animatediff)

1. Other extension scripts which doesn't have control arguments, e.g. [Hardware Info in metadata](https://github.com/light-and-ray/sd-webui-hardware-info-in-metadata), [NudeNet NSFW Censor](https://github.com/w-e-w/sd-webui-nudenet-nsfw-censor), built-in **Hypertile**

</blockquote>



## Docs:
### - [Usage of Replacer](/docs/usage.md)
### - [Video: AnimateDiff and Frame by frame](/docs/video.md)
### - [Replacer Options](/docs/options.md)
### - [Information about Replacer API](/docs/api.md)
### - [Useful tips: how to change defaults, maximal value of sliders, and how to get inpainting model](/docs/tips.md)


--------------------------------

## Need to do:

- ☑️ cache mask
- ☑️ batch processing
- ☑️ "apply hires fix" button
- ☑️ additional options
- ☑️ progress bar + interrupt
- ☑️ option for pass into hires fix automatically
- ☑️ control net
- ☑️ pass previous frame into ControlNet for video
- tiled vae
- ☑️ "hide segment anything extension" option
- ☑️ txt2img script
- more video and mask input types
- RIFE frames interpolation
- allow multiply instances (presets)

### small todo:
- add hires fix args into metadata
- FreeInit support
