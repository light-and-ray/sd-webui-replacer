# Replacer

The goal of this extention is to automate objects masking by detection prompt, using [sd-webui-segment-anything](https://github.com/continue-revolution/sd-webui-segment-anything), and img2img inpainting in one easy to use tab

![](images/img1.jpg)


## Installation
1. Install [sd-webui-segment-anything](https://github.com/continue-revolution/sd-webui-segment-anything) extention
2. Put model [sam_hq_vit_l.pth](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth) into `extensions/sd-webui-segment-anything/models/sam`. Only this model supported now
4. Install this extention
5. Reload UI

## Usage
You just need to upload your image, enter 3 prompts, and click "Run". You can override prompts examples in Settings with your commonly using prompts. Don't forget to select inpaint checkpoint

By default if a prompt is empty, it uses first prompt from examples. You can disable this behavior in settings for positive and negative prompts. Detection prompt can not be empty

If you don't like "Replacer" name of this extention, you can override it using envirovment variable `SD_WEBUI_REPLACER_EXTENTION_NAME`


--------------------------

Need to do:
- ☑️ cache mask
- batch processing
- "apply hires fix" button
- additional options
- progress bar
- control net
- "hide segment anything extention" option
- txt2img script
