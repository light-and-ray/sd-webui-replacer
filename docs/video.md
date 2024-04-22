# Video

## Common
![](/docs/images/replacer_video_common.jpg)

You need to provide a path to your video file, or url in `file://` format. On Windows you need to make right mouse click on your file with Alt key holded, and then select "copy file as path"

## AnimateDiff mode
![](/docs/images/replacer_video_animate_diff.jpg)

### General advice
Almost all advanced options work here. Inpaint padding doesn't, because it's ControlNet inpainting. Lama cleaner in masked content enables CN inpaint_only+lama module instead of inpaint_only

Due to high AnimateDiff's consistency in comparison with *"Frame by frame"* mode you can use high `mask blur` and `mask expand`.

Hires fix doesn't work here, and as I think, it basically can't, because it will decrease the consistency. But you can use the "upscaler for img2img" option - these upscalers work and consistent enough.

To increase consistency between fragments, you can use ControlNet, especially `SparseCtrl`, or try to use `Fragment length` = 0 (or just very big) and set up `Context batch size`, `Stride`, `Overlap`. I recomment make `Fragment length` few times more then `Context batch size`

`Context batch size` is set up for 12GB VRAM with one additional ControlNet unit. If you get OutOfMemort error, decrease it

If you know any other good advice, please send them into github issues, I can place them here

### AnimateDiff options
1. **Number of frames** (*Fragment length, frames* inside Replacer) — Choose whatever number you like. 

    If you enter something smaller than your `Context batch size` other than 0: you will get the first `Number of frames` frames as your output fragment from your whole generation. 
1. **FPS** (*Internal AD FPS* inside Replacer)— Frames per second, which is how many frames (images) are shown every second. If 16 frames are generated at 8 frames per second, your fragment’s duration is 2 seconds.

1. **Context batch size** — How many frames will be passed into the motion module at once. The SD1.5 motion modules are trained with 16 frames, so it’ll give the best results when the number of frames is set to `16`. SDXL HotShotXL motion modules are trained with 8 frames instead. Choose [1, 24] for V1 / HotShotXL motion modules and [1, 32] for V2 / AnimateDiffXL motion modules.
1. **Stride** — Max motion stride as a power of 2 (default: 1).
    1. Due to the limitation of the infinite context generator, this parameter is effective only when `Number of frames` > `Context batch size`, including when ControlNet is enabled and the source video frame number > `Context batch size` and `Number of frames` is 0.
    1. "Absolutely no closed loop" is only possible when `Stride` is 1.
    1. For each `1 <= 2^i <= Stride`, the infinite context generator will try to make frames `2^i` apart temporal consistent. For example, if `Stride` is 4 and `Number of frames` is 8, it will make the following frames temporal consistent:
        - `Stride` == 1: [0, 1, 2, 3, 4, 5, 6, 7]
        - `Stride` == 2: [0, 2, 4, 6], [1, 3, 5, 7]
        - `Stride` == 4: [0, 4], [1, 5], [2, 6], [3, 7]
1. **Overlap** — Number of frames to overlap in context. If overlap is -1 (default): your overlap will be `Context batch size` // 4.
    1. Due to the limitation of the infinite context generator, this parameter is effective only when `Number of frames` > `Context batch size`, including when ControlNet is enabled and the source video frame number > `Context batch size` and `Number of frames` is 0.
1. **Latent power** and **Latent scale** — Initial latent for each AnimateDiff's frame is calculated using `init_alpha` made with this formula: `init_alpha = 1 - frame_number ^ latent_power / latent_scale`. You can see these factors in console log `AnimateDiff - INFO - Randomizing init_latent according to [ ... ]`. It describes the straight of initial image of the frames. Inside Replacer initial image is the last image of the previous fragment, or inpainted the first frame in the first fragment

### SparseCtrl
SparseCtrl is a special ControlNet models for AnimateDiff. Set up it in a ControlNet unit to use it. Produces much better correspondence of the first result frame of the fragment and the initial image

![](/docs/images/replacer_video_sparsectrl.jpg)

You can download them here: https://huggingface.co/conrevo/AnimateDiff-A1111/tree/main/control RGB is for "none" preprocessor, scribble is for scribbles

## Frame by frame mode
You can use Replacer to inpaint video with an old frame by frame method with regular stable diffusion inpaint method. It is very inconsistent, but in a few cases it can produce good enough results

![](/docs/images/replacer_video_frame_by_frame.jpg)

To increase consistency, it's better to inpaint clear objects on video with good quality and enough. Your prompts need to produce consistent results.

To suppress flickering you can generate in little fps (e.g. 10), then interpolate (x2) it with ai interpolation algorithm (e.g [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) or [frame interpolation in deforum sd-webui extension](https://github.com/deforum-art/sd-webui-deforum/wiki/Upscaling-and-Frame-Interpolation))

You can also use [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) or [lama-cleaner](https://github.com/light-and-ray/sd-webui-lama-cleaner-masked-content) with (low denosing) extensions to increase consistency, if it fits to your scenario

Also a good can be to use `Pass the previous frame into ControlNet` with _IP-Adapter_, _Reference_, _Shuffle_, _T2IA-Color_, _T2IA-Style_

