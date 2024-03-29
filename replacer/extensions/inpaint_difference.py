from modules import errors


# --- InpaintDifference ---- https://github.com/John-WL/sd-webui-inpaint-difference

Globals = None
computeInpaintDifference = None

def initInpaintDiffirence():
    global Globals, computeInpaintDifference
    try:
        from lib_inpaint_difference.globals import DifferenceGlobals as Globals
    except:
        Globals = None
        return

    try:
        from lib_inpaint_difference.mask_processing import compute_mask
        def computeInpaintDifference(
            non_altered_image_for_inpaint_diff,
            image,
            mask_blur,
            mask_expand,
            erosion_amount,
            inpaint_diff_threshold,
            inpaint_diff_contours_only,
        ):
            if image is None or non_altered_image_for_inpaint_diff is None:
                return None
            return compute_mask(
                non_altered_image_for_inpaint_diff.convert('RGB'),
                image.convert('RGB'),
                mask_blur,
                mask_expand,
                erosion_amount,
                inpaint_diff_threshold,
                inpaint_diff_contours_only,
            )
            
    except Exception as e:
        errors.report(f"Cannot init InpaintDiffirence {e}", exc_info=True)
        Globals = None

