function submit_replacer() {
    var arguments_ = Array.from(arguments);
    galleryId = arguments_.pop();
    buttonsId = arguments_.pop();
    extraShowButtonsId = arguments_.pop();
    showSubmitButtons(buttonsId, false);

    var id = randomId();

    requestProgress(id,
        gradioApp().getElementById(galleryId + "_gallery_container"),
        gradioApp().getElementById(galleryId + "_gallery"),
        function () {
            showSubmitButtons(buttonsId, true);
            showSubmitButtons(extraShowButtonsId, true);
        }
    );

    var res = create_submit_args(arguments_);

    res[0] = id;

    console.log(res);
    return res;
}



titles = {
    ...titles,
    "Max resolution on detection": "If one side of the image is smaller than that, it will be resized before detection. It doesn't have effect on inpainting. Reduces vram usage and mask generation time.",
    "Mask Expand": "Mask dilation, px, relative to \"Max resolution on detection\"",
    "Extra mask expand": "Extra mask dilation on hires fix step, px, relative to \"Max resolution on detection\"",
    "Limit avoidance mask canvas resolution on creating": "Limit the canvas created by the button, using \"Max resolution on detection\" option",
    "Limit custom mask canvas resolution on creating": "Limit the canvas created by the button, using \"Max resolution on detection\" option",
    "Hires supersampling": "1.0 is the resolution of original image's crop region, but not smaller then firstpass resolution. More then 1.0 - multiplying on this number each sides. It calculates before limiting resolution, so it still can't be bigger then you set above",
    "Correct aspect ratio": "Preserve original width x height number of pixels, but follow generated mask's aspect ratio. In some cases can hide necessary context",
};


let replacer_gallery = undefined;
onAfterUiUpdate(function () {
    if (!replacer_gallery) {
        replacer_gallery = attachGalleryListeners("replacer");
    }
});


function replacerGetCurrentSourceImg(dummy_component, isAvoid, needLimit, maxResolutionOnDetection) {
    const img = gradioApp().querySelector('#replacer_image div div img');
    var maskId = '';
    if (isAvoid) {
        maskId = 'replacer_avoidance_mask';
    } else {
        maskId = 'replacer_custom_mask';
    }
    const removeButton = gradioApp().getElementById(maskId).querySelector('button[aria-label="Remove Image"]');
    if (removeButton) {
        removeButton.click();
    }
    let resImg = img ? img.src : null;
    return [resImg, isAvoid, needLimit, maxResolutionOnDetection];
}



async function replacer_waitForOpts() {
    for (; ;) {
        if (window.opts && Object.keys(window.opts).length) {
            return window.opts;
        }
        await new Promise(resolve => setTimeout(resolve, 100));
    }
}

var isZoomAndPanIntegrationApplied = false;

function replacerApplyZoomAndPanIntegration() {
    if (typeof window.applyZoomAndPanIntegration === "function" && !isZoomAndPanIntegrationApplied) {
        window.applyZoomAndPanIntegration("#replacer_advanced_options", ["#replacer_avoidance_mask", "#replacer_custom_mask"]);

        const maskIds = [...Array(10).keys()].map(i => `#replacer_video_mask_${i + 1}`);
        window.applyZoomAndPanIntegration("#replacer_video_masking_tab", maskIds);

        isZoomAndPanIntegrationApplied = true;
    }
}

function replacerApplyZoomAndPanIntegration_withMod() {
    if (typeof window.applyZoomAndPanIntegration === "function" && typeof window.applyZoomAndPanIntegration_replacer_mod === "function" && !isZoomAndPanIntegrationApplied) {
        window.applyZoomAndPanIntegration_replacer_mod("#replacer_advanced_options", ["#replacer_avoidance_mask", "#replacer_custom_mask"]);
        isZoomAndPanIntegrationApplied = true;
    }
}

onUiUpdate(async () => {
    if (isZoomAndPanIntegrationApplied) return;
    const opts = await replacer_waitForOpts();

    if ('set_scale_by_when_changing_upscaler' in opts) { // webui 1.9
        replacerApplyZoomAndPanIntegration();
    } else {
        replacerApplyZoomAndPanIntegration_withMod();
    }
});


function replacerRemoveInpaintDiffMaskUpload() {
    const mask = gradioApp().getElementById('replacer_inpaint_diff_mask_view');
    if (!mask) return;
    var imageContainer = mask.getElementsByClassName('image-container')[0];
    if (!imageContainer) return;
    const images = imageContainer.getElementsByTagName('img');

    if (images.length == 0) {
        imageContainer.style.visibility = 'hidden';
    } else {
        imageContainer.style.visibility = 'visible';
    }
}

onUiUpdate(replacerRemoveInpaintDiffMaskUpload);


function replacerRemoveVideoMaskUpload() {
    const maskIds = [...Array(10).keys()].map(i => `replacer_video_mask_${i + 1}`);
    maskIds.forEach((maskId) => {
        const mask = gradioApp().getElementById(maskId);
        if (!mask) return;

        const removeButton = mask.querySelector('button[title="Remove Image"]');
        if (removeButton) {
            removeButton.style.display = "none";
        }

        const imageContainer = mask.getElementsByClassName('image-container')[0];
        if (!imageContainer) return;
        const images = imageContainer.getElementsByTagName('canvas');
        if (images.length == 0) {
            imageContainer.style.visibility = 'hidden';
        } else {
            imageContainer.style.visibility = 'visible';
        }
    });
}

onUiUpdate(replacerRemoveVideoMaskUpload);


onUiLoaded(function () {
    let replacer_generate = gradioApp().getElementById('replacer_generate');
    let replacer_hf_generate = gradioApp().getElementById('replacer_hf_generate');
    replacer_generate.title = '';
    replacer_hf_generate.title = '';
});




function sendBackToReplacer() {
    var res = Array.from(arguments);

    res[1] = selected_gallery_index();

    return res;
}



function replacer_imageComparisonloadImage() {
    let source_a = gradioApp().getElementById('replacer_image').querySelector('img');
    let source_b = gradioApp().getElementById('replacer_gallery').querySelector('img');

    if (source_a == null || source_b == null) return;

    ImageComparator.img_A.src = source_a.src;
    ImageComparator.img_B.src = source_b.src;
    ImageComparator.reset();
}

function replacer_imageComparisonAddButton() { // https://github.com/Haoming02/sd-webui-image-comparison
    // 0: Off ; 1: Text ; 2: Icon
    var option = 0;
    const replacer_image_comparison = gradioApp().getElementById('replacer_image_comparison');
    if (replacer_image_comparison) {
        option = 2;
        const inputs = gradioApp().getElementById('tab_sd-webui-image-comparison')?.getElementsByTagName('input');
        for (var i = 0; i < inputs.length; i++) {
            inputs[i].disabled = false;
        }
    } else {
        const config = gradioApp().getElementById('setting_comp_send_btn')?.querySelectorAll('label');
        if (!config) return;
        for (let i = 1; i < 3; i++) {
            if (config[i].classList.contains('selected')) {
                option = i;
                break;
            }
        }
    }

    if (option === 0) return;

    const row = gradioApp().getElementById("image_buttons_replacer").querySelector('.form');
    const btn = row.lastElementChild.cloneNode();

    btn.id = "replacer_send_to_comp";
    btn.title = "Send images to comparison tab.";
    if (btn.classList.contains("hidden")) {
        btn.classList.remove("hidden");
    }
    if (option === 1) {
        btn.textContent = "Send to Comparison";
    } else {
        btn.textContent = "🆚";
    }

    btn.addEventListener('click', () => {
        replacer_imageComparisonloadImage();
        ImageComparator.switch_to_comparison();
    });
    row.appendChild(btn);
}

onUiLoaded(replacer_imageComparisonAddButton);


function closeAllVideoMasks() {
    const videoMasks = document.querySelectorAll('.replacer_video_mask');
    videoMasks.forEach((mask, index) => {
        const removeButton = mask.querySelector('button[title="Remove Image"]');
        if (removeButton) {
            removeButton.click();
            const canvases = mask.querySelectorAll('canvas');
            canvases.forEach((canvas) => {
                const ctx = canvas.getContext('2d');
                if (ctx) {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                }
                canvas.width = 0;
                canvas.height = 0;
            });
            const images = mask.querySelectorAll('img');
            images.forEach((img) => {
                img.src = '';
                img.onload = null;
                img.onerror = null;
            });
        }
    });
    return [...arguments]
}

