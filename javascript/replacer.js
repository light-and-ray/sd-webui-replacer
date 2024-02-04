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
        function() {
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
    "Mask Expand": "Mask dilation, px, releative to \"Max resolution on detection\"",
    "Extra mask expand": "Extra mask dilation on hires fix step, px, releative to \"Max resolution on detection\"",
    "Limit avoidance mask canvas resolution on creating": "Limit the canvas create by the buttom, using \"Max resolution on detection\" option",
    "Limit custom mask canvas resolution on creating": "Limit the canvas create by the buttom, using \"Max resolution on detection\" option",
};


let replacer_gallery = undefined;
onAfterUiUpdate(function() {
    if (!replacer_gallery) {
        replacer_gallery = attachGalleryListeners("replacer");
    }
});


function replacerGetCurrentSourceImg(dummy_component, isAvoid, needLimit, maxResolutionOnDetection) {
        const img = gradioApp().querySelector('#replacer_image div div img');
        var maskId = ''
        if (isAvoid){
            maskId = 'replacer_avoidance_mask'
        } else {
            maskId = 'replacer_custom_mask'
        }
        const removeButton = gradioApp().getElementById(maskId).querySelector('button[aria-label="Remove Image"]');
        if (removeButton){
            removeButton.click();
        }
        let resImg = img ? img.src : null;
        return [resImg, isAvoid, needLimit, maxResolutionOnDetection]
    }


function replacerApplyZoomAndPanIntegration () {
    if (typeof window.applyZoomAndPanIntegration === "function") {
        window.applyZoomAndPanIntegration_replacer_mod("#replacer_advanced_options", ["#replacer_avoidance_mask", "#replacer_custom_mask"]);
        var index = uiUpdateCallbacks.indexOf(replacerApplyZoomAndPanIntegration);
        if (index !== -1) {
            uiUpdateCallbacks.splice(index, 1);
        }
    }
}

onUiUpdate(replacerApplyZoomAndPanIntegration);


