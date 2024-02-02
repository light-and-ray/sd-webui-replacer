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
};


let replacer_gallery = undefined;
onAfterUiUpdate(function() {
    if (!replacer_gallery) {
        replacer_gallery = attachGalleryListeners("replacer");
        replacer_gallery = attachGalleryListeners("replacer_dedicated");
    }
});


function replacerGetCurrentSourceImgForAvoidanceMask(dummy_component, imgCom) {
    const img = gradioApp().querySelector('#replacer_image div div img');
    const removeButton = gradioApp().getElementById('replacer_avoidance_mask').querySelector('button[aria-label="Remove Image"]');
    if (removeButton){
        removeButton.click();
    }
    return img ? [img.src] : [null];
}


function replacerGetCurrentSourceImgForCustomMask(dummy_component, imgCom) {
    const img = gradioApp().querySelector('#replacer_image div div img');
    const removeButton = gradioApp().getElementById('replacer_custom_mask').querySelector('button[aria-label="Remove Image"]');
    if (removeButton){
        removeButton.click();
    }
    return img ? [img.src] : [null];
}


function replacerApplyZoomAndPanIntegration () {
    if (typeof window.applyZoomAndPanIntegration === "function") {
        window.applyZoomAndPanIntegration_replacer_mod("#replacer_advanced_options", ["#replacer_avoidance_mask"]);
        window.applyZoomAndPanIntegration_replacer_mod("#replacer_advanced_options", ["#replacer_custom_mask"]);
        var index = uiUpdateCallbacks.indexOf(replacerApplyZoomAndPanIntegration);
        if (index !== -1) {
            uiUpdateCallbacks.splice(index, 1);
        }
    }
}

onUiUpdate(replacerApplyZoomAndPanIntegration);


