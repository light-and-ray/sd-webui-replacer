function submit_replacer() {
    showSubmitButtons('replacer', false);

    var id = randomId();

    requestProgress(id, gradioApp().getElementById('replacer_gallery_container'), gradioApp().getElementById('replacer_gallery'), function() {
        showSubmitButtons('replacer', true);
    });

    var res = create_submit_args(arguments);

    res[0] = id;

    console.log(res);
    return res;
}


function submit_replacer_hf() {
    showSubmitButtons('replacer_hf', false);

    var id = randomId();

    requestProgress(id, gradioApp().getElementById('replacer_gallery_container'), gradioApp().getElementById('replacer_gallery'), function() {
        showSubmitButtons('replacer_hf', true);
    });

    var res = create_submit_args(arguments);

    res[0] = id;

    console.log(res);
    return res;
}
