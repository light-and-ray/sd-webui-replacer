import gradio as gr
from modules import shared, sd_samplers
from modules.ui_components import ToolButton
from modules.api.api import encode_pil_to_base64, decode_base64_to_image
from replacer.options import EXT_NAME
from replacer.tools import limitSizeByOneDemention


try:
    from modules import ui_toprow
except:
    ui_toprow = None



IS_WEBUI_1_5 = False
if not hasattr(sd_samplers, 'visible_sampler_names'): # webui 1.5
    sd_samplers.visible_sampler_names = lambda: [x.name for x in sd_samplers.samplers_for_img2img if x.name not in shared.opts.hide_samplers]
    IS_WEBUI_1_5 = True

try:
    from modules.ui_common import OutputPanel # webui 1.8+
    IS_WEBUI_1_8 = True
except Exception as e:
    IS_WEBUI_1_8 = False

if IS_WEBUI_1_8:
    from modules import infotext_utils



def update_mask_brush_color(color):
    return gr.Image.update(brush_color=color)

def get_current_image(image, isAvoid, needLimit, maxResolutionOnDetection):
    if image is None:
        return
    if needLimit:
        image = decode_base64_to_image(image)
        image = limitSizeByOneDemention(image, maxResolutionOnDetection)
        image = 'data:image/png;base64,' + encode_pil_to_base64(image).decode()
    return gr.Image.update(image)


def unloadModels():
    mem_stats = {k: -(v//-(1024*1024)) for k, v in shared.mem_mon.stop().items()}
    memBefore = mem_stats['reserved']
    from scripts.sam import clear_cache
    clear_cache()
    mem_stats = {k: -(v//-(1024*1024)) for k, v in shared.mem_mon.stop().items()}
    memAfter = mem_stats['reserved']
    
    text = f'[{EXT_NAME}] {(memBefore - memAfter) / 1024 :.2f} GB of VRAM were freed'
    print(text, flush=True)
    if not IS_WEBUI_1_5:
        gr.Info(text)



def getSubmitJsFunction(galleryId, buttonsId, extraShowButtonsId, fillGalleryIdx):
    if not ui_toprow:
        return ''
    fillGalleryIdxCode = ''
    if fillGalleryIdx:
        fillGalleryIdxCode = 'arguments_[1] = selected_gallery_index();'
    return 'function(){'\
        'var arguments_ = Array.from(arguments);'\
        f'{fillGalleryIdxCode}'\
        f'arguments_.push("{extraShowButtonsId}", "{buttonsId}", "{galleryId}");'\
        'return submit_replacer.apply(null, arguments_);'\
    '}'


def sendBackToReplacer(gallery, gallery_index):
    assert len(gallery) > 0, 'No image'
    assert 0 <= gallery_index < len(gallery), f'Bad image index: {gallery_index}'
    assert not IS_WEBUI_1_5, 'sendBackToReplacer is not supported for webui < 1.8'
    image_info = gallery[gallery_index] if 0 <= gallery_index < len(gallery) else gallery[0]
    image = infotext_utils.image_from_url_text(image_info)
    return image




class OuputPanelWatcher():
    send_to_img2img = None
    send_to_inpaint = None
    send_to_extras = None
    send_back_to_replacer = None


def watchOuputPanel(component, **kwargs):
    elem_id = kwargs.get('elem_id', None)
    if elem_id is None:
        return

    if elem_id == 'replacer_send_to_img2img' or elem_id == 'img2img_tab':
        OuputPanelWatcher.send_to_img2img = component

    if elem_id == 'replacer_send_to_inpaint' or elem_id == 'inpaint_tab':
        OuputPanelWatcher.send_to_inpaint = component

    if elem_id == 'replacer_send_to_extras' or elem_id == 'extras_tab':
        OuputPanelWatcher.send_to_extras = component
        OuputPanelWatcher.send_back_to_replacer = ToolButton('↙️', elem_id=f'replacer_send_back_to_replacer', tooltip="Send image back to Replcer's input")


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

IS_WEBUI_1_9 = hasattr(shared.cmd_opts, 'unix_filenames_sanitization')


