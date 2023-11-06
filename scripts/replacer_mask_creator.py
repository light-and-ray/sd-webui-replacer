from PIL import Image, PngImagePlugin
from PIL import ImageChops


def is_images_the_same(image_one, image_two):
    diff = ImageChops.difference(image_one, image_two)

    if diff.getbbox():
        return False
    else:
        return True


masksCreatorCached = None


class MasksCreator:
    def __init__(self, detectionPrompt, image, samModel, grdinoModel, boxThreshold):
        self.detectionPrompt = detectionPrompt
        self.image = image
        self.samModel = samModel
        self.grdinoModel = grdinoModel
        self.boxThreshold = boxThreshold

        global masksCreatorCached

        if masksCreatorCached is not None and \
                self.detectionPrompt == masksCreatorCached.detectionPrompt and\
                self.samModel == masksCreatorCached.samModel and\
                self.grdinoModel == masksCreatorCached.grdinoModel and\
                self.boxThreshold == masksCreatorCached.boxThreshold and\
                is_images_the_same(self.image, masksCreatorCached.image):
            self.previews = masksCreatorCached.previews
            self.masksExpanded = masksCreatorCached.masksExpanded
            self.cutted = masksCreatorCached.cutted
            print('MasksCreator restored from cache')
        else:
            self._createMasks()
            masksCreatorCached = self
            print('MasksCreator cached')


    def _createMasks(self):
        from scripts.sam import sam_predict, update_mask

        masks, samLog = sam_predict(self.samModel, self.image, [], [], True,
            self.grdinoModel, self.detectionPrompt, self.boxThreshold, False, [])
        print(samLog)
        masks = [masks[3], masks[4], masks[5]]
        
        self.previews = []
        self.masksExpanded = []
        self.cutted = []
        
        for mask in masks:
            expanded = update_mask(mask, 0, 35, self.image)
            self.previews.append(expanded[0])
            self.masksExpanded.append(expanded[1])
            self.cutted.append(expanded[2])