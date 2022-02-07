from insightface.model_zoo import get_model


class RetinaDetector(object):
    def __init__(self, gpu):
        self.model = get_model("retinaface_r50_v1")
        self.model.prepare(gpu, 0.4)

    def infer(self,img):
        bboxes, landmarks = self.model.detect(img, threshold=0.6, scale = 1)
        return bboxes,landmarks