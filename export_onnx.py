from PIL import Image
import numpy as np
import torch

from models.rapid_export import RAPiD
from utils import utils, visualization


def post_processing(dts, pad_info, conf_thres=0.3, nms_thres=0.3):
    assert isinstance(dts, torch.Tensor)
    dts = dts[dts[:,5] >= conf_thres].cpu()
    dts = utils.nms(dts, is_degree=True, nms_thres=nms_thres)
    dts = utils.detection2original(dts, pad_info.squeeze())
    return dts


@torch.inference_mode()
def export():
    device = torch.device('cpu')
    input_size = 1024 # input image will be resized to this size

    # ======== load model ========
    model = RAPiD(input_hw=(input_size, input_size))
    # load weights
    url = 'https://github.com/duanzhiihao/RAPiD/releases/download/v0.1/pL1_MWHB1024_Mar11_4000.ckpt'
    checkpoint = torch.hub.load_state_dict_from_url(url)
    model.load_state_dict(checkpoint['model'])

    model.eval()
    model = model.to(device=device)

    dummy_input = torch.rand(1, 3, input_size, input_size, device=device)
    torch.onnx.export(model, dummy_input, 'rapid.onnx', verbose=False, opset_version=11)


def test_onnx():
    import onnx
    import onnxruntime

    onnx_model = onnx.load('rapid.onnx')
    onnx.checker.check_model(onnx_model) # sanity check

    ort_session = onnxruntime.InferenceSession("rapid.onnx", providers=["CPUExecutionProvider"])

    # ======== load image ========
    input_size = 1024 # input image will be resized to this size
    img = Image.open('images/exhibition.jpg')
    img_resized, _, pad_info = utils.rect_to_square(img, None, input_size)
    im_numpy = np.expand_dims(np.array(img_resized), 0).transpose(0,3,1,2).astype(np.float32) / 255.0

    # ======== run detection ========
    ort_inputs = {ort_session.get_inputs()[0].name: im_numpy}
    ort_outs = ort_session.run(None, ort_inputs)
    dts = ort_outs[0].squeeze(0) # remove the batch dimension
    print(type(dts), dts.shape, dts.dtype) # numpy.ndarray, (N, 6), float32

    # post-processing
    dts = torch.from_numpy(dts)
    dts = post_processing(dts, pad_info, conf_thres=0.3, nms_thres=0.3)

    # ======== visualize the results ========
    im_uint8 = np.array(img)
    visualization.draw_dt_on_np(im_uint8, dts)
    Image.fromarray(im_uint8).save('result-onnx.png')


if __name__ == '__main__':
    export()
    test_onnx()
