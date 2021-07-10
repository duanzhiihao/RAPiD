import argparse
import torch

from models.rapid_export import RAPiD


def export():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/pL1_MWHB1024_Mar11_4000.ckpt')
    parser.add_argument('--device',  type=str, default='cuda:0')
    parser.add_argument('--half',    action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)
    input_shape = (1, 3, 1024, 1024)

    model = RAPiD(input_hw=input_shape[2:4])
    weights = torch.load(args.weights)
    # from mycv.utils.torch_utils import summary_weights
    # summary_weights(weights['model'])
    model.load_state_dict(weights['model'])
    model = model.to(device=device)
    model.eval()

    # for k, m in model.named_modules():
    #     if hasattr(m, 'num_batches_tracked'):
    #         m.num_batches_tracked = m.num_batches_tracked.float()

    if args.half:
        model = model.half()

    if args.half:
        x = torch.rand(*input_shape, dtype=torch.float16, device=device)
    else:
        x = torch.rand(*input_shape, device=device)
    torch.onnx.export(model, x, 'rapid.onnx', verbose=True, opset_version=11)


def check():
    import onnx
    model = onnx.load("rapid.onnx")
    # Check that the IR is well formed
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    s = onnx.helper.printable_graph(model.graph)
    with open('tmp.txt', 'a') as f:
        print(s, file=f)
    debug = 1


if __name__ == '__main__':
    export()
    # check()
