import argparse
import torch

from models.rapid import RAPiD



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/pL1_MWHB1024_Mar11_4000.ckpt')
    parser.add_argument('--device',  type=str, default='cuda:0')
    parser.add_argument('--half',    action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)
    input_shape = (1, 3, 1024, 1024)

    model = RAPiD()
    weights = torch.load(args.weights)
    model.load_state_dict(weights['model'])
    model = model.to(device=device)
    model.eval()

    if args.half:
        model = model.half()

    if args.half:
        x = torch.rand(*input_shape, dtype=torch.float16, device=device)
    else:
        x = torch.rand(*input_shape, device=device)
    torch.onnx.export(model, x, 'rapid.onnx', verbose=True, opset_version=11)


if __name__ == '__main__':
    main()
