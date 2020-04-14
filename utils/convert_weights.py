import torch
from collections import OrderedDict
from models import YOLOv3

def load_weights_from(model, path):
    pretrained = torch.load(path)
    pretrained = list(pretrained.items())
    # with open('./official_80.txt', 'w') as f:
    #     for k,v in pretrained:
    #         print(k, file=f)
    state_dict = model.state_dict()
    state_dict = list(state_dict.items())
    assert len(pretrained) == len(state_dict)

    new_dict = []
    for (_, tensor), (key, _) in zip(pretrained, state_dict):
        if 'to_box' in key:
            continue
        new_dict.append((key,tensor))

    new_dict = OrderedDict(new_dict)
    model.load_state_dict(new_dict, strict=False)

    return model


if __name__ == "__main__":
    anchors = [
        [10, 13, 0], [16, 30, 0], [33, 23, 0],
        [30, 61, 0], [62, 45, 0], [59, 119, 0],
        [116, 90, 0], [156, 198, 0], [373, 326, 0]
    ]
    indices = [[6,7,8], [3,4,5], [0,1,2]]
    model = YOLOv3(anchors=anchors, indices=indices, loss_angle='none')
    # model.apply(weights_init_normal)
    model = load_weights_from(model, './weights/official_80.pth')
    torch.save(model.state_dict(), './weights/official_noyolo.pth')

    debug = 1