import json
import argparse
from api import Detector


def eval_cepdof_api(gt_path, dts_json):
    '''
    Evaluate using CEPDOF API. CEPDOF API only support the AP metric.
    '''
    from utils.cepdof_api import CEPDOFeval
    gt_json = json.load(open(gt_path, 'r'))
    cepdof = CEPDOFeval(gt_json=gt_json, dt_json=dts_json)
    cepdof.evaluate()
    cepdof.accumulate()
    cepdof.summarize()


def eval_custom(gt_path, dts_json, metric):
    '''
    Evaluate using custom code.
    '''
    from utils.MWtools import MWeval
    valset = MWeval(gt_path)
    summary = valset.evaluate_dtList(dt_json=dts_json, metric=metric)
    print(summary)


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_path', type=str,
                        default='images/tiny_val/one')
    parser.add_argument('--gt_path', type=str,
                        default='images/tiny_val/one.json')
    parser.add_argument('--metric', type=str,
                        default='AP',
                        choices=['AP', 'F', 'counting'])
    args = parser.parse_args()

    # initialize RAPiD
    rapid = Detector(model_name='rapid',
                     weights_path='./weights/pL1_MWHB1024_Mar11_4000.ckpt')

    # Run RAPiD on the image sequence
    conf_thres = 0.005 if args.metric == 'AP' else 0.3
    dts_json = rapid.detect_imgSeq(args.imgs_path, input_size=1024, conf_thres=conf_thres)

    # Calculate metric
    if args.metric == 'AP':
        # the eval_cepdof_api() and eval_custom() are equivalent in terms of AP
        print('-------------------Evaluate using cepdof_api.py-------------------')
        eval_cepdof_api(args.gt_path, dts_json)
        print('-------------------Evaluate using MWtools.py-------------------')
        eval_custom(args.gt_path, dts_json, args.metric)

    elif args.metric == 'F':
        # Precision, Recall, and F-measure
        print('-------------------Evaluate using MWtools.py-------------------')
        eval_custom(args.gt_path, dts_json, args.metric)

    elif args.metric == 'counting':
        # Object (people) counting
        print('-------------------Evaluate using MWtools.py-------------------')
        eval_custom(args.gt_path, dts_json, args.metric)
