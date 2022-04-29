import torch
import os
import struct
from models.retinaface import RetinaFace
from data import cfg_mnet, cfg_re50
import torch.backends.cudnn as cudnn
import argparse

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',type=str)
parser.add_argument('--network', default='mobile0.25', help='mobile0.25 or resnet50')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

  

if __name__ == '__main__':
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
        cfg["pretrain"] = False
    elif args.network == "resnet50":
        cfg = cfg_re50
        cfg["pretrain"] = False
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, False)
    net.eval()
    print('Finished loading pytorch model!')
    cudnn.benchmark = True
    device = torch.device("cuda")
    net = net.to(device)

    if os.path.exists('retinaface.wts'):
        print("Error retinaface.wts already exist!")
    else:
        f = open("retinaface.wts", 'w')
        f.write("{}\n".format(len(net.state_dict().keys())))
        for k,v in net.state_dict().items():
            print('key: ', k)
            print('value: ', v.shape)
            vr = v.reshape(-1).cpu().numpy()
            f.write("{} {}".format(k, len(vr)))
            for vv in vr:
                f.write(" ")
                f.write(struct.pack(">f", float(vv)).hex())
            f.write("\n")
        print("success")

