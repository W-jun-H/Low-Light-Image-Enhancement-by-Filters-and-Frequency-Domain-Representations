import torch
import models.archs.LLIE_F2 as LLIE_F2

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'LLIE_F2':
        netG = LLIE_F2.LLIE_F2(
            y_nf=opt_net['latt_nf'],
            f_nf=opt_net['first_stage_nf'],
            s_nf=opt_net['second_stage_nf'],
            filter_config=opt_net.get('filter', {}),
        )
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
