from torch import tensor

from biophysical_models.unit_conversions import convert

smith_2007_cvs = {
    'p_pl': tensor(convert(-4, 'mmHg')),
    'p_pl_affects_pu_and_pa': tensor(False),

    'mt.r': tensor(convert(0.0600, 'kPa s/l')),
    'av.r': tensor(convert(1.4000, 'kPa s/l')),
    'tc.r': tensor(convert(0.1800, 'kPa s/l')),
    'pv.r': tensor(convert(0.4800, 'kPa s/l')),
    'pul.r': tensor(convert(19., 'kPa s/l')),
    'sys.r': tensor(convert(140., 'kPa s/l')),

    'lvf.e_es': tensor(convert(454., 'kPa/l')),
    'rvf.e_es': tensor(convert(87., 'kPa/l')),
    'spt.e_es': tensor(convert(6500., 'kPa/l')),
    'vc.e_es': tensor(convert(1.5000, 'kPa/l')),
    'pa.e_es': tensor(convert(45., 'kPa/l')),
    'pu.e_es': tensor(convert(0.8000, 'kPa/l')),
    'ao.e_es': tensor(convert(94., 'kPa/l')),

    'lvf.v_d': tensor(convert(0.0050, 'l')),
    'rvf.v_d': tensor(convert(0.0050, 'l')),
    'spt.v_d': tensor(convert(0.0020, 'l')),
    'vc.v_d': tensor(convert(2.8300, 'l')),
    'pa.v_d': tensor(convert(0.1600, 'l')),
    'pu.v_d': tensor(convert(0.2000, 'l')),
    'ao.v_d': tensor(convert(0.8000, 'l')),

    'lvf.v_0': tensor(convert(0.0050, 'l')),
    'rvf.v_0': tensor(convert(0.0050, 'l')),
    'spt.v_0': tensor(convert(0.0020, 'l')),
    'pcd.v_0': tensor(convert(0.2000, 'l')),

    'lvf.lam': tensor(convert(15., '1/l')),
    'rvf.lam': tensor(convert(15., '1/l')),
    'spt.lam': tensor(convert(435., '1/l')),
    'pcd.lam': tensor(convert(30., '1/l')),

    'lvf.p_0': tensor(convert(0.1700, 'kPa')),
    'rvf.p_0': tensor(convert(0.1600, 'kPa')),
    'spt.p_0': tensor(convert(0.1480, 'kPa')),
    'pcd.p_0': tensor(convert(0.0667, 'kPa')),

    'e.a': tensor(1.),
    'e.b': tensor(80.),
    'e.hr': tensor(80.),

    'v_tot': tensor(convert(5.5, 'l')),
}

paeme_2011_cvs = {
    'p_pl': tensor(convert(-4, 'mmHg')),
    'p_pl_affects_pu_and_pa': tensor(True),

    'mt.l': tensor(convert(7.6967e-5, 'mmHg s^2/ml')),
    'tc.l': tensor(convert(8.0093e-5, 'mmHg s^2/ml')),
    'av.l': tensor(convert(1.2189e-4, 'mmHg s^2/ml')),
    'pv.l': tensor(convert(1.4868e-4, 'mmHg s^2/ml')),

    'mt.r': tensor(convert(0.0158, 'mmHg s/ml')),
    'tc.r': tensor(convert(0.0237, 'mmHg s/ml')),
    'av.r': tensor(convert(0.0180, 'mmHg s/ml')),
    'pv.r': tensor(convert(0.0055, 'mmHg s/ml')),
    'pul.r': tensor(convert(0.1552, 'mmHg s/ml')),  # from .m file, absent from Paeme 2011?
    'sys.r': tensor(convert(1.0889, 'mmHg s/ml')),  # from .m file, absent from Paeme 2011?

    'lvf.e_es': tensor(convert(2.8798, 'mmHg s/ml')),
    'rvf.e_es': tensor(convert(0.5850, 'mmHg s/ml')),
    'spt.e_es': tensor(convert(48.7540, 'mmHg s/ml')),
    'vc.e_es': tensor(convert(0.0059, 'mmHg s/ml')),
    'pa.e_es': tensor(convert(0.3690, 'mmHg s/ml')),
    'pu.e_es': tensor(convert(0.0073, 'mmHg s/ml')),
    'ao.e_es': tensor(convert(0.6913, 'mmHg s/ml')),  # 0.0 in Paeme 2011?

    'lvf.v_d': tensor(0.0),
    'rvf.v_d': tensor(0.0),
    'spt.v_d': tensor(convert(2, 'ml')),
    'vc.v_d':  tensor(0.0),
    'pa.v_d':  tensor(0.0),
    'pu.v_d':  tensor(0.0),
    'ao.v_d':  tensor(0.0),

    'lvf.v_0': tensor(0.0),
    'rvf.v_0': tensor(0.0),
    'spt.v_0': tensor(convert(2.0, 'ml')),
    'pcd.v_0': tensor(convert(200.0, 'ml')),

    'lvf.lam': tensor(convert(0.033, '1/ml')),
    'rvf.lam': tensor(convert(0.023, '1/ml')),
    'spt.lam': tensor(convert(0.435, '1/ml')),
    'pcd.lam': tensor(convert(0.030, '1/ml')),

    'lvf.p_0': tensor(convert(0.1203, 'mmHg')),
    'rvf.p_0': tensor(convert(0.2157, 'mmHg')),
    'spt.p_0': tensor(convert(1.1101, 'mmHg')),
    'pcd.p_0': tensor(convert(0.5003, 'mmHg')),

    'e.a': tensor(1.),
    'e.b': tensor(80.),
    'e.hr': tensor(80.),

    'v_tot': tensor(convert(1.5, 'l')),  # 5.5 in Paeme 2011 but only simulates stressed volume?
}
