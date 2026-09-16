def compute_egfr_tensor(scr, age, sex):

    scr = torch.tensor(scr, dtype=torch.float32)
    age = torch.as_tensor(age, dtype=torch.float32)#.to(dev)
    sex = torch.as_tensor(sex, dtype=torch.float32)#.to(dev)
    
    age = age.view(scr.shape)
    sex = sex.view(scr.shape)
    
    kappa = torch.where(sex==1, 0.7, 0.9)
    alpha = torch.where(sex==1, -0.241, -0.302)
    gender_mult = torch.where(sex==1, 1.012, 1.0)

    term1 = torch.minimum(scr / kappa, torch.ones_like(scr)) ** alpha
    term2 = torch.maximum(scr / kappa, torch.ones_like(scr)) ** -1.200
    term3 = 0.9938 ** age

    return 142 * term1 * term2 * term3 * gender_mult

    