WINDOW_SIZE = 5
MIN_SEQ_LEN = 7
MAX_SEQ_LEN = 7
new_subj_Th = 10



#*******************************************
#*******************************************
DEV = 'cuda:1'
HORIZON = 7
iteration = 0
#*******************************************
#*******************************************

n_mech = 11
n_drug = 71+26
n_ddi = 44

N_Urea = 200
N_SCRr = 13
N_eGFR = 200
N_eGFR_Class = 4
    

ablation = 0




start_seq = MAX_SEQ_LEN - MIN_SEQ_LEN
end_seq = MAX_SEQ_LEN

Sequences = list(range(MAX_SEQ_LEN))

T = end_seq - start_seq
DEVICE = DEV


# Hyperparameters
batch_size  = 1024
EPOCHS      = 400
LR          = 3e-4
SEED        = 42