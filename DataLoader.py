# Load data
data = np.load(f"../filtered_data/dataset_win-{WINDOW_SIZE}_seq-{MIN_SEQ_LEN}-{MAX_SEQ_LEN}_horiz-{HORIZON}_subj-{new_subj_Th}_v3_fold-{iteration}_rz35.npz")

X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']
Subject_test = data['Subject_test']

data.close()



def Perproc(X):

    X0 = X.copy()

    # Remove additional features
    scr = X0[:,-1,5]
    age = X0[:,-1,-2]
    sex = X0[:,-1,-1]
    X[:,:,12: 12 + n_mech + n_drug + n_ddi][X[:,:, 12: 12 + n_mech + n_drug + n_ddi]>1] = 1
    X[:, Sequences, 6:8] = 0
    X[:, Sequences, 12: 12 + n_mech] = 0
    X[:, Sequences, 0:5] = 0
    X[:, Sequences, 8:11] = 0

    # Normalization
    feat = np.concatenate([
            X[:, start_seq:end_seq, 0:6]/N_SCRr,       # SCr
            X[:, start_seq:end_seq, 11:12]/N_Urea,      # Urea
            X[:, start_seq:end_seq, 12 + n_mech + n_drug: 12 + n_mech + n_drug + n_ddi],            # DDI types
            (X[:, start_seq:end_seq, -2:-1]-18)/(91-18),      # Age
            X[:, start_seq:end_seq, -1:]               # Sex
    ], axis=2)
    return feat, scr, age, sex


X_train, scr_train, age_train, sex_train = Perproc(X_train)
X_val, scr_val, age_val, sex_val   = Perproc(X_val)
X_test, scr_test, age_test, sex_test  = Perproc(X_test)

num_features = X_train.shape[-1]
print(f"num_features: {num_features}")
print(f"y_train: {y_train.shape}")




torch.manual_seed(SEED)
print(f"Using device: {DEVICE}")



y_egfr_train = compute_egfr_tensor(y_train, age_train, sex_train).cpu().numpy()
y_egfr_val = compute_egfr_tensor(y_val, age_val, sex_val).cpu().numpy()
y_egfr_test = compute_egfr_tensor(y_test, age_test, sex_test).cpu().numpy()
baseline_egfr_test = compute_egfr_tensor(scr_test, age_test, sex_test).cpu().numpy()


X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
y_train_t = torch.as_tensor(y_egfr_train, dtype=torch.float32)

X_val_t = torch.as_tensor(X_val, dtype=torch.float32)
y_val_t = torch.as_tensor(y_egfr_val, dtype=torch.float32)

X_test_t = torch.as_tensor(X_test, dtype=torch.float32)
y_test_t = torch.as_tensor(y_egfr_test, dtype=torch.float32)

print("Any NaNs in Train Features?", torch.isnan(X_train_t).any().item())
print("Any Infs in Train Features?", torch.isinf(X_train_t).any().item())



# 2. Create the Datasets using the new Tensors
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t, y_val_t)
test_dataset  = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader  =  DataLoader(val_dataset,  batch_size=batch_size, shuffle=False)
test_loader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=False)





feature_columns = ['SCr_mean','SCr_STD','SCr_min','SCr_max','SCr_slope','SCr_last','eGFR_VALUE','eGFR_CLASS','Urea_mean','Urea_STD','Urea_min',
                   'Urea_max'] + [f'MECH_{i}' for i in range(n_mech)] + [f'Drug_{i}' for i in range(n_drug)] + [f'DDI_{i}' for i in range(n_ddi)] + ['Age', 'Sex']

drug_names_list = [
        'Cyclosporine', 'Valsartan', 'Ramipril', 'Pantoprazole', 'Torsemide',
        'Iopamidol', 'Diatrizoate', 'Olmesartan', 'Tenofovir disoproxil', 'Flucloxacillin',
        'Chlorthalidone', 'Piperacillin', 'Indomethacin', 'Omeprazole', 'Cidofovir',
        'Triamterene', 'Zoledronic acid', 'Ampicillin', 'Spironolactone', 'Allopurinol',
        'Trimethoprim', 'Lansoprazole', 'Ketorolac', 'Amikacin', 'Celecoxib',
        'Dicloxacillin', 'Fosinopril', 'Cimetidine', 'Vancomycin', 'Cisplatin',
        'Trandolapril', 'Metolazone', 'Foscarnet', 'Ciprofloxacin', 'Piroxicam',
        'Methotrexate', 'Cephalexin', 'Enalapril', 'Diclofenac', 'Amiloride',
        'Nafcillin', 'Pemetrexed', 'Losartan', 'Amphotericin B', 'Warfarin',
        'Tobramycin', 'Furosemide', 'Eplerenone', 'Oxacillin', 'Lisinopril',
        'Esomeprazole', 'Etodolac', 'Clopidogrel', 'Meropenem', 'Acyclovir',
        'Naproxen', 'Perindopril', 'Gentamicin', 'Colistin', 'Indapamide',
        'Meloxicam', 'Tacrolimus', 'Eprosartan', 'Quinapril', 'Bumetanide',
        'Ethacrynic acid', 'Aspirin (ASA)', 'Carboplatin', 'Telmisartan', 'Methicillin',
        'Adenofovir', 'Hydrochlorothiazide', 'Irbesartan', 'Rifampin', 'Ibuprofen',
        'Amoxicillin', 'Heparin', 'Cefuroxime', 'Levofloxacin', 'Captopril',
        'Ceftriaxone', 'Enoxaparin', 'Cilazapril', 'Iohexol', 'Cefepime',
        'Ticarcillin / Tazobactam', 'Nedaplatin', 'Dronedarone', 'Rivaroxaban', 'Apixaban',
        'Ticagrelor', 'Canagliflozin', 'Dolutegravir', 'Empagliflozin', 'Cobicistat',
        'Iodixanol', 'Candesartan',
    ]
ddi_names_list = [
        'Decreased metabolism', 'Increased hyperkalemia', 'Increased serum conc',
        'Increased metabolism', 'Increased hyperkalemic act',
        'lower serum level & reduction in efficacy', 'Decrease in efficacy',
        'Increased bleeding', 'Increased nephrotoxicity', 'Increased hepatotoxic act',
        'Decreased bioavailability', 'Increased bleeding&hemorrhage',
        'Decreased serum conc', 'Increased anticoagulant act', 'Increased GI-bleeding',
        'Increased nephrotoxic act', 'Increased immunosuppressive act',
        'Decreased protein-binding', 'Decreased anticoagulant', 'Loss in efficacy',
        'Increased renal-failure&hyperkalemia&hypertension', 'Decreased absorption',
        'Increased serum conc(A) & active metab(B)', 'Worsening of adverse effects',
        'Increased hypokalemic act', 'Decreased serum conc(A) & active metab(B)',
        'Increased antiplatelet act', 'Increased myopathic rhabdomyolysis act',
        'Increased renal-failure&hypotension&hyperkalemia', 'Decreased hypoglycemic act',
        'Increased ototoxicity&nephrotoxicity', 'Increased bioavailability',
        'Increased bleeding&GI-bleeding', 'Increased hypoglycemic act',
        'Increased nephrotoxicity&hypocalcemia', 'Increased myelosuppressive act',
        'Increased protein-binding', 'Increased hyperkalemia&acidosis',
        'Increased thrombogenic act', 'Increased nephrotoxic&ototoxic act',
        'Decreased therapeutic efficacy', 'Decreased nephrotoxic act',
        'Increased hypotension&hyperkalemia&reduced intravascular vol',
        'Increased hypercalcemic act',
    ]
feature_column_names = ['SCr_mean','SCr_STD','SCr_min','SCr_max','SCr_slope','SCr_last','eGFR_VALUE','eGFR_CLASS','Urea_mean','Urea_STD','Urea_min',
                   'Urea_max'] + [f'MECH_{i}' for i in range(n_mech)] + drug_names_list + ddi_names_list + ['Age', 'Sex']

