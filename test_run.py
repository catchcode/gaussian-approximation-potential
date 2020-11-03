import dataset
import soap


# Upper case E implies that molecular energy E consists of atomice energies e's
# Lower case e/f refers to the atomic energy/force
# In a similar spirit, molecular variables R, Q, and so on below denote collective sets of atomic variables r, q, etc.
calc_e_from_E   = True  # Basic mode of computation
f_data_given    = False
calc_e_from_Ef  = calc_e_from_E and f_data_given # calculate e from E & f data
calc_f_from_rep = False # calculate f from the precomputed representative descriptors and alpha
calc_ef_from_E  = calc_e_from_E and calc_f_from_rep
calc_ef_from_Ef = calc_e_from_Ef and calc_f_from_rep
calc_ef_from_Ef_experimental = False    # same as calc_ef_from_Ef but with an experimental scheme 
                                        # extending a sparse cov matrix with force terms  

if precision == 'single':
    float_dtype = np.float32
    complex_dtype = np.complex64
elif precision == 'double':
    float_dtype = np.float64
    complex_dtype = np.complex128
else:
    float_dtype = np.float
    complex_dtype = np.complex

def run_s22a():
    '''Run s22 test case.'''
    import time
    t0 = time.time()

    train_set = dataset('data/s22-a.xyz')
    #print(train_set.total_force_list)
    #print(train_set.y)

    model = soap(lmax=4, nmax=6, sigma=0.3, rcut=5.0,
                 zeta=4, sigma_nu_energy=0.003)
    model.fit(train_set)

    test_set = train_set
    #test_set = dataset('data/s22-b.xyz')
    true_energies = test_set.total_energy_list
    model_energies = model.total_energies(test_set)

    df = pd.DataFrame(list(zip(true_energies, model_energies, test_set.y, model_energies-model.y0)), columns=['True', 'Model', 'y_true', 'y_model'])
    df['Diff'] = df['Model'] - df['True']
    print(df, end='\n')
    print(model)
    from sklearn.metrics import mean_absolute_error
    print('Mean Absolute Error (MAE) = {0.real:.4f}'.format(mean_absolute_error(df['True'], df['Model'])))
    if calc_f_from_rep:
        model_forces = model.total_forces_from_e(test_set)
        print(model_forces)
    t1 = time.time()
    print('Elapsed time = {0.real:.0f} seconds'.format(t1 - t0))

def run_s22b():
    '''Run s22 test case.'''
    import time
    t0 = time.time()

    train_set = dataset('data/s22-b.xyz')
    model = soap(lmax=4, nmax=6, sigma=0.3, rcut=5.0,
                 zeta=4, sigma_nu_energy=0.003)
    model.fit(train_set)

    test_set = train_set
    true_energies = test_set.total_energy_list
    model_energies = model.total_energies(test_set)

    df = pd.DataFrame(list(zip(true_energies, model_energies)), columns=['True', 'Model'])
    df['Diff'] = df['Model'] - df['True']
    print(df, end='\n')
    print(model)

    from sklearn.metrics import mean_absolute_error
    print('Mean Absolute Error (MAE) = {0.real:.4f}'.format(mean_absolute_error(df['True'], df['Model'])))

    if calc_f_from_rep:
        model_forces = model.total_forces_from_e(test_set)
        print(model_forces)

    t1 = time.time()
    print('Elapsed time = {0.real:.0f} seconds'.format(t1 - t0))

def run_s22ab():
    '''Run s22 test case.'''
    import time
    t0 = time.time()

    train_set = dataset('data/s22-ab.xyz')
    model = soap(lmax=4, nmax=6, sigma=0.3, rcut=5.0,
                 zeta=4, sigma_nu_energy=0.003)
    model.fit(train_set)

    test_set = train_set
    true_energies = test_set.total_energy_list
    model_energies = model.total_energies(test_set)

    df = pd.DataFrame(list(zip(true_energies, model_energies)), columns=['True', 'Model'])
    df['Diff'] = df['Model'] - df['True']
    print(df, end='\n')
    print(model)

    from sklearn.metrics import mean_absolute_error
    print('Mean Absolute Error (MAE) = {0.real:.4f}'.format(mean_absolute_error(df['True'], df['Model'])))

    if calc_f_from_rep:
        model_forces = model.total_forces_from_e(test_set)
        print(model_forces)

    t1 = time.time()
    print('Elapsed time = {0.real:.0f} seconds'.format(t1 - t0))

def run_s22():
    '''Run s22 test case.'''
    import time
    t0 = time.time()

    train_set = dataset('data/s22.xyz')
    model = soap(lmax=4, nmax=6, sigma=0.3, rcut=5.0,
                 zeta=4, sigma_nu_energy=0.003)
    model.fit(train_set)

    test_set = train_set
    true_energies = test_set.total_energy_list
    model_energies = model.total_energies(test_set)

    df = pd.DataFrame(list(zip(true_energies, model_energies)), columns=['True', 'Model'])
    df['Diff'] = df['Model'] - df['True']

    df1, df2, df3 = df[:22], df[22:44], df[44:]
    df1.index += 1
    df2.index = df1.index
    df3.index = df1.index
    df4 = df1.join(df2, lsuffix='_A', rsuffix='_B').join(df3)

    binding_energy = df1 + df2 - df3
    df5 = df4.join(binding_energy, lsuffix='_AB', rsuffix='_dE')
    print(df5, end='\n')

    print(model)

    from sklearn.metrics import mean_absolute_error
    print('Mean Absolute Error (MAE) = {0.real:.4f}'.format(mean_absolute_error(df5['True_dE'], df5['Model_dE'])))

    if calc_f_from_rep:
        model_forces = model.total_forces_from_e(test_set)
        print(model_forces)

    t1 = time.time()
    print('Elapsed time = {0.real:.0f} seconds'.format(t1 - t0))

    #plot_s22(df5['True_dE'], df5['Model_dE'])

def plot_s22(True_dE, Model_dE):
    '''Plot s22 binding energy results'''
    import numpy as np
    import matplotlib.pyplot as plt
    from ase.data.s22 import s22, get_interaction_energy_s22
    CCSD_dE = [get_interaction_energy_s22(mol) for mol in s22]

    x1 = -np.array(CCSD_dE)
    y1 = np.array(True_dE)
    y2 = np.array(Model_dE)

    x0 = np.linspace(-0.2, 0.95, 100)
    y0 = x0

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x0, y0)
    ax.scatter(x1, y1)
    ax.scatter(x1, y2)
    ax.axis([0, 0.95, 0, 0.95])
    ax.set_aspect(1.0)
    plt.savefig('s22.pdf', bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    # run_s22a()
    # run_s22b()
    # run_s22ab()
    run_s22()
