
import numpy as np
import os


working_dir = '/'.join(os.path.abspath(__file__).replace('\\','/').split('/')[:-3])
os.chdir(working_dir)
import CMS_TPA_Finite_Elements as CMS


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""Model of 5 interconnected cavities"""


#%% Load model info from .rst and .full files
load_data = True
dof_f = 1
density_f = 1.225

subfolder = 'examples/acoustic_model'
# Define the domains
subdomains = {
            'F1': (f'{working_dir}/{subfolder}/model_data/F1/', dof_f, None, '', density_f, load_data),
            'F2': (f'{working_dir}/{subfolder}/model_data/F2/', dof_f, None, '', density_f, load_data),
            'F3': (f'{working_dir}/{subfolder}/model_data/F3/', dof_f, None, '', density_f, load_data),
            'F4': (f'{working_dir}/{subfolder}/model_data/F4/', dof_f, None, '', density_f, load_data),
            'F5': (f'{working_dir}/{subfolder}/model_data/F5/', dof_f, None, '', density_f, load_data),
              }


dom_i = {'F1':0,'F2':1, 'F3':2, 'F4':3, 'F5':4, 'F6':5
}


# Define the interface pairs
interface_pairs = {
            # Interfaces first level
            1: [('INT_1_S1_TO_S2', 'INT_1_S1_TO_S2'), ('F1', 'F2'), (dom_i['F1'], dom_i['F2']),([],[]),('NODE-NODE',0.01),True, load_data],
            2: [('INT_2_S1_TO_S3', 'INT_2_S1_TO_S3'), ('F1', 'F3'), (dom_i['F1'], dom_i['F3']),([],[]),('NODE-NODE',0.01),True, load_data],
            3: [('INT_3_S1_TO_S4', 'INT_3_S1_TO_S4'), ('F1', 'F4'), (dom_i['F1'], dom_i['F4']),([],[]),('NODE-NODE',0.01),True, load_data],
            
            # Interfaces second level
            4: [('INT_4_S2_TO_S5', 'INT_4_S2_TO_S5'), ('F2', 'F5'), (dom_i['F2'], dom_i['F5']),([],[]),('NODE-NODE',0.01),True, load_data],
            5: [('INT_5_S3_TO_S5', 'INT_5_S3_TO_S5'), ('F3', 'F5'), (dom_i['F3'], dom_i['F5']),([],[]),('NODE-NODE',0.01),True, load_data],
            6: [('INT_6_S4_TO_S5', 'INT_6_S4_TO_S5'), ('F4', 'F5'), (dom_i['F4'], dom_i['F5']),([],[]),('NODE-NODE',0.01),True, load_data]

}


jobcase = 'coupling_example'
job_folder = f'{working_dir}/{subfolder}/jobcases/{jobcase}/'
model_name = f'{jobcase}.joblib'


# Get the component matrices and the coupling information
component_matrices, coupling_info = CMS.build_ds_models_vibroacoustic(subdomains, interface_pairs, save_info=(True, job_folder, model_name))

#%% Divisions of the Transfer Path Analysis: all components

level_components = {'AP': ['F1', 'F2', 'F3', 'F4', 'F5'],
                    'P1': ['F2', 'F3', 'F4', 'F5',],
                    'P2': ['F5'],
                    }


level_interfaces = {'AP': [],
                    'P1': [1,2,3],
                    'P2': [4,5,6],
                }


#%% Calculate the Transfer Path Analysis matrices
save_job_info = (True, job_folder, model_name)
red_method = 'FDCBM'
n_modes = {'F1':[1,20], 'F2':[1,20],'F3':[1,20],'F4':[1,20],'F5':[1,20]} # in case CMS methods are selected
Mat_red, Mat_pres, Coup_info, time_RAM = CMS.TPA_matrices_vibroacoustic(component_matrices, coupling_info, level_components, level_interfaces, CMS_method=[red_method,n_modes],  save_job_info = save_job_info)


#%% Prepare the load case
ref_comp_shapes_f = sum([component_matrices['shapes'][comp] for comp in coupling_info['components'] if (len(comp.split('_'))>1 and comp[0]=='F')])

# Select the frequency range
freqs = np.arange(1,100,10)

# Select the load case: distributed nodal force in X direction applied in 'FORCE_EXT_STR_S1'
load_case = {
            'F1': ['NODAL_FORCE', 'FORCE_EXT_S1', 1, ['P'], True],
            'F2': ['NONE'],
            'F3': ['NONE'],
            'F4': ['NONE'],
            'F5': ['NONE'],
            'F-REF': ['NONE',ref_comp_shapes_f],
}


# External force
force_ext = CMS.create_load(subdomains, load_case)


#%% Run the load case

# Settings
analysis_settings_f = ('HARMONIC', freqs, 'mkl_pardiso')

# Saving data settings
study_case = f'_{int(freqs[0])}-{int(freqs[-1])}Hz_{len(freqs)}'
job_name = f"{model_name.split('.')[0]}{study_case}.joblib" #modify this
mat_job_name = model_name # matrices model name (load)
save_job_info = (True, job_folder, job_name)
load_matrices_info = (True, job_folder, mat_job_name)



# Run the analysis
X, Coup_info, TPA_CPU_RAM = CMS.TPA_CMS_vibroacoustic(component_matrices, coupling_info, level_components, level_interfaces, force_ext,  
                                           analysis_settings_f, type_TPA='1L-TPA', CMS_method=[red_method,n_modes], complex_analysis=True,
                                           save_job_info = save_job_info, load_matrices_info=load_matrices_info)




#%% Analyse the contributions
descriptions = {'0':'FULL ANALYSIS', #AP
                '1':'F1 TO F2 (1)',
                '2':'F1 TO F3 (2)',
                '3':'F1 TO F4 (3)',
                '4':'F2 TO F5 (4)',
                '5':'F3 TO F5 (5)',
                '6':'F4 TO F5 (6)',
                }

# Select the level
level = 'P2'

# Components
comp_sel_f = ['F1','F2','F3','F4','F5'] # Acoustic
comp_sel_f = ['F5'] # Acoustic
components = ([], comp_sel_f)


# Select the path name
all_path_names = {'AP':['0'], 'P1':['1','2','3','ALL'], 'P2':['4','5','6','ALL']}
path_names = all_path_names[level]

# Define the variable that wants to be plotted
var = 'SPL'
mag = {'UX':'[m]', 'UY':'[m]', 'P':'[Pa]','SPL':'[dB]','SPL-A':'[dBA]' ,
       'PABS':'[Pa]', 'UNORM' :'[m]', 'ULOG':'[dB]'} # magnitudes

# Define the frequency set
freq_set = np.arange(0,len(freqs),1)

# Define the section plane
section_plane = [None,[]]

# Define the selected nodes for the averaging
nodes_sel = [] # if [], selects all nodes of the selected component(s)

# Select the y limit
ylim = [-120, 120]

# Define the plot types
plot_types = ['mag_strips', 'mag_lines', 'mag_bars',
              'contr_strips', 'contr_lines', 'contr_bars']
in_oct_bands = [False, False, True, 
                False ,False, True]
titles = ['Magnitude of paths', 'Magnitude of paths', 'Magnitude of paths',
          'Path contributions', 'Path contributions', 'Path contributions']

for j in range(0,len(plot_types)):
    filename = ''
    cont, amp_phases, path_name_title, freq_range_names= CMS.plot_TPA_contribution_results_3(X, Coup_info, level, components, path_names, var, freq_set, section = [None,], nodes_sel=nodes_sel,
                                                                                             ylabel=f'{var} {mag[var]}', cmap_name="jet", xlabel='Frequency [Hz]',
                                                                                             ylim=ylim, in_n_octave_bands=[in_oct_bands[j],3,[20,500]], descriptions=descriptions,
                                                                                             plot_title=titles[j],
                                                                                             fontsize=24, figsize=(24,8), legend=True, plot_type=plot_types[j],
                                                                                             # savefig=filename,
                                                                                             )



#%% Plot using pyvista
descriptions = {'0':'FULL ANALYSIS', #AP
                '1':'F1 TO F2 (1)',
                '2':'F1 TO F3 (2)',
                '3':'F1 TO F4 (3)',
                '4':'F2 TO F5 (4)',
                '5':'F3 TO F5 (5)',
                '6':'F4 TO F5 (6)',
                }

# Select the level
level = 'P2'

# Components
comp_sel_f = ['F1','F2','F3','F4','F5'] # Acoustic
components = ([], comp_sel_f)


# Select the path name
all_path_names = {'AP':['0'], 'P1':['1','2','3','ALL'], 'P2':['4','5','6','ALL']}
path_names = all_path_names[level]

# Define the variable that wants to be plotted
var = 'PABS'
mag = {'UX':'[m]', 'UY':'[m]', 'P':'[Pa]','SPL':'[dB]','SPL-A':'[dBA]' ,
       'PABS':'[Pa]', 'UNORM' :'[m]', 'ULOG':'[dB]'} # magnitudes

# Define the frequency set
freq_highlight = 321 # -> at this frequency, several paths are cancelling each other
freq_set = [np.argmin(np.abs(X['AP']['0']['stamps'] - freq_highlight))]

# Define the section plane
section_plane = [None,[1.9,2.2]]


cont = CMS.plot_TPA_results_pyvista(X, Coup_info, level, components, path_names, var, freq_set, def_factor=1e8, show_real_imag_values='real', plot_size=(1900,1000),
                                descriptions=descriptions, share_clim=True,  orientation=(1,1,-1), roll_angles=(0,0,0), show_min_max=True, plot_contributions=False, 
                                show_edges=False, make_zeros_transparent=False, vector_scale = True, section = section_plane,
                                parallel_projection = True, result_on_node=False)    








