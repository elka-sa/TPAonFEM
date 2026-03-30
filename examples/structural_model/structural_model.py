
import numpy as np
import os


working_dir = '/'.join(os.path.abspath(__file__).replace('\\','/').split('/')[:-3])
os.chdir(working_dir)
import CMS_TPA_Finite_Elements as CMS



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""Model of the 2 story building"""


#%% Load model info from .rst and .full files
load_data = True
dof_s, dof_f = 6, 1
density_s = 1.0

subfolder = 'examples/structural_model'
# Define the domains
subdomains = {
            'S1': (f'{working_dir}/{subfolder}/model_data/S1/', dof_s, None, '', density_s, load_data),
            'S2': (f'{working_dir}/{subfolder}/model_data/S2/', dof_s, None, '', density_s, load_data),
            'S3': (f'{working_dir}/{subfolder}/model_data/S3/', dof_s, None, '', density_s, load_data),
            'S4': (f'{working_dir}/{subfolder}/model_data/S4/', dof_s, None, '', density_s, load_data),
            'S5': (f'{working_dir}/{subfolder}/model_data/S5/', dof_s, None, '', density_s, load_data),
            'S6': (f'{working_dir}/{subfolder}/model_data/S6/', dof_s, None, '', density_s, load_data),
              }


dom_i = {'S1':0,'S2':1, 'S3':2, 'S4':3, 'S5':4, 'S6':5
}


# Define the interface pairs
interface_pairs = {
            1: [('INT_S1_S2_1', 'INT_S2_S1_1'), ('S1', 'S2'), (dom_i['S1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data],
            2: [('INT_S1_S2_2', 'INT_S2_S1_2'), ('S1', 'S2'), (dom_i['S1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data],
            3: [('INT_S1_S3_1', 'INT_S3_S1_1'), ('S1', 'S3'), (dom_i['S1'], dom_i['S3']),([],[]),('NODE-NODE',0.01),True, load_data],
            4: [('INT_S1_S4_1', 'INT_S4_S1_1'), ('S1', 'S4'), (dom_i['S1'], dom_i['S4']),([],[]),('NODE-NODE',0.01),True, load_data],
            5: [('INT_S3_S6_1', 'INT_S6_S3_1'), ('S3', 'S6'), (dom_i['S3'], dom_i['S6']),([],[]),('NODE-NODE',0.01),True, load_data],
            6: [('INT_S4_S5_1', 'INT_S5_S4_1'), ('S4', 'S5'), (dom_i['S4'], dom_i['S5']),([],[]),('NODE-NODE',0.01),True, load_data],
            7: [('INT_S5_S6_1', 'INT_S6_S5_1'), ('S5', 'S6'), (dom_i['S5'], dom_i['S6']),([],[]),('NODE-NODE',0.01),True, load_data],
            8: [('INT_S5_S6_2', 'INT_S6_S5_2'), ('S5', 'S6'), (dom_i['S5'], dom_i['S6']),([],[]),('NODE-NODE',0.01),True, load_data],
}



jobcase = 'coupling_example'
job_folder = f'{working_dir}/{subfolder}/jobcases/{jobcase}/'
model_name = f'{jobcase}.joblib'



# Get the component matrices and the coupling information
component_matrices, coupling_info = CMS.build_ds_models_vibroacoustic(subdomains, interface_pairs, save_info=(True, job_folder, model_name))

#%% Divisions of the Transfer Path Analysis: all components

level_components = {'AP': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
                    'P1': ['S2', 'S3', 'S4', 'S5', 'S6'],
                    'P2': ['S5', 'S6'],
                    'P3': ['S6'],
                    }


level_interfaces = {'AP': [],
                    'P1': [3,4],
                    'P2': [5,6],
                    'P3': [5,7,8],
                }


#%% Calculate the Transfer Path Analysis matrices
save_job_info = (True, job_folder, model_name)
red_method = 'FDCBM'
n_modes = {'S1':20, 'S2':20,'S3':20,'S4':20,'S5':20,'S6':20} # in case CMS methods are selected
Mat_red, Mat_pres, Coup_info, time_RAM = CMS.TPA_matrices_vibroacoustic(component_matrices, coupling_info, level_components, level_interfaces, CMS_method=[red_method,n_modes],  save_job_info = save_job_info)


#%% Prepare the load case
ref_comp_shapes_s = sum([component_matrices['shapes'][comp] for comp in coupling_info['components'] if (len(comp.split('_'))>1 and comp[0]=='S')])

# Select the frequency range
freqs = np.arange(1,100,1)

# Select the load case: distributed nodal force in X direction applied in 'FORCE_EXT_STR_S1'
load_case = {
            'S1': ['NODAL_FORCE', 'FORCE_A1', 1, ['Y'], True],
            'S2': ['NONE'],
            'S3': ['NONE'],
            'S4': ['NONE'],
            'S5': ['NONE'],
            'S6': ['NONE'],
            'S-REF': ['NONE',ref_comp_shapes_s],
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
                '1':'S1 TO S2 (1)',
                '2':'S1 TO S2 (2)',
                '3':'S1 TO S3 (3)',
                '4':'S1 TO S4 (4)',
                '5':'S3 TO S6 (5)',
                '6':'S4 TO S5 (6)',
                '7':'S5 TO S6 (7)',
                '8':'S5 TO S6 (8)',
                }

# Select the level
level = 'P3'

# Components
comp_sel_s = ['S1','S2','S3','S4','S5','S6'] # Structural
comp_sel_s = ['S6'] # Structural
components = (comp_sel_s, [])


# Select the path name
all_path_names = {'AP':['0'], 'P1':['3','4','ALL'], 'P2':['5','6','ALL'], 'P3':['5','7','8','ALL']}
path_names = all_path_names[level]

# Define the variable that wants to be plotted
var = 'ULOG'
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
in_oct_bands = [False, False, False, 
                False ,False, False]
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
                '1':'S1 TO S2 (1)',
                '2':'S1 TO S2 (2)',
                '3':'S1 TO S3 (3)',
                '4':'S1 TO S4 (4)',
                '5':'S3 TO S6 (5)',
                '6':'S4 TO S5 (6)',
                '7':'S5 TO S6 (7)',
                '8':'S5 TO S6 (8)',
                }

# Select the level
level = 'P3'

# Components
comp_sel_s = ['S1','S2','S3','S4','S5','S6'] # Structural
components = (comp_sel_s, [])


# Select the path name
all_path_names = {'AP':['0'], 'P1':['3','4','ALL'], 'P2':['5','6','ALL'], 'P3':['5','7','8','ALL']}
path_names = all_path_names[level]

# Define the variable that wants to be plotted
var = 'UVECT'

# Define the frequency set
freq_highlight = 21 # -> at this frequency, several paths are cancelling each other
freq_set = [np.argmin(np.abs(X['AP']['0']['stamps'] - freq_highlight))]

# Define the section plane
section_plane = [None,[1.9,2.2]]


cont = CMS.plot_TPA_results_pyvista(X, Coup_info, level, components, path_names, var, freq_set, def_factor=1e8, show_real_imag_values='real', plot_size=(1900,1000),
                                descriptions=descriptions, share_clim=True,  orientation=(0,0,1), roll_angles=(0.1,2,2), show_min_max=True, plot_contributions=False, 
                                show_edges=False, make_zeros_transparent=False, vector_scale = True, section = section_plane,
                                parallel_projection = True, result_on_node=False)    











