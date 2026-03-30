import numpy as np
import os


working_dir = '/'.join(os.path.abspath(__file__).replace('\\','/').split('/')[:-3])
os.chdir(working_dir)
import CMS_TPA_Finite_Elements as CMS

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""Model of the 4 structure + 2 acoustic subdomains."""


#%% Load model info from .rst and .full files
load_data = True
dof_s, dof_f = 6, 1
density_s, density_f = 1.0, 1.225

subfolder = 'examples/vibro-acoustic_model'
# Define the domains
subdomains    = { 
                    'S1': (f'{working_dir}/{subfolder}/model_data/S1/', dof_s, None, '', density_s, load_data), # Emitter panel
                    'S2': (f'{working_dir}/{subfolder}/model_data/S4/', dof_s, None, '', density_s, load_data), # Center panel
                    'S3': (f'{working_dir}/{subfolder}/model_data/S2/', dof_s, None, '', density_s, load_data), # Receiver panel
                    'S4': (f'{working_dir}/{subfolder}/model_data/S3/', dof_s, None, '', density_s, load_data), # Intermediate panel
                    'F1': (f'{working_dir}/{subfolder}/model_data/F1/', dof_f, None, '', density_f, load_data), # First cavity
                    'F2': (f'{working_dir}/{subfolder}/model_data/F2/', dof_f, None, '', density_f, load_data), # Second cavity
                 }

dom_i = {'S1':0,'S2':1, 'S3':2, 'S4':3, 'F1':4, 'F2':5

          }


interface_pairs = {
    # Fluid-structure interfaces
    1:[('INT_1_F1_TO_S1', 'INT_1_S1_TO_F1'),('F1','S1'),(dom_i['F1'], dom_i['S1']),([],[]),('NODE-NODE',0.01),True, load_data], # Emitter panel to first cavity
    3:[('INT_2_F1_TO_S2', 'INT_2_S2_TO_F1'),('F1','S3'),(dom_i['F1'], dom_i['S3']),([],[]),('NODE-NODE',0.01),True, load_data], # Center panel to first cavity
    6:[('INT_3_F2_TO_S2', 'INT_3_S2_TO_F2'),('F2','S3'),(dom_i['F2'], dom_i['S3']),([],[]),('NODE-NODE',0.01),True, load_data], # Center panel to second cavity
    7:[('INT_4_F2_TO_S3', 'INT_4_S3_TO_F2'),('F2','S4'),(dom_i['F2'], dom_i['S4']),([],[]),('NODE-NODE',0.01),True, load_data], # Receiver panel to second cavity 
    8:[('INT_8_F1_TO_S4', 'INT_8_S4_TO_F1'),('F1','S2'),(dom_i['F1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], # Bridge panel to first cavity
    
    
    # Structure-structure interfaces
    2:[('INT_7_S1_TO_S4', 'INT_7_S4_TO_S1'),('S1','S2'),(dom_i['S1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], # First panel to bridge panel
    5:[('INT_6_S4_TO_S2', 'INT_6_S2_TO_S4'),('S2','S3'),(dom_i['S2'], dom_i['S3']),([],[]),('NODE-NODE',0.01),True, load_data], # Second panel to bridge panel
    
    # Acoustic-acoustic interfaces
    4:[('INT_5_F1_TO_F2', 'INT_5_F2_TO_F1'),('F1','F2'),(dom_i['F1'], dom_i['F2']),([],[]),('NODE-NODE',0.01),True, load_data], # First cavity to second cavity

}


jobcase = 'coupling_example'
job_folder = f'{working_dir}/{subfolder}/jobcases/{jobcase}/'
model_name = f'{jobcase}.joblib'

# Get the component matrices and the coupling information
component_matrices, coupling_info = CMS.build_ds_models_vibroacoustic(subdomains, interface_pairs, save_info=(True, job_folder, model_name))



#%% Divisions of the Transfer Path Analysis: all vibro-acoustic components
level_components = {
                    'AP': ['S1','S2','S3','S4','F1','F2'], # all domains
                    'P1': ['S2','S3','S4','F1','F2'], 
                    'P2': ['S3','S4','F2'],
                    'P3': ['F2']
                    }


level_interfaces = {'AP': [],
                    'P1': [1, 2],
                    'P2': [3, 4, 5],
                    'P3': [4, 6, 7]
                }


#%% Calculate the Transfer Path Analysis matrices
save_job_info = (True, job_folder, model_name)
red_method = 'DUAL'
n_modes = {'S1':20, 'S2':20,'S3':20,'S4':80,'F1':20,'F2':20} # in case CMS methods are selected
Mat_red, Mat_pres, Coup_info, time_RAM = CMS.TPA_matrices_vibroacoustic(component_matrices, coupling_info, level_components, level_interfaces, CMS_method=[red_method,n_modes],  save_job_info = save_job_info)


#%% Prepare the load case
ref_comp_shapes_s = sum([component_matrices['shapes'][comp] for comp in coupling_info['components'] if (len(comp.split('_'))>1 and comp[0]=='S')])
ref_comp_shapes_f = sum([component_matrices['shapes'][comp] for comp in coupling_info['components'] if (len(comp.split('_'))>1 and comp[0]=='F')])

# Select the frequency range
freqs = np.arange(1,100,10)

# Select the load case: distributed nodal force in X direction applied in 'FORCE_EXT_STR_S1'
load_case = {
            'S1': ['NODAL_FORCE', 'FORCE_EXT_STR_S1', 1, ['X'], True],
            'S2': ['NONE'],
            'S3': ['NONE'],
            'S4': ['NONE'],
            'S-REF': ['NONE',ref_comp_shapes_s],
            'F1': ['NONE'],
            'F2': ['NONE'],
            'F-REF': ['NONE', ref_comp_shapes_f],
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
                '1':'S1 TO F1',
                '2':'S1 TO S2',
                '3':'S3 TO F1',
                '4':'F1 TO F2',
                '5':'S2 TO S3',
                '6':'S3 TO F2',
                '7':'S4 TO F2',
                '8':'S2 TO F1',
                }


# Select the level
level = 'P2'

# Components
comp_sel_s = ['S1','S2','S3','S4'] # Structural
comp_sel_f = ['F2'] # fluid
components = (comp_sel_s, comp_sel_f)


# Select the path name
all_path_names = {'AP':['0'], 'P1':['1','2','ALL'], 'P2':['3','4','5','ALL'], 'P3':['4','6','7','ALL']}
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
                                                                                             fontsize=24, figsize=(20,8), legend=True, plot_type=plot_types[j],
                                                                                             savefig=filename,
                                                                                             )


#%% Plot using pyvista
descriptions = {'0':'FULL ANALYSIS', #AP
                '1':'S1 TO F1',
                '2':'S1 TO S2',
                '3':'S3 TO F1',
                '4':'F1 TO F2',
                '5':'S2 TO S3',
                '6':'S3 TO F2',
                '7':'S4 TO F2',
                '8':'S2 TO F1',
                }

# Select the level
level = 'P1'

# Components
comp_sel_s = ['S1','S2','S3','S4'] # Structural
comp_sel_f = ['F1','F2'] # fluid
components = (comp_sel_s, comp_sel_f)


# Select the path name
all_path_names = {'AP':['0'], 'P1':['1','2','ALL'], 'P2':['3','4','5','ALL'], 'P3':['4','6','7','ALL']}
path_names = all_path_names[level]

# Define the variable that wants to be plotted
var = 'PABS'

# Define the frequency set
freq_highlight = 21 # -> at this frequency, several paths are cancelling each other
freq_set = [np.argmin(np.abs(X['AP']['0']['stamps'] - freq_highlight))]

# Define the section plane
section_plane = [None,[1.9,2.2]]


cont = CMS.plot_TPA_results_pyvista(X, Coup_info, level, components, path_names, var, freq_set, def_factor=1e5, show_real_imag_values='real', plot_size=(1900,1000),
                                descriptions=descriptions, share_clim=True, orientation=(-1,1,1), show_min_max=True, plot_contributions=True, 
                                show_edges=True, make_zeros_transparent=True, vector_scale = True, section = section_plane,
                                parallel_projection = True, result_on_node=False)    



