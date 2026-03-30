import numpy as np
import os


working_dir = '/'.join(os.path.abspath(__file__).replace('\\','/').split('/')[:-3])
os.chdir(working_dir)
import CMS_TPA_Finite_Elements as CMS

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""Model of a segment of an aircraft."""


#%% Load model info from .rst and .full files
load_data = True
dof_s, dof_f = 6, 1
density_s, density_f = 1.0, 1.225

subfolder = 'examples/cabin_model'
# Define the domains
subdomains    = { 
                    'S1': (f'{working_dir}/{subfolder}/model_data/S1/', dof_s, None, '', density_s, load_data), # Primary structure
                    'S2': (f'{working_dir}/{subfolder}/model_data/S2/', dof_s, None, '', density_s, load_data), # Secondary structure
                    'F1': (f'{working_dir}/{subfolder}/model_data/F1/', dof_f, None, '', density_f, load_data), # Secondary cavity
                    'F2': (f'{working_dir}/{subfolder}/model_data/F2/', dof_f, None, '', density_f, load_data), # Hatrack cavities
                    'F3': (f'{working_dir}/{subfolder}/model_data/F3/', dof_f, None, '', density_f, load_data), # Passengers cavity

                }

dom_i = {'S1':0,'S2':1, 'F1':2, 'F2':3, 'F3':4,
          }


interface_pairs = {

    # Structure-structure interfaces: from S1 to S2
    1:[('FLOOR_STR', 'FLOOR_STR'),('S1','S2'),(dom_i['S1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 
    2:[('DADO_STR_LEFT', 'DADO_STR_LEFT'),('S1','S2'),(dom_i['S1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 
    3:[('DADO_STR_RIGHT', 'DADO_STR_RIGHT'),('S1','S2'),(dom_i['S1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 
    4:[('SIDEWALL_STR_LEFT', 'SIDEWALL_STR_LEFT'),('S1','S2'),(dom_i['S1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data],
    5:[('SIDEWALL_STR_RIGHT', 'SIDEWALL_STR_RIGHT'),('S1','S2'),(dom_i['S1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data],
    6:[('COWL_STR_LEFT', 'COWL_STR_LEFT'),('S1','S2'),(dom_i['S1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data],
    7:[('COWL_STR_RIGHT', 'COWL_STR_RIGHT'),('S1','S2'),(dom_i['S1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data],
    8:[('HATRACK_STR_LEFT', 'HATRACK_STR_LEFT'),('S1','S2'),(dom_i['S1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 
    9:[('HATRACK_STR_RIGHT', 'HATRACK_STR_RIGHT'),('S1','S2'),(dom_i['S1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 
    10:[('CEILING_STR', 'CEILING_STR'),('S1','S2'),(dom_i['S1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 

    # Acoustic-acoustic interfaces: from F1 to F3
    20:[('VENTILATION_ACO_LEFT', 'VENTILATION_ACO_LEFT'),('F1','F3'),(dom_i['F1'], dom_i['F3']),([],[]),('NODE-NODE',0.01),True, load_data],
    21:[('VENTILATION_ACO_RIGHT', 'VENTILATION_ACO_RIGHT'),('F1','F3'),(dom_i['F1'], dom_i['F3']),([],[]),('NODE-NODE',0.01),True, load_data], 

    # Fluid-structure interfaces: from S1 to F1
    30:[('NODES_FUSELAGE', 'NODES_FUSELAGE'),('F1','S1'),(dom_i['F1'], dom_i['S1']),([],[]),('NODE-NODE',0.01),True, load_data], 
    
    # Fluid-structure interfaces: from F1 to S2
    40:[('NODES_FLOOR_VA', 'NODES_FLOOR_VA'),('F1','S2'),(dom_i['F1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 
    41:[('NODES_DADO_VA_LEFT', 'NODES_DADO_VA_LEFT'),('F1','S2'),(dom_i['F1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 
    42:[('NODES_DADO_VA_RIGHT', 'NODES_DADO_VA_RIGHT'),('F1','S2'),(dom_i['F1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 
    43:[('NODES_SIDEWALL_VA_LEFT', 'NODES_SIDEWALL_VA_LEFT'),('F1','S2'),(dom_i['F1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data],
    44:[('NODES_SIDEWALL_VA_RIGHT', 'NODES_SIDEWALL_VA_RIGHT'),('F1','S2'),(dom_i['F1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 
    45:[('NODES_COWL_VA_LEFT', 'NODES_COWL_VA_LEFT'),('F1','S2'),(dom_i['F1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 
    46:[('NODES_COWL_VA_RIGHT', 'NODES_COWL_VA_RIGHT'),('F1','S2'),(dom_i['F1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 
    47:[('NODES_HATRACK_SEC_VA_LEFT', 'NODES_HATRACK_SEC_VA_LEFT'),('F1','S2'),(dom_i['F1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data],
    48:[('NODES_HATRACK_SEC_VA_RIGHT', 'NODES_HATRACK_SEC_VA_RIGHT'),('F1','S2'),(dom_i['F1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data],
    49:[('NODES_CEILING_VA', 'NODES_CEILING_VA'),('F1','S2'),(dom_i['F1'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 

    # Fluid-structure interfaces: from S2 to F3
    50:[('NODES_FLOOR_VA', 'NODES_FLOOR_VA'),('F3','S2'),(dom_i['F3'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 
    51:[('NODES_DADO_VA_LEFT', 'NODES_DADO_VA_LEFT'),('F3','S2'),(dom_i['F3'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 
    52:[('NODES_DADO_VA_RIGHT', 'NODES_DADO_VA_RIGHT'),('F3','S2'),(dom_i['F3'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 
    53:[('NODES_SIDEWALL_VA_LEFT', 'NODES_SIDEWALL_VA_LEFT'),('F3','S2'),(dom_i['F3'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data],
    54:[('NODES_SIDEWALL_VA_RIGHT', 'NODES_SIDEWALL_VA_RIGHT'),('F3','S2'),(dom_i['F3'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 
    55:[('NODES_COWL_VA_LEFT', 'NODES_COWL_VA_LEFT'),('F3','S2'),(dom_i['F3'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 
    56:[('NODES_COWL_VA_RIGHT', 'NODES_COWL_VA_RIGHT'),('F3','S2'),(dom_i['F3'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 
    57:[('NODES_HATRACK_PAS_VA_LEFT', 'NODES_HATRACK_PAS_VA_LEFT'),('F3','S2'),(dom_i['F3'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data],
    58:[('NODES_HATRACK_PAS_VA_RIGHT', 'NODES_HATRACK_PAS_VA_RIGHT'),('F3','S2'),(dom_i['F3'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data],
    59:[('NODES_CEILING_VA', 'NODES_CEILING_VA'),('F3','S2'),(dom_i['F3'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data], 

    # Fluid-structure interfaces: from S2 to F2
    60:[('NODES_HATRACK_PAS_VA_LEFT', 'NODES_HATRACK_PAS_VA_LEFT'),('F2','S2'),(dom_i['F2'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data],
    61:[('NODES_HATRACK_PAS_VA_RIGHT', 'NODES_HATRACK_PAS_VA_RIGHT'),('F2','S2'),(dom_i['F2'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data],
    62:[('NODES_HATRACK_SEC_VA_LEFT', 'NODES_HATRACK_SEC_VA_LEFT'),('F2','S2'),(dom_i['F2'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data],
    63:[('NODES_HATRACK_SEC_VA_RIGHT', 'NODES_HATRACK_SEC_VA_RIGHT'),('F2','S2'),(dom_i['F2'], dom_i['S2']),([],[]),('NODE-NODE',0.01),True, load_data],
}


jobcase = 'coupling_example'
job_folder = f'{working_dir}/{subfolder}/jobcases/{jobcase}/'
model_name = f'{jobcase}.joblib'

# Get the component matrices and the coupling information
component_matrices, coupling_info = CMS.build_ds_models_vibroacoustic(subdomains, interface_pairs, save_info=(True, job_folder, model_name))



#%% Divisions of the Transfer Path Analysis: all vibro-acoustic components
level_components = {
                    'AP': ['S1','S2','F1','F2','F3'], # all domains
                    'P1': ['S2','F2','F3'], 
                    'P2': ['F3',],
                    }


level_interfaces = {'AP': [],
                    'P1': list(np.arange(1,11,1)) + list(np.arange(20,22,1)) + list(np.arange(40,50,1)), #structure-borne, airborne and vibro-acoustic borne
                    'P2': list(np.arange(20,22,1)) + list(np.arange(50,60,1)), # airborne and vibro-acoustic borne
                }


#%% Calculate the Transfer Path Analysis matrices
save_job_info = (True, job_folder, model_name)
red_method = 'DUAL'
n_modes = {'S1':20, 'S2':20,'F1':20,'F2':20,'F3':20} # in case CMS methods are selected
Mat_red, Mat_pres, Coup_info, time_RAM = CMS.TPA_matrices_vibroacoustic(component_matrices, coupling_info, level_components, 
                                                                        level_interfaces, CMS_method=[red_method,n_modes],  save_job_info = save_job_info)


#%% Prepare the load case
ref_comp_shapes_s = sum([component_matrices['shapes'][comp] for comp in coupling_info['components'] if (len(comp.split('_'))>1 and comp[0]=='S')])
ref_comp_shapes_f = sum([component_matrices['shapes'][comp] for comp in coupling_info['components'] if (len(comp.split('_'))>1 and comp[0]=='F')])

# Select the frequency range
freqs = np.arange(50,120,1)

# Select the load case: distributed pressure in the fuselage
load_case = {
            'S1': ['PRESSURE', 'FUSELAGE', 200],
            'S2': ['NONE'],
            'S-REF': ['NONE',ref_comp_shapes_s],
            'F1': ['NONE'],
            'F2': ['NONE'],
            'F3': ['NONE'],
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
load_results = True


if load_results==False:
    # Run the analysis
    X, Coup_info, TPA_CPU_RAM = CMS.TPA_CMS_vibroacoustic(component_matrices, coupling_info, level_components, level_interfaces, force_ext,  
                                               analysis_settings_f, type_TPA='1L-TPA', CMS_method=[red_method,n_modes], complex_analysis=True,
                                               save_job_info = save_job_info, load_matrices_info=load_matrices_info)
else:
    # Load the analysis results
    X, Coup_info = CMS.load_solution_data(job_folder, job_name)

#%% Analyse the contributions
descriptions = {str(intf_id):interface[0][0].replace('NODES_','') for intf_id, interface in interface_pairs.items()}
descriptions['0'] = 'FULL_ANALYSIS'
descriptions['ALL'] = 'ALL PATHS'

# Select the level
level = 'P1'

# Components
comp_sel_s = ['S2'] # Structural
comp_sel_f = ['F3'] # fluid
components = (comp_sel_s, comp_sel_f)


# Select the path name
all_path_names = {'AP':['0'], 'P1':[str(path) for path in level_interfaces['P1']] + ['ALL'], 
              'P2':[str(path) for path in level_interfaces['P2']] + ['ALL'],}

# Filter by keyword
keywords = ['FLOOR', 'DADO', 'SIDEWALL', 'COWL', 'CEILING', 'HATRACK', 'VENTILATION'] # filter by panel type
keywords = ['VA' , 'STR', 'ACO'] # filter by transmission type
if len(keywords) != 0: # separate by groups    
    for l in all_path_names.keys():
        keyword_dict = {}
        for key in keywords:
            paths = '+'.join([path for path in all_path_names[l] if key in descriptions.get(path)])
            if paths != '':
                keyword_dict[key] = paths
                descriptions[keyword_dict[key]] = key
            
        all_path_names[l] = [paths for paths in keyword_dict.values()] + ['ALL']
        
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
    cont, amp_phases, path_name_title, freq_range_names= CMS.plot_TPA_contribution_results_3(X, Coup_info, level, components, path_names, var, freq_set, section = [None,], 
                                                                                             nodes_sel=nodes_sel,ylabel=f'{var} {mag[var]}', 
                                                                                             cmap_name="jet", xlabel='Frequency [Hz]',
                                                                                             ylim=ylim, in_n_octave_bands=[in_oct_bands[j],6,[50,120]], 
                                                                                             descriptions=descriptions, plot_title=titles[j],
                                                                                             fontsize=24, figsize=(20,8), legend=True, plot_type=plot_types[j],
                                                                                             savefig=filename,
                                                                                             )


#%% Plot using pyvista
descriptions = {str(intf_id):interface[0][0].replace('NODES_','') for intf_id, interface in interface_pairs.items()}
descriptions['0'] = 'FULL_ANALYSIS'
descriptions['ALL'] = 'ALL PATHS'

# Select the level
level = 'P1'

# Components
comp_sel_s = ['S1','S2'] # Structural
comp_sel_f = ['F3'] # fluid
components = (comp_sel_s, comp_sel_f)


# Select the path name
all_path_names = {'AP':['0'], 'P1':[str(path) for path in level_interfaces['P1']] + ['ALL'], 
              'P2':[str(path) for path in level_interfaces['P2']] + ['ALL'],}

# Filter by keyword
keywords = ['FLOOR', 'DADO', 'SIDEWALL', 'COWL', 'CEILING', 'HATRACK', 'VENTILATION'] # filter by panel type
keywords = ['VA' , 'STR', 'ACO'] # filter by transmission type
if len(keywords) != 0: # separate by groups    
    for l in all_path_names.keys():
        keyword_dict = {}
        for key in keywords:
            paths = '+'.join([path for path in all_path_names[l] if key in descriptions.get(path)])
            if paths != '':
                keyword_dict[key] = paths
                descriptions[keyword_dict[key]] = key
            
        all_path_names[l] = [paths for paths in keyword_dict.values()] + ['ALL']
        
path_names = all_path_names[level]

# Define the variable that wants to be plotted
var = 'FORCE-UVECT'

# Define the frequency set
freq_highlight = 106 # peak of SPL
freq_set = [np.argmin(np.abs(X['AP']['0']['stamps'] - freq_highlight))]

# Define the section plane
section_plane = [None,[0.8,1.2]]


cont = CMS.plot_TPA_results_pyvista(X, Coup_info, level, components, path_names, var, freq_set, def_factor=1e3, show_real_imag_values='real', plot_size=(1700,1000),
                                descriptions=descriptions, share_clim=True, orientation=(-1,0.0,0.0), roll_angles=[90, -20, 20], show_min_max=True, plot_contributions=False, 
                                show_edges=False, make_zeros_transparent=False, vector_scale = True, section = section_plane, 
                                parallel_projection = True, result_on_node=False)    

