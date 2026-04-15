"""
Script to suppress structural and acoustic interface DOF from the
vibro-acoustic sections. This is necessary for the correct definition
of Dynamic Substructuring.


"""

named_sel = "nodes_suppress_va"
#named_sel = "nodes_suppress_edges"

named_selections = ['membrane_va', 'frame_va']
#named_selections = ['up','down','right','left']



ns_supp = DataModel.GetObjectsByName(named_sel)[0]
mesh    = DataModel.MeshDataByName(ExtAPI.DataModel.MeshDataNames[0])



for interface in named_selections:
    ns_if   = DataModel.GetObjectsByName(interface)[0]

    node_ids = set()
    for gid in ns_if.Location.Ids:
        node_ids.update(mesh.MeshRegionById(gid).NodeIds)
    
    node_ids -= set(ns_supp.Location.Ids)
    
    sel = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.MeshNodes)
    sel.Ids = list(node_ids)
    
    ns_new = Model.AddNamedSelection()
    ns_new.Name = "nodes_" + ns_if.Name
    ns_new.Location = sel
