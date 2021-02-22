import numpy as np
#from covar import cov_shrink_ss
from scipy.linalg import sqrtm
import pickle
from collections import defaultdict
import bezier
#from surfer import Brain
from mne.viz import Brain
from mayavi import mlab
import pyvista as pv
from matplotlib.pyplot import cm
import mne
mne.viz.set_3d_backend("pyvista")

class TriuSparse():
    def __init__(self,mat,k=1,precision=np.float32):
        mat = mat.astype(precision)
        if mat.shape[-1] != mat.shape[-2]:
            raise ValueError("Last two dimensions must be the same.")
        self.mat_res = mat.shape[-1]
        self.mat_inds = np.triu_indices(self.mat_res,k=1,m=self.mat_res)
        self.mat_sparse = mat[...,self.mat_inds[0],self.mat_inds[1]]
        self.k = k
    def save(self,filename):
        out_dics = {"mat_res":self.mat_res,"mat_inds":self.mat_inds,
                    "mat_sparse":self.mat_sparse}
        with open(filename,"wb") as f:
            pickle.dump(out_dics,f)

def load_sparse(filename,convert=True,full=False,nump_type="float32"):
    with open(filename,"rb") as f:
        result = pickle.load(f)
    if convert:
        full_mat = np.zeros(result["mat_sparse"].shape[:-1] + \
          (result["mat_res"],result["mat_res"])).astype(nump_type)
        full_mat[...,result["mat_inds"][0],result["mat_inds"][1]] = \
          result["mat_sparse"]
        result = full_mat
    return result

# make a dPTE without directed information and normalise to 0-1
def dPTE_to_undirected(dPTE):
    tu_inds = np.triu_indices(dPTE.shape[1], k=1)
    undir = np.zeros(dPTE.shape,dtype="float32")
    undir_vec = np.abs(dPTE[:,tu_inds[0],tu_inds[1]]-.5)*-2 # 2 is negative here as convenience for laplacian later
    undir[:,tu_inds[0],tu_inds[1]] = undir_vec
    undir[:,tu_inds[1],tu_inds[0]] = undir_vec
    return undir

# laplacian on a (nondirected) dPTE, acts in place if undirect=False
def dPTE_to_laplace(x, undirect=True):
    if undirect:
        undir = dPTE_to_undirected(x)
    else:
        undir = x
    rowsums = np.nansum(undir,axis=1)*-1
    undir[:,np.arange(rowsums.shape[1]),np.arange(rowsums.shape[1])] = rowsums
    return undir

def phi(mat, k=0):
    if len(mat.shape)>2:
        triu_inds = np.triu_indices(mat.shape[1],k=k)
        return mat[...,triu_inds[0],triu_inds[1]]
    else:
        triu_inds = np.triu_indices(mat.shape[0],k=k)
        return mat[triu_inds[0],triu_inds[1]]

def covariance(mat_a, mat_b=None, reg=False):
    mean_a = np.mean(mat_a, axis=0)
    if reg:
        cov_a, _ = cov_shrink_ss(np.ascontiguousarray(phi(mat_a)).astype(np.float64))
    else:
        cov_a = (phi(mat_a)-phi(mean_a)).T.dot(phi(mat_a)-phi(mean_a))/(len(mat_a)-1)
    if mat_b is None:
        cov_out = cov_a
    else:
        mean_b = np.mean(mat_b, axis=0)
        if reg:
            cov_b, _ = cov_shrink_ss(np.ascontiguousarray(phi(mat_b)).astype(np.float64))
        else:
            cov_b = (phi(mat_b)-phi(mean_b)).T.dot(phi(mat_b)-phi(mean_b))/(len(mat_b)-1)
        cov_out = (cov_a*len(mat_a) + cov_b*len(mat_b)) / (len(mat_a) + len(mat_b) - 2)
    return cov_out

def samp2_chi(mat_a, mat_b):
    mean_a = np.mean(mat_a,axis=0)
    mean_b = np.mean(mat_b,axis=0)
    prec_mat = np.linalg.inv(covariance(mat_a, mat_b=mat_b, reg=True))
    sq_diff_var = np.linalg.multi_dot(((phi(mean_a)-phi(mean_b)).T,
                                      prec_mat, phi(mean_a)-phi(mean_b)))
    t2 = (len(mat_a)*len(mat_b))/(len(mat_a)+len(mat_b)) * sq_diff_var

    # now compute the contribution of individual edges
    mean = np.mean(np.vstack((mat_a,mat_b)),axis=0)
    prec_sqrt = sqrtm(prec_mat)
    lambda_a = np.linalg.multi_dot((prec_sqrt, phi(mean_a)-phi(mean)))
    lambda_b = np.linalg.multi_dot((prec_sqrt, phi(mean_b)-phi(mean)))
    kappa = (lambda_a**2 + lambda_b**2)/2
    return t2, kappa

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self,u,v):
        self.graph[u].append(v)

    def build_undirected(self,edges):
         nodes = list(set([e for ed in edges for e in ed]))
         for n in nodes:
             for e in edges:
                 if n in e:
                     if e[1] not in self.graph[e[0]]:
                        self.add_edge(e[0],e[1])
                     if e[0] not in self.graph[e[1]]:
                         self.add_edge(e[1],e[0])

    def BFS(self, s):
        visited = {n:False for n in self.graph.keys()}
        queue = []
        queue.append(s)
        visited[s] = True
        while queue:
            s = queue.pop(0)
            for i in self.graph[s]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True
        return [k for k,v in visited.items() if v] # list of nodes visited

    def find_components(self, edges):
        components = []
        for n in self.graph.keys():
            visited_nodes = set(self.BFS(n))
            if len(visited_nodes) == len(self.graph): # component encompasses all nodes, can stop here
                return [visited_nodes]
            contained = [set.intersection(visited_nodes,comp) for comp in components]
            if not any(contained):
                components.append(set(visited_nodes))
        return components

def get_active_edge_inds(components, edges):
    comp_edge_inds = []
    for comp in components:
        comp_edge_inds.append([])
        for edge_idx, edge in enumerate(edges):
            if any([edge[0] in comp, edge[1] in comp]):
                comp_edge_inds[-1].append(edge_idx)
    return comp_edge_inds

def cnx_cluster(f_vals, p_vals, cnx_n, p_thresh=0.05, edges=None):
    psig_inds = np.where(p_vals<p_thresh)[0] # which connections pass threshold
    if not any(psig_inds): # nothing passed threshold; return 0 and no edges
        return [0], []
    # convert to upper triangular square matrix indices
    if edges == None:
        all_edges = np.triu_indices(cnx_n,k=1)
    else:
        all_edges = edges
    edges = [(all_edges[0][idx],all_edges[1][idx]) for idx in psig_inds]
    # divide edges into graph components
    g = Graph()
    g.build_undirected(edges)
    components = g.find_components(edges)
    # sum of f vals for each component, and their component edges
    active_edge_inds = get_active_edge_inds(components, edges)
    comp_f = []
    out_edges = []
    for act_ed_inds in active_edge_inds:
        f_inds = psig_inds[act_ed_inds]
        comp_f.append(np.sum(np.abs(f_vals)[f_inds]))
        out_edges.append([edges[aei] for aei in act_ed_inds])
    return comp_f, out_edges

def plot_directed_cnx(mat,labels,parc,lup_title=None,ldown_title=None,rup_title=None,
                      rdown_title=None, figsize=2160, lineres=100,
                      subjects_dir="/home/jev/hdd/freesurfer/subjects",
                      alpha_max=None, alpha_min=None, uniform_weight=False,
                      surface="inflated", alpha=1, top_cnx=50, bot_cnx=None,
                      centre=0, min_alpha=0.1, cmap_name="RdBu"):

    cmap = cm.get_cmap(cmap_name)

    if mat.min() >= centre or mat.max() < centre:
        print("Warning: Values do not seem to match specified centre of {}.".format(centre))
    mat -= centre

    if top_cnx is not None:
        matflat = np.abs(mat.flatten())
        try:
            thresh = np.sort(matflat[matflat>0])[-top_cnx]
        except:
            thresh = matflat[matflat>0].min()
        mat[np.abs(mat)<thresh] = 0
    if bot_cnx is not None:
        matflat = np.abs(mat.flatten())
        try:
            thresh = np.sort(matflat[matflat>0])[-bot_cnx]
        except:
            thresh = matflat[matflat>0].max()
        mat[np.abs(mat)>thresh] = 0

    lingrad = np.linspace(0,1,lineres)

    brain = Brain('fsaverage', 'both', surface, alpha=alpha,
                  subjects_dir=subjects_dir, size=figsize, show=False)
    if lup_title:
        brain.add_text(0, 0.8, lup_title, "lup", font_size=40)
    if ldown_title:
        brain.add_text(0, 0, ldown_title, "ldown", font_size=40)
    if rup_title:
        brain.add_text(0.7, 0.8, rup_title, "rup", font_size=40)
    if rdown_title:
        brain.add_text(0.7, 0., rdown_title, "rdown", font_size=40)
    brain.add_annotation(parc,color="black")
    rrs = np.array([brain.geo[l.hemi].coords[l.center_of_mass()] for l in labels])

    if alpha_max is None:
        alpha_max = np.abs(mat).max()
    if alpha_min is None:
        alpha_min = np.abs(mat[mat!=0]).min()

    inds_pos = np.where(mat>0)
    origins = rrs[inds_pos[0],]
    dests = rrs[inds_pos[1],]
    inds_neg = np.where(mat<0)
    origins = np.vstack((origins,rrs[inds_neg[1],]))
    dests = np.vstack((dests,rrs[inds_neg[0],]))
    inds = (np.hstack((inds_pos[0],inds_neg[0])),np.hstack((inds_pos[1],inds_neg[1])))

    area_red = np.zeros(len(labels))
    area_blue = np.zeros(len(labels))
    np.add.at(area_red,inds_pos[0],1)
    np.add.at(area_blue,inds_pos[1],1)
    np.add.at(area_red,inds_neg[1],1)
    np.add.at(area_blue,inds_neg[0],1)
    area_weight = area_red + area_blue
    area_red = area_red/np.max((area_red.max(),area_blue.max()))
    area_blue = area_blue/np.max((area_red.max(),area_blue.max()))
    area_weight = area_weight/area_weight.max()

    lengths = np.linalg.norm(origins-dests, axis=1)
    lengths = np.broadcast_to(lengths,(3,len(lengths))).T
    midpoints = (origins+dests)/2
    midpoint_units = midpoints/np.linalg.norm(midpoints,axis=1,keepdims=True)
    spline_mids = midpoints + midpoint_units*lengths*2
    if uniform_weight:
        alphas = np.ones(len(inds[0]))*0.8
        area_weight[area_weight>0] = 0.8
        area_red[area_red>0] = 1
        area_blue[area_blue>0] = 1
    else:
        alphas = (1-min_alpha)*((np.abs(mat[inds[0],inds[1]])-alpha_min)/(alpha_max-alpha_min)) + min_alpha
        alphas[alphas<0],alphas[alphas>1] = 0, 1

    # mlab.points3d(origins[:,0],origins[:,1],origins[:,2],
    #               alphas,scale_factor=10,color=(1,0,0),transparent=True)
    # mlab.points3d(dests[:,0],dests[:,1],dests[:,2],
    #               alphas,scale_factor=10,color=(0,0,1),transparent=True)
    for l_idx, l in enumerate(labels):
        if area_weight[l_idx] == 0:
            continue
        brain.add_label(l,color=(area_red[l_idx],0,area_blue[l_idx]),
                        alpha=area_weight[l_idx])
        point = pv.Sphere(center=rrs[l_idx], radius=area_weight[l_idx]*5+2)
        brain._renderer.plotter.add_mesh(point,
                                         color=(area_red[l_idx],0,
                                         area_blue[l_idx]),
                                         opacity=area_weight[l_idx])
        # mlab.points3d(rrs[l_idx,0],rrs[l_idx,1],rrs[l_idx,2],
        #               area_weight[l_idx],scale_factor=10,
        #               color=(area_red[l_idx],0,area_blue[l_idx]),
        #               transparent=True)
    spl_pts = np.empty((len(origins),3,lineres))
    for idx in range(len(origins)):
        curve = bezier.Curve(np.array([[origins[idx,0],spline_mids[idx,0],dests[idx,0]],
                                      [origins[idx,1],spline_mids[idx,1],dests[idx,1]],
                                      [origins[idx,2],spline_mids[idx,2],dests[idx,2]]]),
                                      degree=2)
        spl_pts[idx,] = curve.evaluate_multi(lingrad)
        spline = pv.Spline(spl_pts[idx,].T, lineres)
        tube = spline.tube(radius=alphas[idx])
        tube["scalars"] = np.linspace(0,1,tube.n_points)
        brain._renderer.plotter.add_mesh(tube, cmap=cmap,
                                         opacity=alphas[idx])
        # mlab.plot3d(spl_pts[idx,0,],spl_pts[idx,1,],spl_pts[idx,2,],
        #             lingrad*255,tube_radius=alphas[idx]*2,colormap="RdBu",
        #             opacity=alphas[idx])

    brain._renderer.plotter.remove_scalar_bar()
    brain.show()
    return brain

def plot_undirected_cnx(mat, labels, parc, fig=None, lup_title=None,
                        ldown_title=None, rup_title=None, rdown_title=None,
                        figsize=(3840,2160), lineres=1000,
                        subjects_dir="/home/jev/hdd/freesurfer/subjects",
                        alpha_max=None, alpha_min=None, uniform_weight=False,
                        surface="inflated", alpha=1, top_cnx=50, bot_cnx=None,
                        color=(1,0,0)):
    if top_cnx is not None:
        matflat = mat.flatten()
        thresh = np.sort(matflat[matflat>0])[-top_cnx]
        mat[mat<thresh] = 0
    if bot_cnx is not None:
        matflat = np.abs(mat.flatten())
        thresh = np.sort(matflat[matflat>0])[-bot_cnx]
        mat[np.abs(mat)>thresh] = 0
    lingrad = np.linspace(0,1,lineres)
    if fig is None:
        fig = mlab.figure(size=figsize)
    brain = Brain('fsaverage', 'both', surface, alpha=alpha,
                  subjects_dir=subjects_dir, figure=fig)
    if lup_title:
        brain.add_text(0, 0.8, lup_title, "lup", font_size=40)
    if ldown_title:
        brain.add_text(0, 0, ldown_title, "ldown", font_size=40)
    if rup_title:
        brain.add_text(0.7, 0.8, rup_title, "rup", font_size=40)
    if rdown_title:
        brain.add_text(0.7, 0., rdown_title, "rdown", font_size=40)
    brain.add_annotation(parc,color="black")
    rrs = np.array([brain.geo[l.hemi].coords[l.center_of_mass()] for l in labels])

    if alpha_max is None:
        alpha_max = np.abs(mat).max()
    if alpha_min is None:
        alpha_min = np.abs(mat[mat!=0]).min()

    inds = np.where(mat>0)
    origins = rrs[inds[0],]
    dests = rrs[inds[1],]

    areas = np.zeros(len(labels))
    np.add.at(areas,inds[0],1)
    np.add.at(areas,inds[1],1)
    area_weights = areas/areas.max()

    lengths = np.linalg.norm(origins-dests, axis=1)
    lengths = np.broadcast_to(lengths,(3,len(lengths))).T
    midpoints = (origins+dests)/2
    midpoint_units = midpoints/np.linalg.norm(midpoints,axis=1,keepdims=True)
    spline_mids = midpoints + midpoint_units*lengths*2
    if uniform_weight:
        alphas = np.ones(len(inds[0]))*0.8
        area_weights[area_weights>0] = 0.8
    else:
        alphas = ((np.abs(mat[inds[0],inds[1]])-alpha_min)/(alpha_max-alpha_min))
        alphas[alphas<0],alphas[alphas>1] = 0, 1

    mlab.points3d(origins[:,0],origins[:,1],origins[:,2],
                  alphas,scale_factor=10,color=color,transparent=True)
    mlab.points3d(dests[:,0],dests[:,1],dests[:,2],
                  alphas,scale_factor=10,color=color,transparent=True)
    for l_idx, l in enumerate(labels):
        if area_weights[l_idx] == 0:
            continue
        brain.add_label(l,color=color, alpha=area_weights[l_idx])
    spl_pts = np.empty((len(origins),3,lineres))
    for idx in range(len(origins)):
        curve = bezier.Curve(np.array([[origins[idx,0],spline_mids[idx,0],dests[idx,0]],
                                      [origins[idx,1],spline_mids[idx,1],dests[idx,1]],
                                      [origins[idx,2],spline_mids[idx,2],dests[idx,2]]]),
                                      degree=2)
        spl_pts[idx,] = curve.evaluate_multi(lingrad)
        mlab.plot3d(spl_pts[idx,0,],spl_pts[idx,1,],spl_pts[idx,2,],
                    lingrad*255,tube_radius=alphas[idx]*2,color=color,
                    opacity=alphas[idx])

    return brain

def plot_rgba_cnx(mat_rgba, labels, parc, lup_title=None,
                  ldown_title=None, rup_title=None, rdown_title=None,
                  figsize=2160, lineres=100,
                  subjects_dir="/home/jev/hdd/freesurfer/subjects",
                  uniform_weight=False, surface="inflated", brain_alpha=1,
                  top_cnx=50, bot_cnx=None):

    if top_cnx is not None:
        matflat = np.abs(mat_rgba.flatten())
        try:
            thresh = np.sort(matflat[matflat>0])[-top_cnx]
        except:
            thresh = matflat[matflat>0].min()
        mat_rgba[np.abs(mat_rgba)<thresh] = 0
    if bot_cnx is not None:
        matflat = np.abs(mat.flatten())
        try:
            thresh = np.sort(matflat[matflat>0])[-bot_cnx]
        except:
            thresh = matflat[matflat>0].max()
        mat_rgba[np.abs(mat_rgba)>thresh] = 0

    alpha = mat_rgba[...,-1].copy()
    alpha_inds = alpha>0
    alpha[alpha_inds] = (alpha[alpha_inds] - alpha[alpha_inds].min()) / \
      (alpha.max() - alpha[alpha_inds].min())
    # put a floor on smallest value
    alpha[alpha_inds] = alpha[alpha_inds] * .9 + 0.1
    mat_rgba[...,-1] = alpha

    lingrad = np.linspace(0,1,lineres)
    brain = Brain('fsaverage', 'both', surface, alpha=brain_alpha,
                  subjects_dir=subjects_dir, size=figsize, show=False)
    if lup_title:
        brain.add_text(0, 0.8, lup_title, "lup", font_size=40)
    if ldown_title:
        brain.add_text(0, 0, ldown_title, "ldown", font_size=40)
    if rup_title:
        brain.add_text(0.7, 0.8, rup_title, "rup", font_size=40)
    if rdown_title:
        brain.add_text(0.7, 0., rdown_title, "rdown", font_size=40)
    brain.add_annotation(parc,color="black")
    rrs = np.array([brain.geo[l.hemi].coords[l.center_of_mass()] for l in labels])

    inds = np.where(mat_rgba[...,-1]>0)
    origins = rrs[inds[0],]
    dests = rrs[inds[1],]

    lengths = np.linalg.norm(origins-dests, axis=1)
    lengths = np.broadcast_to(lengths,(3,len(lengths))).T
    midpoints = (origins+dests)/2
    midpoint_units = midpoints/np.linalg.norm(midpoints,axis=1,keepdims=True)
    spline_mids = midpoints + midpoint_units*lengths*2

    spl_pts = np.empty((len(origins),3,lineres))
    for idx in range(len(origins)):
        color = (mat_rgba[inds[0][idx], inds[1][idx], 0],
                 mat_rgba[inds[0][idx], inds[1][idx], 1],
                 mat_rgba[inds[0][idx], inds[1][idx], 2])
        alpha = mat_rgba[inds[0][idx], inds[1][idx], 3]
        curve = bezier.Curve(np.array([[origins[idx,0],spline_mids[idx,0],dests[idx,0]],
                                       [origins[idx,1],spline_mids[idx,1],dests[idx,1]],
                                       [origins[idx,2],spline_mids[idx,2],dests[idx,2]]]),
                                       degree=2)
        spl_pts[idx,] = curve.evaluate_multi(lingrad)

        spline = pv.Spline(spl_pts[idx,].T, lineres)
        tube = spline.tube(radius=alpha*2)
        brain._renderer.plotter.add_mesh(tube, color=color, opacity=alpha)
        # mlab.plot3d(spl_pts[idx,0,], spl_pts[idx,1,], spl_pts[idx,2,],
        #             lingrad*255, tube_radius=alpha*2, color=color,
        #             opacity=alpha)

    ## origin/destination points and label colouring
    # first get row/column alpha maxima, calculate range for normalisation
    alpha_max = np.stack((mat_rgba[...,3].max(axis=0),
                           mat_rgba[...,3].max(axis=1))).max(axis=0)
    alpha_max_max, alpha_max_min = (alpha_max.max(),
                                    alpha_max[alpha_max!=0].min())
    # normalise
    alpha_inds = alpha_max!=0
    alpha_max[alpha_inds] = ((alpha_max[alpha_inds] -
                              alpha_max[alpha_inds].min()) /
                             (alpha_max.max() - alpha_max[alpha_inds].min()))
    # put a floor on smallest value
    alpha_max[alpha_inds] = alpha_max[alpha_inds] * .7 + 0.3
    # go through each region and colour it
    for idx in range(len(labels)):
        if not (np.any(mat_rgba[idx,]) or np.any(mat_rgba[:,idx,])):
            continue
        color_vec = np.dot(mat_rgba[idx,:,:3].T, mat_rgba[idx,:,3]) + \
          np.dot(mat_rgba[:,idx,:3].T, mat_rgba[:,idx,3])
        color_vec = color_vec / np.linalg.norm(color_vec)
        alpha = alpha_max[idx]
        brain.add_label(labels[idx], color=color_vec, alpha=alpha)
        point = pv.Sphere(center=rrs[idx], radius=alpha*5+0.2)
        brain._renderer.plotter.add_mesh(point, color=color_vec, opacity=1)

        # mlab.points3d(origins[idx,0],origins[idx,1],origins[idx,2],
        #               alpha,scale_factor=10,color=color,transparent=True)
        # mlab.points3d(dests[idx,0],dests[idx,1],dests[idx,2],
        #               alpha,scale_factor=10,color=color,transparent=True)

    brain.show()
    return brain

def plot_rgba(vec_rgba, labels, parc, hemi="both", lup_title=None,
              ldown_title=None, rup_title=None, rdown_title=None,
              figsize=2160, subjects_dir="/home/jev/hdd/freesurfer/subjects",
              uniform_weight=False, surface="inflated", brain_alpha=1.):

    brain = Brain('fsaverage', hemi, surface, alpha=brain_alpha,
                   subjects_dir=subjects_dir, size=figsize, show=False)
    if lup_title:
        brain.add_text(0, 0.8, lup_title, "lup", font_size=40)
    if ldown_title:
        brain.add_text(0, 0, ldown_title, "ldown", font_size=40)
    if rup_title:
        brain.add_text(0.7, 0.8, rup_title, "rup", font_size=40)
    if rdown_title:
        brain.add_text(0.7, 0., rdown_title, "rdown", font_size=40)

    brain.add_annotation(parc,color="black", alpha=1.)

    for l_idx, l in enumerate(labels):
        if np.array_equal(vec_rgba[l_idx], [0,0,0,0]):
            continue
        brain.add_label(l, color=vec_rgba[l_idx,:3], alpha=vec_rgba[l_idx,3])

    brain.show()
    return brain

""" calculate the corelation distance of dPTE connectivity matrices """
def pw_cor_dist(mat,inds):
    k = len(inds)
    tu_inds = np.triu_indices(mat.shape[1], k=1)
    vecs = mat[:, tu_inds[0], tu_inds[1]]
    vecs -= 0.5 # dPTE is 0.5 centred; because we do this now, we can leave out the subtraction terms in distance calculation
    dists = np.zeros(k,dtype="float32")
    for pair_idx, pair in enumerate(inds):
        dist = (vecs[pair[0]].dot(vecs[pair[1]]) /
                np.sqrt(np.sum(vecs[pair[0]]**2) * np.sum(vecs[pair[1]]**2)))
        dist = (dist * -1 + 1)/2 # transform to distance measure
        dists[pair_idx] = dist
    return dists
