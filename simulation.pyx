import numpy as np
cimport numpy as np
from libc.math cimport sqrt
import cython
import random
import pickle

cdef class Simulation:
  """A class for simulating particle behavior in a 2D space.

    Parameters:
    ----------
    n_p : int
        The number of particles in the simulation.

    dt : float
        Time step for integration.

    trun : int
        Total number of time steps for the simulation.

    v : float
        Particle velocity.

    om_mu : float
        Mean value for particle orientation.

    om_sig : float
        Standard deviation for particle orientation.

    Dr : float
        Rotational diffusion constant.

    rcut : float
        Cutoff distance for Weeks-Chandler-Anderson (WCA) potential.

    rho : float
        Particle density.

    box_ratio : float
        Aspect ratio of the simulation box.

    dt_save : float
        Time interval for saving simulation data.

    eps : float
        Strength of the WCA potential.

    if_align : int
        Flag for particle alignment (1 for enabled, 0 for disabled).

    if_wca : int
        Flag for WCA potential (1 for enabled, 0 for disabled).

    kap : float
        Alignment strength.

    rK : float
        Range for alignment interactions.

    dt_neighborupdate : float
        Time interval for neighbor list updates.

    init_form : int
        Initialization method (0 for lattice, 1 for random, 2 for file, 3 for partially sorted, 4 for partially sorted with 2 waves, 5 for partially sorted with 3 waves, 6 for partially sorted with mirrored, 7 for partially sorted downwards).

    if_omconst : bool
        Flag for using a constant orientation (True for constant, False for variable).

    if_vnoise : int
        Flag for adding noise to particle velocity (1 for enabled, 0 for disabled).

    om_dist : str
        Distribution type for particle orientation ('uniform', 'normal', or 'doublenormal').

    num : int
        Simulation number.

    Attributes:
    ----------
    r : np.ndarray
        Array of particle positions, shape (n, 2).

    phi : np.ndarray
        Array of particle orientations, shape (n).

    forces : np.ndarray
        Array of forces acting on particles, shape (n, 2).

    rdiff : np.ndarray
        Array of random diffusion values, shape (n).

    rdiff_t : np.ndarray
        Array of translational random diffusion values, shape (n).

    om : np.ndarray
        Array of particle orientations, shape (n).

    distances : np.ndarray
        Array of distances between particles, shape (n, n).

    distance_vectors : np.ndarray
        Array of distance vectors between particles, shape (n, n, 2).

    alignments : np.ndarray
        Array of alignment values for each particle, shape (n).

    om_n : np.ndarray
        Array of particle orientations, shape (n).

    cell_list : np.ndarray
        3D array representing the cell list for neighbor search.

    n_in_cell : np.ndarray
        Array indicating the number of particles in each cell.

    neighbor_list : np.ndarray
        2D array representing the neighbor list for each particle.

    n_neighbors : np.ndarray
        Array indicating the number of neighbors for each particle.

    count_new : int
        Count of new interactions.

    count_old : int
        Count of old interactions.

    init_form : int
        Initialization method (0 for lattice, 1 for random, 2 for file, 3 for partially sorted, 4 for partially sorted with 2 waves, 5 for partially sorted with 3 waves, 6 for partially sorted with mirrored, 7 for partially sorted downwards).

    if_omconst : bool
        Flag for using a constant orientation (True for constant, False for variable).

    if_vnoise : int
        Flag for adding noise to particle velocity (1 for enabled, 0 for disabled).

    om_dist : str
        Distribution type for particle orientation ('uniform', 'normal', or 'doublenormal').

    box_hl_a : float
        Half-length of the simulation box along the 'a' direction.

    box_hl_b : float
        Half-length of the simulation box along the 'b' direction.

    dt_save : float
        Time interval for saving simulation data.

    save_steps : int
        Number of time steps between data saving.

    dt_neighborupdate : float
        Time interval for neighbor list updates.

    neighbor_steps : int
        Number of time steps between neighbor list updates.

    num : int
        Simulation number.

    rcut : float
        Cutoff distance for Weeks-Chandler-Anderson (WCA) potential.

    eps : float
        Strength of the WCA potential.

    kap : float
        Alignment strength.

    rK : float
        Range for alignment interactions.

    cell_length_min : float
        Minimum cell length based on rK and velocity.

    nx_cells : int
        Number of cells in the 'a' direction.

    ny_cells : int
        Number of cells in the 'b' direction.

    cell_list : np.ndarray
        3D array representing the cell list for neighbor search.

    n_in_cell : np.ndarray
        Array indicating the number of particles in each cell.

    neighbor_list : np.ndarray
        2D array representing the neighbor list for each particle.

    n_neighbors : np.ndarray
        Array indicating the number of neighbors for each particle.

    Methods:
    --------
    initialize():
        Initialize particle positions, orientations, and om_n based on the chosen initialization method.

    initialize_lattice():
        Initialize particles in a lattice formation.

    initialize_random():
        Initialize particles randomly while avoiding overlap.

    initialize_from_file():
        Initialize particles based on data from a file.

    initialize_phi_partially_sorted():
        Initialize particles with partially sorted orientations.

    initialize_phi_partially_sorted_2waves():
        Initialize particles with partially sorted orientations using 2 waves.

    initialize_phi_partially_sorted_3waves():
        Initialize particles with partially sorted orientations using 3 waves.

    initialize_phi_partially_sorted_mirror():
        Initialize particles with partially sorted orientations mirrored.

    initialize_phi_partially_sorted_downwards():
        Initialize particles with partially sorted orientations downwards.

    distance_fct():
        Calculate distances and distance vectors between particles.

    WCA():
        Calculate forces based on the Weeks-Chandler-Anderson (WCA) potential.

    alignment():
        Update particle alignments.

    cell_list_update():
        Update the cell list for neighbor search.

    neighbor_list_update():
        Update the neighbor list for each particle.

    integrate():
        Integrate the equations of motion for particles.

    run_save():
        Run the simulation and save data at specified intervals.

    Examples:
    --------
    # Create a Simulation instance with desired parameters
    sim = Simulation(n_p=100, dt=0.01, trun=1000, v=1.0, om_mu=0.0, om_sig=0.1, Dr=0.1, rcut=1.0,
                    rho=0.1, box_ratio=1.0, dt_save=1.0, eps=0.1, if_align=1, if_wca=1, kap=0.1, rK=2.0,
                    dt_neighborupdate=1.0, init_form=1, if_omconst=False, if_vnoise=1, om_dist='normal', num=1)

    # Initialize the simulation
    sim.initialize()

    # Run the simulation and save data
    sim.run_save()
    ```

This code defines a Python class called "Simulation" that simulates the behavior of particles in a 2D space with various parameters and options for initialization, integration, and data saving. It also includes methods for initializing particle positions and orientations, calculating forces, updating neighbor lists, and integrating the equations of motion. The class provides the flexibility to customize different aspects of the simulation and can be used to study the behavior of particles in various scenarios."""

  cdef long trun
  cdef int n, save_steps, nx_cells, ny_cells, count_new, count_old, neighbor_steps, init_form, rand_om_steps, num
  cdef double v, dt, dt_save, Dr, rcut, rho, box_ratio, box_hl_a,box_hl_b, eps, om_mu, om_sig, kap, rK, cell_length_min, dt_neighborupdate, om0
  cdef public np.ndarray r, phi, forces, rdiff, rdiff_t, om, distances, distance_vectors, alignments, om_n, cell_list, n_in_cell, neighbor_list, n_neighbors
  cdef bint rnoise, if_align, if_wca, if_omconst, if_vnoise, if_wca_only_at_start
  cdef str om_dist

  def __init__(self,n_p,dt,trun,v,om_mu,om_sig,Dr,rcut,rho,box_ratio,dt_save,eps,if_align,if_wca,kap,rK,dt_neighborupdate,init_form,if_omconst,if_vnoise,om_dist,if_wca_only_at_start,num):
    self.box_ratio=box_ratio
    self.n=n_p
#    self.n=int(self.box_ratio*n)
    self.trun=trun
    self.v=v
    self.om_mu=om_mu
    self.om_sig=om_sig
    self.dt=dt
    self.Dr=Dr
    self.r = np.zeros((self.n, 2), dtype=np.float64)
    self.phi = np.zeros(self.n, dtype=np.float64)
    self.om = np.zeros(self.n, dtype=np.float64)
    self.alignments = np.zeros(self.n, dtype=np.float64)
    self.kap = kap
    self.rK = rK
    self.forces = np.zeros((self.n, 2), dtype=np.float64)
    self.distances = np.zeros((self.n, self.n), dtype=np.float64)
    self.distance_vectors = np.zeros((self.n, self.n, 2), dtype=np.float64)
    self.om_n = np.zeros(self.n, dtype=np.float64)
    self.rdiff=np.zeros(self.n, dtype=np.float64)
    self.rdiff_t=np.zeros(self.n, dtype=np.float64)
    self.rho = rho
    #self.box_ratio = box_ratio
    self.box_hl_a=sqrt(self.box_ratio*self.n/self.rho)/2
    self.box_hl_b=sqrt(1/self.box_ratio*self.n/self.rho)/2
    self.dt_save=dt_save
    self.save_steps = int(round(self.dt_save/self.dt))
    self.dt_neighborupdate=dt_neighborupdate
    self.neighbor_steps = int(self.dt_neighborupdate/self.dt)
    self.num = num
    self.rcut=rcut
    self.eps=eps
    self.if_align=if_align
    self.if_wca=if_wca
    self.if_wca_only_at_start=if_wca_only_at_start
    self.cell_length_min = self.rK + self.v*self.neighbor_steps*self.dt
    self.nx_cells = int(2*self.box_hl_a / self.cell_length_min)
    self.ny_cells = int(2*self.box_hl_b / self.cell_length_min)
    self.cell_list = np.zeros((self.nx_cells, self.ny_cells, self.n), dtype=np.int32)
    self.n_in_cell = np.zeros((self.nx_cells, self.ny_cells), dtype=np.int32)
    self.neighbor_list = np.zeros((self.n, self.n), dtype=np.int32)
    self.n_neighbors = np.zeros(self.n, dtype=np.int32)
    self.count_new = 0
    self.count_old = 0
    self.init_form = init_form
    self.if_omconst=if_omconst
    self.if_vnoise=if_vnoise
    self.om_dist=om_dist

# Initialisierung---------------------------------------------------------------
  cpdef void initialize(self):
    cdef int i
    if self.init_form == 0:
      self.initialize_lattice()
    if self.init_form == 1:
      self.initialize_random()
    elif self.init_form == 2:
      self.initialize_from_file()
    elif self.init_form == 3:
      self.initialize_phi_partially_sorted()
    elif self.init_form == 4:
      self.initialize_phi_partially_sorted_2waves()
    elif self.init_form == 5:
      self.initialize_phi_partially_sorted_3waves()
    elif self.init_form == 6:
      self.initialize_phi_partially_sorted_mirror()
    elif self.init_form == 7:
      self.initialize_phi_partially_sorted_downwards()

    if self.if_omconst == True:
      for i in range(self.n):
        self.om_n[i] = self.om_mu
    else:
      if self.om_dist == 'uniform':
        mu_sig_equiv = sqrt(self.om_mu**2+self.om_sig**2)*sqrt(3)
        self.om_n = np.random.uniform(-mu_sig_equiv, mu_sig_equiv, self.n)
      if self.om_dist == 'normal':
        mu_sig_equiv = sqrt(self.om_mu**2+self.om_sig**2)
        self.om_n = np.random.normal(0, mu_sig_equiv, self.n)
      if self.om_dist == 'doublenormal':
        self.om_n = np.random.normal(-self.om_mu, self.om_sig, self.n)
        for i in range(self.n):
          self.om_n[i] *= random.randrange(-1,2,2)

  #lattice
  cpdef void initialize_lattice(self): #init_form 0
    cdef int _i=0, _ix=0, _iy=0
    cdef int _n_per_side_a = np.ceil(sqrt(self.box_ratio*self.n))
    cdef int _n_per_side_b = np.ceil(sqrt(1/self.box_ratio*self.n))
    cdef double _dist_a = 2*self.box_hl_a / _n_per_side_a
    cdef double _dist_b = 2*self.box_hl_b / _n_per_side_b
    cdef np.ndarray _xy = np.zeros(2)
    while _i < self.n :
      if _ix < _n_per_side_a:
        if _iy < _n_per_side_b:
          _xy[0] = _dist_a * _ix - self.box_hl_a
          _xy[1] = _dist_b * _iy - self.box_hl_b
          self.r[_i] = _xy
          _i+=1
          _iy +=1
        else:
          _iy=0
          _ix+=1
      else:
        print('lattice error, _ix>_n_per_side')
    self.phi = np.random.uniform(0, 2*np.pi, self.n)


  #random
  cpdef void initialize_random(self):#init_form 1
    cdef int _i=0, _j=0
    cdef double _x, _y, _dist_x, _dist_y, _r_abs
    cdef np.ndarray _distances = 2*self.rcut*np.ones(self.n)
    _i = 0
    while _i < self.n :
      _x = np.random.uniform(-self.box_hl_a+self.rcut*0.5, self.box_hl_a-self.rcut*0.5)
      _y = np.random.uniform(-self.box_hl_b+self.rcut*0.5, self.box_hl_b-self.rcut*0.5)
      _r_abs = sqrt(_x*_x + _y*_y)
      for _j in range(_i):
        _dist_x = self.r[_j,0] - _x
        _dist_y = self.r[_j,1] - _y
        _distances[_j] = sqrt(_dist_x*_dist_x + _dist_y*_dist_y)
      if min(_distances) > self.rcut :
        self.r[_i,0] = _x
        self.r[_i,1] = _y
        _i += 1
      else:
        print('init: ', _i ,'/',self.n, 'within other particle')
    self.phi = np.random.uniform(0, 2*np.pi, self.n)

  #init from file
  cpdef void initialize_from_file(self):#init_form 2
    cdef np.ndarray _r, _phi
    cdef char* _filename='init_files/init_waves_pb_rho03.pickle'
    with open(_filename, 'rb') as handle:
        _data = pickle.load(handle)
    self.r=_data["r"]
    self.phi=_data["phi"]

        #random, phi partially sorted
  cpdef void initialize_phi_partially_sorted(self): #init_form 3
    cdef double sort_ratio = 0.5
    cdef int _i=0, _j=0
    cdef double _x, _y, _dist_x, _dist_y, _r_abs
    cdef np.ndarray _distances = 2*self.rcut*np.ones(self.n)
    _i = 0
    self.phi = np.random.uniform(0, 2*np.pi, self.n)
    while _i < self.n :
      _x = np.random.uniform(-self.box_hl_a+self.rcut*0.5, self.box_hl_a-self.rcut*0.5)
      _y = np.random.uniform(-self.box_hl_b+self.rcut*0.5, self.box_hl_b-self.rcut*0.5)
      _r_abs = sqrt(_x*_x + _y*_y)
      for _j in range(_i):
        _dist_x = self.r[_j,0] - _x
        _dist_y = self.r[_j,1] - _y
        _distances[_j] = sqrt(_dist_x*_dist_x + _dist_y*_dist_y)
      if min(_distances) > self.rcut :
        self.r[_i,0] = _x
        self.r[_i,1] = _y
        if _i < (sort_ratio*self.n) :
          self.phi[_i] = -_x/self.box_hl_a *np.pi
        _i += 1
      else:
        print('init: ', _i ,'/',self.n, 'within other particle')


        #random, phi partially sorted, 2 waves
  cpdef void initialize_phi_partially_sorted_2waves(self): #init_form 4
    cdef double sort_ratio = 0.5
    cdef int _i=0, _j=0
    cdef double _x, _y, _dist_x, _dist_y, _r_abs
    cdef np.ndarray _distances = 2*self.rcut*np.ones(self.n)
    _i = 0
    self.phi = np.random.uniform(0, 2*np.pi, self.n)
    while _i < self.n :
      _x = np.random.uniform(-self.box_hl_a+self.rcut*0.5, self.box_hl_a-self.rcut*0.5)
      _y = np.random.uniform(-self.box_hl_b+self.rcut*0.5, self.box_hl_b-self.rcut*0.5)
      _r_abs = sqrt(_x*_x + _y*_y)
      for _j in range(_i):
        _dist_x = self.r[_j,0] - _x
        _dist_y = self.r[_j,1] - _y
        _distances[_j] = sqrt(_dist_x*_dist_x + _dist_y*_dist_y)
      if min(_distances) > self.rcut :
        self.r[_i,0] = _x
        self.r[_i,1] = _y
        if _i < (sort_ratio*self.n) :
          self.phi[_i] = -_x/self.box_hl_a *np.pi *2
        _i += 1
      else:
        print('init: ', _i ,'/',self.n, 'within other particle')

        #random, phi partially sorted, 2 waves
  cpdef void initialize_phi_partially_sorted_3waves(self): #init_form 5
    cdef double sort_ratio = 0.5
    cdef int _i=0, _j=0
    cdef double _x, _y, _dist_x, _dist_y, _r_abs
    cdef np.ndarray _distances = 2*self.rcut*np.ones(self.n)
    _i = 0
    self.phi = np.random.uniform(0, 2*np.pi, self.n)
    while _i < self.n :
      _x = np.random.uniform(-self.box_hl_a+self.rcut*0.5, self.box_hl_a-self.rcut*0.5)
      _y = np.random.uniform(-self.box_hl_b+self.rcut*0.5, self.box_hl_b-self.rcut*0.5)
      _r_abs = sqrt(_x*_x + _y*_y)
      for _j in range(_i):
        _dist_x = self.r[_j,0] - _x
        _dist_y = self.r[_j,1] - _y
        _distances[_j] = sqrt(_dist_x*_dist_x + _dist_y*_dist_y)
      if min(_distances) > self.rcut :
        self.r[_i,0] = _x
        self.r[_i,1] = _y
        if _i < (sort_ratio*self.n) :
          self.phi[_i] = -_x/self.box_hl_a *np.pi *3
        _i += 1
      else:
        print('init: ', _i ,'/',self.n, 'within other particle')

        #random, phi partially sorted, other mirrored
  cpdef void initialize_phi_partially_sorted_mirror(self): #init_form 6
    cdef double sort_ratio = 0.5
    cdef int _i=0, _j=0
    cdef double _x, _y, _dist_x, _dist_y, _r_abs
    cdef np.ndarray _distances = 2*self.rcut*np.ones(self.n)
    _i = 0
    self.phi = np.random.uniform(0, 2*np.pi, self.n)
    while _i < self.n :
      _x = np.random.uniform(-self.box_hl_a+self.rcut*0.5, self.box_hl_a-self.rcut*0.5)
      _y = np.random.uniform(-self.box_hl_b+self.rcut*0.5, self.box_hl_b-self.rcut*0.5)
      _r_abs = sqrt(_x*_x + _y*_y)
      for _j in range(_i):
        _dist_x = self.r[_j,0] - _x
        _dist_y = self.r[_j,1] - _y
        _distances[_j] = sqrt(_dist_x*_dist_x + _dist_y*_dist_y)
      if min(_distances) > self.rcut :
        self.r[_i,0] = _x
        self.r[_i,1] = _y
        if _i < (sort_ratio*self.n) :
          self.phi[_i] = _x/self.box_hl_a *np.pi
        _i += 1
      else:
        print('init: ', _i ,'/',self.n, 'within other particle')

        #random, phi partially sorted, band downwards
  cpdef void initialize_phi_partially_sorted_downwards(self): #init_form 7
    cdef double sort_ratio = 0.5
    cdef int _i=0, _j=0
    cdef double _x, _y, _dist_x, _dist_y, _r_abs
    cdef np.ndarray _distances = 2*self.rcut*np.ones(self.n)
    _i = 0
    self.phi = np.random.uniform(0, 2*np.pi, self.n)
    while _i < self.n :
      _x = np.random.uniform(-self.box_hl_a+self.rcut*0.5, self.box_hl_a-self.rcut*0.5)
      _y = np.random.uniform(-self.box_hl_b+self.rcut*0.5, self.box_hl_b-self.rcut*0.5)
      _r_abs = sqrt(_x*_x + _y*_y)
      for _j in range(_i):
        _dist_x = self.r[_j,0] - _x
        _dist_y = self.r[_j,1] - _y
        _distances[_j] = sqrt(_dist_x*_dist_x + _dist_y*_dist_y)
      if min(_distances) > self.rcut :
        self.r[_i,0] = _x
        self.r[_i,1] = _y
        if _i < (sort_ratio*self.n) :
          self.phi[_i] = -_y/self.box_hl_b *np.pi
        _i += 1
      else:
        print('init: ', _i ,'/',self.n, 'within other particle')

  cpdef void distance_fct(self):
    cdef double _dist_x, _dist_y
    cdef double _distance, _dist6i
    cdef double [:,:] r=self.r
    cdef int i, j, i_neighbor
    for i in range(self.n):
      for i_neighbor in range(self.n_neighbors[i]):
        j = self.neighbor_list[i,i_neighbor]
        self.count_new +=1
        _dist_x = r[j,0] - r[i,0]
        _dist_y = r[j,1] - r[i,1]
        if _dist_x > self.box_hl_a:
          _dist_x -= 2*self.box_hl_a
        elif _dist_x < -self.box_hl_a:
          _dist_x += 2*self.box_hl_a
        if _dist_y > self.box_hl_b:
          _dist_y -= 2*self.box_hl_b
        elif _dist_y < -self.box_hl_b:
          _dist_y += 2*self.box_hl_b
        _distance = sqrt(_dist_x*_dist_x + _dist_y*_dist_y)
        self.distances[i,j] = _distance
        self.distance_vectors[i,j] = [_dist_x,_dist_y]

# Weeks-Chandler-Anderson Potential---------------------------------------------
  cdef void WCA(self):
    self.forces = np.zeros((self.n, 2), dtype=np.float64)
    cdef double [:,:] forces=self.forces
    cdef double _ff, _dist_x, _dist_y
    cdef double _distance, _dist6i, _richtff_x, _richtff_y
    cdef int i, j, i_neighbor
    for i in range(self.n):
      for i_neighbor in range(self.n_neighbors[i]):
        j = self.neighbor_list[i,i_neighbor]
        _distance = self.distances[i,j]
        _dist_x, _dist_y = self.distance_vectors[i,j]
        if _distance < self.rcut:
          _dist6i = 1/(_distance*_distance*_distance*_distance*_distance*_distance)
          _ff = 24 * self.eps * (- 2 *_dist6i*_dist6i/_distance + _dist6i/_distance)
          _richtff_x = _dist_x/_distance
          _richtff_y = _dist_y/_distance
          forces[i,0] += _ff * _richtff_x
          forces[j,0] += -_ff * _richtff_x
          forces[i,1] += _ff * _richtff_y
          forces[j,1] += -_ff * _richtff_y
          if _distance < 1.0:
            print(_distance)

# Alignment
  cpdef void alignment(self):
    cdef double _align
    cdef int i, j, i_neighbor
    self.alignments = np.zeros(self.n, dtype=np.float64)
    for i in range(self.n):
      for i_neighbor in range(self.n_neighbors[i]):
        j = self.neighbor_list[i,i_neighbor]
        if self.distances[i,j] < self.rK:
          _align = self.kap / (np.pi * self.rK * self.rK) * np.sin(self.phi[j] - self.phi[i])
          self.alignments[i] += _align
          self.alignments[j] -= _align

# Cell List--------------------------------------------------------------------
  cpdef void cell_list_update(self):
    cdef int _ix, _iy
    cdef double _cell_length_a = 2*self.box_hl_a / self.nx_cells
    cdef double _cell_length_b = 2*self.box_hl_b / self.ny_cells
    self.cell_list = np.zeros((self.nx_cells, self.ny_cells, self.n), dtype=np.int32)
    self.n_in_cell = np.zeros((self.nx_cells, self.ny_cells), dtype=np.int32)
    for i in range(self.n):
      _ix = int((self.r[i,0] + self.box_hl_a) / _cell_length_a)
      _iy = int((self.r[i,1] + self.box_hl_b) / _cell_length_b)
      self.cell_list[_ix,_iy][ self.n_in_cell[_ix,_iy] ] = i
      self.n_in_cell[_ix,_iy] += 1


# Neighbor List--------------------------------------------------------------------
  cpdef void neighbor_list_update(self):
    cdef int _ix, _iy, i, _ip, j
    self.neighbor_list = np.zeros((self.n, self.n), dtype=np.int32)
    self.n_neighbors = np.zeros(self.n, dtype=np.int32)
    for _ix in range(self.nx_cells):
      for _iy in range(self.ny_cells):
        for i in range(self.n_in_cell[_ix,_iy]): #number of p in cell
          _ip = self.cell_list[_ix,_iy,i] #index of originparticle
          for j in range(i+1,self.n_in_cell[_ix,_iy]): #range(i+1,..) to not double entries
            self.neighbor_list[_ip, self.n_neighbors[_ip] ] = self.cell_list[_ix][_iy][j]
            self.n_neighbors[_ip] += 1
          for j in range(self.n_in_cell[_ix-1,_iy]):
            self.neighbor_list[_ip, self.n_neighbors[_ip] ] = self.cell_list[_ix-1][_iy][j]
            self.n_neighbors[_ip] += 1
          for j in range(self.n_in_cell[_ix-1,_iy-1]):
            self.neighbor_list[_ip, self.n_neighbors[_ip] ] = self.cell_list[_ix-1][_iy-1][j]
            self.n_neighbors[_ip] += 1
          for j in range(self.n_in_cell[_ix,_iy-1]):
            self.neighbor_list[_ip, self.n_neighbors[_ip] ] = self.cell_list[_ix][_iy-1][j]
            self.n_neighbors[_ip] += 1
          if _ix+1 == self.nx_cells: #out of bounds of celllist
            for j in range(self.n_in_cell[0,_iy-1]):
              self.neighbor_list[_ip, self.n_neighbors[_ip] ] = self.cell_list[0][_iy-1][j]
              self.n_neighbors[_ip] += 1
          else:
            for j in range(self.n_in_cell[_ix+1,_iy-1]):
              self.neighbor_list[_ip, self.n_neighbors[_ip] ] = self.cell_list[_ix+1][_iy-1][j]
              self.n_neighbors[_ip] += 1


# Integration der eom ----------------------------------------------------------
  cpdef void integrate(self):
    self.rdiff = sqrt(2*self.Dr/self.dt)*np.random.normal(0, 1, self.n)
    self.om = self.om_n+ self.rdiff + self.alignments #om_mu or om_n
    self.phi = self.phi + self.om * self.dt
    if self.if_vnoise == 0:
      self.r = self.r + ((self.v*np.array([np.cos(self.phi), np.sin(self.phi)])).T + self.forces) * self.dt
    else:
      self.rdiff_t = sqrt(2/3*self.Dr/self.dt)*np.random.normal(0, 1, self.n) #D_r=3 D_t
      self.r = self.r + (((self.v+ self.rdiff_t) *np.array([np.cos(self.phi), np.sin(self.phi)])).T + self.forces) * self.dt
    cdef int i
    # periodic boudaries
    for i in range(self.n):
      if self.r[i,0] > self.box_hl_a:
        self.r[i,0] -= 2*self.box_hl_a
      elif self.r[i,0] < -self.box_hl_a:
        self.r[i,0] += 2*self.box_hl_a
      if self.r[i,1] > self.box_hl_b:
        self.r[i,1] -= 2*self.box_hl_b
      elif self.r[i,1] < -self.box_hl_b:
        self.r[i,1] += 2*self.box_hl_b


# Simulation--------------------------------------------------------------------
  cpdef void run_save(self):
    traj='traj_pb_n' + str(self.n) + '_v' +str(self.v) + "_ommu" + str(self.om_mu)+ "_omsig" + str(self.om_sig)+ "_rho" + str(self.rho)+ "_box_ratio" + str(self.box_ratio)+  "_kap" + str(self.if_align*self.kap) +"_eps" + str(self.if_wca*self.eps) +  "_rK" + str(self.rK) + "_t" + str(int(round(self.trun*self.dt))) + "_omconst" + str(int(self.if_omconst)) + "_omdist" + self.om_dist + "_vnoise" + str(int(self.if_vnoise)) + "_init" + str(int(self.init_form)) + "_num" + str(int(self.num)) +  ".pickle"
    data = {}
    data["N"]=self.n
    data["box_hl_a"]=self.box_hl_a
    data["box_hl_b"]=self.box_hl_b
    data["box_ratio"]=self.box_ratio
    data["cycle"]=self.dt*self.save_steps
    data["rho"]=self.rho
    data["eps"]=self.eps
    data["kap"]=self.kap
    data["rK"]=self.rK
    data["rcut"]=self.rcut
    data["Dr"]=self.Dr
    data["v"]=self.v
    data["om_mu"]=self.om_mu
    data["om_sig"]=self.om_sig
    data["om_n"]=self.om_n
    data["om_dist"]=self.om_dist
    cdef int t
    cdef int _tsave = int(self.trun/self.save_steps)+1
    cdef np.ndarray _t_all = np.zeros(_tsave, dtype=np.float64)
    cdef np.ndarray _phi_all = np.zeros((_tsave, self.n), dtype=np.float64)
    cdef np.ndarray _r_all = np.zeros((_tsave, self.n, 2), dtype=np.float64)
    cdef np.ndarray _align_all = np.zeros((_tsave, self.n), dtype=np.float64)
    cdef np.ndarray _om0_all = np.zeros(_tsave, dtype=np.float64)
    cdef int _i_savestep = 0
    _t_all[0] = 0
    _r_all[0] = self.r
    _phi_all[0] = self.phi
    _align_all[0] = self.alignments
# RUN SIMULATION
    for t in range(self.trun):
      if t % self.neighbor_steps == 0:
        self.cell_list_update()
        self.neighbor_list_update()
      self.distance_fct()
      if self.if_align == 1:
        self.alignment()
      if self.if_wca == 1:
        if self.if_wca_only_at_start ==0:
          self.WCA()
        else:
          if t < self.trun/2:
            self.WCA()
      self.integrate()

      if t % self.save_steps == 0:
        _i_savestep += 1
        _t_all[_i_savestep] = t*self.dt
        _r_all[_i_savestep] = self.r
        _phi_all[_i_savestep] = self.phi
        _align_all[_i_savestep] = self.alignments
        _om0_all[_i_savestep] = self.om0
        print(_t_all[_i_savestep])

      if (t+1) % int(self.trun/4) == 0:
        data["t"]=_t_all
        data["r"]=_r_all
        data["phi"]=_phi_all
        data["align"]=_align_all
        data["om0"]=_om0_all
        with open(traj, 'wb') as handle:
          pickle.dump(data,handle)
