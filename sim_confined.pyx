import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libc.math cimport atan2
import cython
import random
import pickle

cdef class Simulation:

  cdef int n, trun, save_steps, nx_cells, ny_cells, count_new, count_old, neighbor_steps, init_form, rand_om_steps, num
  cdef double v, dt, dt_save, Dr, rcut, rho, box_hl, eps, om_mu, om_sig, kap, rK, cell_length_min, dt_neighborupdate, om0
  cdef public np.ndarray r, phi, forces, rdiff, om, distances, distance_vectors, alignments, om_n, cell_list, n_in_cell, neighbor_list, n_neighbors
  cdef bint rnoise, if_align, if_wca, if_omconst
  cdef str om_dist
  def __init__(self,n,dt,trun,v,om_mu,om_sig,Dr,rcut,rho,dt_save,eps,if_align,if_wca,kap,rK,dt_neighborupdate,init_form,if_omconst,om_dist,num):
    self.n=n
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
    self.rho = rho
    self.box_hl=sqrt(self.n/self.rho/np.pi) #sqrt(self.n/self.rho/)/2 (square)
    self.dt_save=dt_save
    self.save_steps = int(self.dt_save/self.dt)
    self.dt_neighborupdate=dt_neighborupdate
    self.neighbor_steps = int(self.dt_neighborupdate/self.dt)
    self.rcut=rcut
    self.eps=eps
    self.if_align=if_align
    self.if_wca=if_wca
    self.num=num
    self.cell_length_min = self.rK + self.v*self.neighbor_steps*self.dt
    self.nx_cells = int(2*self.box_hl / self.cell_length_min)
    self.ny_cells = int(2*self.box_hl / self.cell_length_min)
    self.cell_list = np.zeros((self.nx_cells, self.ny_cells, self.n), dtype=np.int32)
    self.n_in_cell = np.zeros((self.nx_cells, self.ny_cells), dtype=np.int32)
    self.neighbor_list = np.zeros((self.n, self.n), dtype=np.int32)
    self.n_neighbors = np.zeros(self.n, dtype=np.int32)
    self.count_new = 0
    self.count_old = 0
    self.init_form = init_form
    self.if_omconst=if_omconst
    self.om_dist=om_dist

# Initialisierung---------------------------------------------------------------
  cpdef void initialize(self):
    cdef int i
    cdef double mu_sig_equiv
    if self.init_form == 0:
      self.initialize_lattice()
    if self.init_form == 1:
      self.initialize_random()
    elif self.init_form == 2:
      self.initialize_from_file()
      #testing omega_0 same for all particles
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
  cpdef void initialize_lattice(self):
    cdef int _i=0, _ix=0, _iy=0
    cdef int _n_per_side=np.ceil(sqrt(self.n))
    cdef double _dist = 2*self.box_hl / _n_per_side
    cdef np.ndarray _xy = np.zeros(2)
    while _i < self.n :
      if _ix < _n_per_side:
        if _iy < _n_per_side:
          _xy[0] = _dist * _ix - self.box_hl
          _xy[1] = _dist * _iy - self.box_hl
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
  cpdef void initialize_random(self):
    cdef int _i=0, _j=0
    cdef double _x, _y, _dist_x, _dist_y, _r_abs
    cdef np.ndarray _distances = 2*self.rcut*np.ones(self.n)
    _i = 0 #xx
    while _i < self.n :
      _x = np.random.uniform(-self.box_hl+self.rcut*0.5, self.box_hl-self.rcut*0.5)
      _y = np.random.uniform(-self.box_hl+self.rcut*0.5, self.box_hl-self.rcut*0.5)
      _r_abs = sqrt(_x*_x + _y*_y)
      if _r_abs < self.box_hl - self.rcut: #if in circle
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
      else:
        print('init: ', _i ,'/',self.n, 'out of bounds (confinement)')
    self.phi = np.random.uniform(0, 2*np.pi, self.n)

  #load from file
  cpdef void initialize_from_file(self):
    cdef np.ndarray _r, _phi
    cdef char* _filename='init_files/init_waves_co.pickle'
    with open(_filename, 'rb') as handle:
        _data = pickle.load(handle)
    self.r=_data["r"]
    self.phi=_data["phi"]

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
        _distance = sqrt(_dist_x*_dist_x + _dist_y*_dist_y)
        self.distances[i,j] = _distance
        self.distance_vectors[i,j] = [_dist_x,_dist_y]

  cpdef void distance_fct_old(self):
    cdef double _dist_x, _dist_y
    cdef double _distance, _dist6i
    cdef double [:,:] r=self.r
    cdef int i, j
    cdef int _count = 0

    for i in range(self.n):
      for j in range(i+1, self.n):
        self.count_old +=1
        _dist_x = r[j,0] - r[i,0]
        _dist_y = r[j,1] - r[i,1]
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

# Alignement
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


    # Confinement --------------------------------------------------------------------
  cpdef void Confinement_WCA(self):
    cdef double [:,:] forces=self.forces
    cdef double [:,:] r=self.r
    cdef double _ff
    cdef double _r_abs, _distance, _dist6i
    cdef int i
    for i in range(self.n):
      _r_abs = sqrt(r[i,0]*r[i,0] + r[i,1]*r[i,1])
      _richtff_x = - r[i,0]
      _richtff_y = - r[i,1]
      if _r_abs > self.box_hl - self.rcut:#rcut/2
        _distance = self.box_hl - _r_abs
        _dist6i = 1/(_distance*_distance*_distance*_distance*_distance*_distance)
        _ff = 24 * self.eps * (- 2 *_dist6i*_dist6i/_distance + _dist6i/_distance)
        forces[i,0] += _ff * r[i,0]/_r_abs #_richtff_x
        forces[i,1] += _ff * r[i,1]/_r_abs #_richtff_y

  cpdef void Confinement_alignment(self):
    cdef double _r_abs, _align, _phi_wall
    cdef int i
    cdef double [:] alignments = self.alignments
    cdef double [:,:] r=self.r
    for i in range(self.n):
      _r_abs = sqrt(r[i,0]*r[i,0] + r[i,1]*r[i,1])
      if _r_abs > self.box_hl - self.rK:
        _phi_wall = atan2(r[i,1],r[i,0])
        _align = self.kap / (np.pi * self.rK * self.rK) * np.cos(self.phi[i] - _phi_wall)
        alignments[i] += _align

# Cell List--------------------------------------------------------------------
  cpdef void cell_list_update(self):
    cdef int _ix, _iy
    cdef double _cell_length = 2*self.box_hl / self.nx_cells
    self.cell_list = np.zeros((self.nx_cells, self.ny_cells, self.n), dtype=np.int32)
    self.n_in_cell = np.zeros((self.nx_cells, self.ny_cells), dtype=np.int32)
    for i in range(self.n):
      _ix = int((self.r[i,0] + self.box_hl) / _cell_length)
      _iy = int((self.r[i,1] + self.box_hl) / _cell_length)
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
          if ((_ix != -1) and (_iy !=-1)):#for confinement
            for j in range(self.n_in_cell[_ix-1,_iy]):
              self.neighbor_list[_ip, self.n_neighbors[_ip] ] = self.cell_list[_ix-1][_iy][j]
              self.n_neighbors[_ip] += 1
            for j in range(self.n_in_cell[_ix-1,_iy-1]):
              self.neighbor_list[_ip, self.n_neighbors[_ip] ] = self.cell_list[_ix-1][_iy-1][j]
              self.n_neighbors[_ip] += 1
            for j in range(self.n_in_cell[_ix,_iy-1]):
              self.neighbor_list[_ip, self.n_neighbors[_ip] ] = self.cell_list[_ix][_iy-1][j]
              self.n_neighbors[_ip] += 1
            if _ix+1 != self.nx_cells:
              for j in range(self.n_in_cell[_ix+1,_iy-1]):
                self.neighbor_list[_ip, self.n_neighbors[_ip] ] = self.cell_list[_ix+1][_iy-1][j]
                self.n_neighbors[_ip] += 1


# Integration der eom ----------------------------------------------------------
  cpdef void integrate(self):
    self.rdiff = sqrt(2*self.Dr/self.dt)*np.random.normal(0, 1, self.n)
    self.om = self.om_n+ self.rdiff + self.alignments #om_mu or om_n
    self.phi = self.phi + self.om * self.dt
    self.r = self.r + ((self.v*np.array([np.cos(self.phi), np.sin(self.phi)])).T + self.forces) * self.dt
    cdef int i

# Simulation--------------------------------------------------------------------
  cpdef void run_save(self):
    traj='traj_n' + str(self.n) + '_v' +str(self.v) + "_ommu" + str(self.om_mu)+ "_omsig" + str(self.om_sig) + "_rho" + str(self.rho)+  "_kap" + str(self.if_align*self.kap)+  "_rK" + str(self.rK) + "_t" + str(self.trun*self.dt) + "_omconst" + str(int(self.if_omconst)) + "_omdist" + self.om_dist + "_num" + str(int(self.num)) + ".pickle"
    data = {}
    data["N"]=self.n
    data["box_hl"]=self.box_hl
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
        self.Confinement_alignment()
      if self.if_wca == 1:
        self.WCA()
      self.Confinement_WCA()
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
