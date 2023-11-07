import simulation
import sim_confined
import numpy as np
import time
from parameter import Parameter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--parafile", type=str)
args = parser.parse_args()

p = Parameter(args.parafile)

start = time.time()
#sim = sim_confined.Simulation(p.n,p.dt,p.trun,p.v,p.om_mu,p.om_sig,p.Dr,p.rcut,p.rho,p.dt_save,p.eps,p.if_align,p.if_wca,p.kap,p.rK,p.dt_neighborupdate,p.init_form,p.if_omconst,p.om_dist,p.num)
# for confined simulations use the following instead
sim = simulation.Simulation(p.n_p,p.dt,p.trun,p.v,p.om_mu,p.om_sig,p.Dr,p.rcut,p.rho,p.box_ratio,p.dt_save,p.eps,p.if_align,p.if_wca,p.kap,p.rK,p.dt_neighborupdate,p.init_form,p.if_omconst,p.if_vnoise,p.om_dist,p.if_wca_only_at_start,p.num)
sim.initialize()
sim.run_save()

ende = time.time()
print('Time:')
print('{:5.3f}s'.format(ende - start), end='  ')
