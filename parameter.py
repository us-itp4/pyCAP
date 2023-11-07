import numpy as np



class Parameter:

    def __init__(self, parameterfile):
        with open(parameterfile, 'r') as par:
            parameter = [x.split() for x in par.readlines()]
        par.close()

        self.n_p = int(parameter[0][1])
        self.dt = float(parameter[1][1])
        self.trun = float(parameter[2][1])
        self.v = float(parameter[3][1])
        self.om_mu = float(parameter[4][1])
        self.om_sig = float(parameter[5][1])
        self.Dr = float(parameter[6][1])
        self.rcut = float(eval(parameter[7][1]))
        self.rho = float(parameter[8][1])
        self.box_ratio = float(parameter[9][1])
        self.dt_save = float(parameter[10][1])
        self.eps = float(parameter[11][1])
        self.if_align = bool(int(parameter[12][1]))
        self.if_wca = bool(int(parameter[13][1]))
        self.kap = float(parameter[14][1])
        self.rK = float(eval(parameter[15][1]))
        self.dt_neighborupdate = float(parameter[16][1])
        self.init_form = int(parameter[17][1])
        self.if_omconst = bool(int(parameter[18][1]))
        self.if_vnoise = bool(int(parameter[19][1]))
        self.om_dist = str(parameter[20][1])
        self.if_wca_only_at_start = bool(int(parameter[21][1]))
        self.num = int(parameter[22][1])
