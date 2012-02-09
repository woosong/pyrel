import numpy as np
import readfiles
import neighborlist
import spline
from scipy import weave as W

class eam_model:
    def __init__(self, eamfile, rcut, rcutsm):
        # Read eam file
        (nvr,scr,npp,sce,pairs), (rs,Vs,er), (mele,rce0,rce1,dre,em) = readfiles.readeampot(eamfile)
        (self.nvr,self.scr,self.npp,self.sce,self.pairs), (self.rs,self.Vs,self.er), (self.mele,self.rce0,self.rce1,self.dre,self.em) = (nvr,scr,npp,sce,pairs), (rs,Vs,er), (mele,rce0,rce1,dre,em)

        self.cutoff = rcut
        # Setup splines
        self.V_spl = spline.spline(rs, Vs.T, rcut=rcut, rcutsm=rcutsm)
        self.rho_spl = spline.spline(rs, er.T, rcut=rcut, rcutsm=rcutsm)
        rce = np.array([[rce0+dre*i for i in range(nvr)] for (rce0,dre) in zip(self.rce0, self.dre)])
        ems = np.array([e for e in em])
        self.em_spl = spline.spline(rce.T, ems.T)

        # Setup type tables for fast access
        self.ntypes = len(self.mele)
        self.pair_table = np.zeros((self.ntypes, self.ntypes), dtype='int')
        mele = list(mele)
        self.mele = mele
        for pair, cnt in zip(self.pairs, range(self.npp)):
            i,j = pair
            i_idx = mele.index(i)
            j_idx = mele.index(j)
            self.pair_table[i_idx, j_idx] = cnt
            if i_idx != j_idx:
                self.pair_table[j_idx, i_idx] = cnt

    def set_atoms(self, box, coords, types):  
        self.box = box
        self.coords = coords
        self.neighborlist = neighborlist.neighborlist(box, self.coords, self.cutoff)
        self.neighborlist.build()
        self.atom_types = np.array([self.mele.index(a) for a in types], dtype='int')

    def move_atoms(self, dx):
        self.neighborlist.move(dx)
      
    def calc_rij_and_lists(self):
        support_code = """
        double inline pbc(double x) {
            return (x>0.5) ? x-1.0 : (x<-0.5) ? x+1.0 : x;
        };
        """
        code = """
            // Loop through the neighborlist
            double *rij=rijlist;
            for(int k=0; k<n; k++) {
                int i = *(neighborlist++);
                int j = *(neighborlist++);
                double *xi = (coords+3*i);
                double *xj = (coords+3*j);
                // Calculate r_ij
                // FIXME - handle PBC
                double x = pbc(*(xi++)-*(xj++));
                double y = pbc(*(xi++)-*(xj++));
                double z = pbc(*(xi++)-*(xj++));
                //FIXME - indicies right?
                *(rij++) = x*(*(box+0))+y*(*(box+3))+z*(*(box+6));
                *(rij++) = x*(*(box+1))+y*(*(box+4))+z*(*(box+7));
                *(rij++) = x*(*(box+2))+y*(*(box+5))+z*(*(box+8));
                int ai = *(atom_types+i);
                int aj = *(atom_types+j);
                *(i_type++) = ai;
                *(j_type++) = aj;
                *(ij_pairtype++) = *(pair_table+ai*ntypes+aj);
            }
        """
        coords = self.coords
        neighborlist = self.neighborlist.halflist
        atom_types = self.atom_types
        n = len(neighborlist)
        rijlist = np.zeros((n,3), dtype='double')
        i_type = np.zeros(n, dtype='int')
        j_type = np.zeros(n, dtype='int')
        ij_pairtype = np.zeros(n, dtype='int')
        pair_table = self.pair_table
        box = self.box
        ntypes = self.ntypes
        W.inline(code, ['box', 'neighborlist', 'n', 'rijlist', 'coords', 'atom_types', 'i_type', 'j_type', 'ij_pairtype', 'pair_table', 'ntypes'], support_code=support_code)
        return rijlist, i_type, j_type, ij_pairtype

    def get_energy(self):
        rijlist, i_type, j_type, ij_pairtype = self.calc_rij_and_lists()
        rs = np.sqrt((rijlist**2).sum(axis=1))
        print rs
        V = self.V_spl.eval_y_multi(rs, ij_pairtype)
        print V
        rhoi = self.rho_spl.eval_y_multi(rs, j_type)
        rhoj = self.rho_spl.eval_y_multi(rs, i_type)
        rho_i = np.zeros(len(self.coords), dtype='double')
        code = """
            for(int k=0; k<n; k++) {
                int i = *(neighborlist++);
                int j = *(neighborlist++);
                *(rho_i+i) += *(rhoi++);
                // Using half neighborlist, need to be duplicated
                *(rho_i+j) += *(rhoj++);
            }
        """
        neighborlist = self.neighborlist.halflist
        n = len(neighborlist)
        W.inline(code, ['neighborlist', 'n', 'rhoi', 'rhoj',  'rho_i'])
        F = self.em_spl.eval_y_multi(rho_i, self.atom_types)
        return V.sum()+F.sum()

    def get_force(self):
        rijlist, i_type, j_type, ij_pairtype = self.calc_rij_and_lists()
        rs = np.sqrt((rijlist**2).sum(axis=1))
        rijhat = (rijlist.T/rs).T
        dV = self.V_spl.eval_y_multi(rs, ij_pairtype, d=1)

        N = len(self.coords)
        rhoi, drhoi = self.rho_spl.eval_y_multi(rs, j_type, d=(0,1))
        rhoj, drhoj = self.rho_spl.eval_y_multi(rs, i_type, d=(0,1))
        rho_i = np.zeros(N, dtype='double')
        code = """
            for(int k=0; k<n; k++) {
                int i = *(neighborlist++);
                int j = *(neighborlist++);
                *(rho_i+i) += *(rhoi++);
                // Using half neighborlist, need to be duplicated
                *(rho_i+j) += *(rhoj++);
            }
        """
        neighborlist = self.neighborlist.halflist
        n = len(neighborlist)
        W.inline(code, ['neighborlist', 'n', 'rhoi', 'rhoj',  'rho_i', 'drhoi', 'drhoj', 'drho'])
        dF = self.em_spl.eval_y_multi(rho_i, self.atom_types, d=1)*0

        F = np.zeros((N,3), dtype='double')
        code = """
            for(int k=0; k<n; k++) {
                int i = *(neighborlist++);
                int j = *(neighborlist++);
                double *Fi = F+i*3;
                double *Fj = F+j*3;
                *(Fi++) -= ((*dV+(*(dF+i)**(drhoj))-(*(dF+j)**(drhoi)))*(*rijhat));
                *(Fi++) -= ((*dV+(*(dF+i)**(drhoj))-(*(dF+j)**(drhoi)))*(*(rijhat+1)));
                *(Fi++) -= ((*dV+(*(dF+i)**(drhoj))-(*(dF+j)**(drhoi)))*(*(rijhat+2)));
                *(Fj++) += ((*dV+(*(dF+j)**(drhoi))-(*(dF+i)**(drhoj)))*(*rijhat));
                *(Fj++) += ((*dV+(*(dF+j)**(drhoi))-(*(dF+i)**(drhoj)))*(*(rijhat+1)));
                *(Fj++) += ((*dV+(*(dF+j)**(drhoi))-(*(dF+i)**(drhoj)))*(*(rijhat+2)));
                drhoi++;
                drhoj++;
                dV++;
                rijhat+=3;
            }
        """
        W.inline(code, ['F', 'dV', 'drhoi', 'drhoj', 'rijhat', 'dF', 'n', 'neighborlist']) 
        return F

model = eam_model('eam.tab', 7.0, 0.9)
box, coords, ityp, ityp2, typnam = readfiles.readxyz('test.xyz')
# dirty change
model.set_atoms(box, coords, ityp)
print model.get_energy()
force = model.get_force()
print force
#force_norm = np.sqrt((force*force).sum(axis=1))
