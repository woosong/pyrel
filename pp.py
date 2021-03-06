import numpy as np
import readfiles
import neighborlist
import spline
from scipy import weave as W

class pp_model:
    def __init__(self, ppfile, rcut, rcutsm):
        # Read pp file
        (nvr,scr,npp,sce,pairs), (rs,Vs) = readfiles.readpairpot(ppfile)
        (self.nvr,self.scr,self.npp,self.sce,self.pairs), (self.rs,self.Vs) = (nvr,scr,npp,sce,pairs), (rs,Vs)

        self.cutoff = rcut
        # Setup splines
        self.V_spl = spline.spline(rs, Vs.T, rcut=rcut, rcutsm=rcutsm)

        self.mele = list(set(sorted(pairs.flatten())))
        # Setup type tables for fast access
        self.ntypes = len(self.mele)
        self.pair_table = np.zeros((self.ntypes, self.ntypes), dtype='int')
        mele = list(self.mele)
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
        self.coords += dx
        self.coords = ((self.coords+0.5)%1.0)-0.5
        self.neighborlist.coords = self.coords
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
                double rx = x*(*(box+0))+y*(*(box+3))+z*(*(box+6));
                double ry = x*(*(box+1))+y*(*(box+4))+z*(*(box+7));
                double rz = x*(*(box+2))+y*(*(box+5))+z*(*(box+8));
                bool cutcond = (rx*rx+ry*ry+rz*rz < cutoff*cutoff);
                *(rcutcond++) = cutcond;
                if (cutcond) {
                    (*count)++;
                    *(rij++) = rx;
                    *(rij++) = ry;
                    *(rij++) = rz;
                    int ai = *(atom_types+i);
                    int aj = *(atom_types+j);
                    *(i_type++) = ai;
                    *(j_type++) = aj;
                    *(ij_pairtype++) = *(pair_table+ai*ntypes+aj);
                }
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
        rcutcond = np.zeros(n, dtype='bool')
        pair_table = self.pair_table
        box = self.box
        ntypes = self.ntypes
        cutoff = self.cutoff
        count = np.zeros(1, dtype='int')
        W.inline(code, ['box', 'neighborlist', 'n', 'rijlist', 'coords', 'atom_types', 'i_type', 'j_type', 'ij_pairtype', 'rcutcond', 'count', 'cutoff', 'pair_table', 'ntypes'], support_code=support_code)
        count = count[0]
        return rijlist[:count], i_type[:count], j_type[:count], ij_pairtype[:count], rcutcond

    def get_energy(self):
        rijlist, i_type, j_type, ij_pairtype, rcutcond = self.calc_rij_and_lists()
        rs = np.sqrt((rijlist**2).sum(axis=1))
        V = self.V_spl.eval_y_multi(rs, ij_pairtype)
        return V.sum()

    def get_force(self):
        rijlist, i_type, j_type, ij_pairtype, rcutcond = self.calc_rij_and_lists()
        rs = np.sqrt((rijlist**2).sum(axis=1))
        rijhat = (rijlist.T/rs).T.copy()
        dV = self.V_spl.eval_y_multi(rs, ij_pairtype, d=1)

        N = len(self.coords)
        neighborlist = self.neighborlist.halflist
        n = len(neighborlist)

        F = np.zeros((N,3), dtype='double')
        code = """
            for(int k=0; k<n; k++) {
                int i = *(neighborlist++);
                int j = *(neighborlist++);
                if (*(rcutcond++)) {
                    double *Fi = F+i*3;
                    double *Fj = F+j*3;
                    double delF = (*dV);
                    *(Fi++) -= delF*(*rijhat);
                    *(Fi++) -= delF*(*(rijhat+1));
                    *(Fi++) -= delF*(*(rijhat+2));
                    *(Fj++) += delF*(*rijhat);
                    *(Fj++) += delF*(*(rijhat+1));
                    *(Fj++) += delF*(*(rijhat+2));
                    dV++;
                    rijhat+=3;
                }
            }
        """
        W.inline(code, ['F', 'rcutcond', 'dV', 'rijhat', 'n', 'neighborlist']) 
        return F

    def get_pressure(self, GPa=False):
        rijlist, i_type, j_type, ij_pairtype, rcutcond = self.calc_rij_and_lists()
        rs = np.sqrt((rijlist**2).sum(axis=1))
        dV = self.V_spl.eval_y_multi(rs, ij_pairtype, d=1)

        N = len(self.coords)
        neighborlist = self.neighborlist.halflist
        n = len(neighborlist)

        P = np.zeros(1, dtype='double')
        code = """
            for(int k=0; k<n; k++) {
                int i = *(neighborlist++);
                int j = *(neighborlist++);
                if (*(rcutcond++)) {
                    double delF = (*dV);
                    *P -= delF*(*(rs++));
                    dV++;
                }
            }
        """
        W.inline(code, ['P', 'rs', 'rcutcond', 'dV', 'n', 'neighborlist']) 
        V = np.linalg.det(self.box)
        if GPa:
            return P/3/V*160.217646
        return P/3/V

    def get_hessian_directional(self, d):
        rijlist, i_type, j_type, ij_pairtype, rcutcond = self.calc_rij_and_lists()
        rs = np.sqrt((rijlist**2).sum(axis=1))
        rijhat = (rijlist.T/rs).T.copy()
        dV, ddV = self.V_spl.eval_y_multi(rs, ij_pairtype, d=(1,2))

        N = len(self.coords)
        neighborlist = self.neighborlist.halflist
        n = len(neighborlist)

        H = np.zeros(1, dtype='double')
        code = """
            for(int k=0; k<n; k++) {
                int i = *(neighborlist++);
                int j = *(neighborlist++);
                if (*(rcutcond++)) {
                    double d_dot_ri = (*rijhat)*(*(d+i*3));
                    d_dot_ri += (*(rijhat+1))*(*(d+1+i*3));
                    d_dot_ri += (*(rijhat+2))*(*(d+2+i*3));
                    double d_dot_rj = (*rijhat)*(*(d+j*3));
                    d_dot_rj += (*(rijhat+1))*(*(d+1+j*3));
                    d_dot_rj += (*(rijhat+2))*(*(d+2+j*3));

                    *H += (*ddV)*(d_dot_ri-d_dot_rj)*(d_dot_ri-d_dot_rj);
                    ddV++;
                    rijhat+=3;
                }
            }
        """
        W.inline(code, ['d','H', 'ddV', 'rijhat', 'rcutcond', 'n', 'neighborlist']) 
        return H[0]

    def get_relaxation_step(self):
        #Implement the Marek's relaxation step determining method
        rijlist, i_type, j_type, ij_pairtype, rcutcond = self.calc_rij_and_lists()
        rs = np.sqrt((rijlist**2).sum(axis=1))
        rijhat = (rijlist.T/rs).T.copy()
        dV, ddV = self.V_spl.eval_y_multi(rs, ij_pairtype, d=(1,2))

        N = len(self.coords)
        neighborlist = self.neighborlist.halflist
        n = len(neighborlist)

        F = np.zeros((N,3), dtype='double')
        M = np.zeros((N,3,3), dtype='double')
        code = """
            for(int k=0; k<n; k++) {
                int i = *(neighborlist++);
                int j = *(neighborlist++);
                if (*(rcutcond++)) {
                    double *Fi = F+i*3;
                    double *Fj = F+j*3;
                    double delF = *dV;
                    *(Fi++) -= delF*(*rijhat);
                    *(Fi++) -= delF*(*(rijhat+1));
                    *(Fi++) -= delF*(*(rijhat+2));
                    *(Fj++) += delF*(*rijhat);
                    *(Fj++) += delF*(*(rijhat+1));
                    *(Fj++) += delF*(*(rijhat+2));
                    double *Mij = M+i*9;
                    double *Mji = M+j*9;
                    double dM = delF/(*(rs++));
                    double ddM = (*ddV);
                    for(int u=0; u<3; u++)
                        for(int v=0; v<3; v++) {
                                double rijhatmat = *(rijhat+u)**(rijhat+v);
                                *(Mij++) += (ddM-dM)*rijhatmat;
                                *(Mji++) += (ddM-dM)*rijhatmat;
                                if (u==v) {
                                    *(Mij-1)+= dM;
                                    *(Mji-1)+= dM;
                                }
                    }
                    dV++;
                    ddV++;
                    rijhat+=3;
                }
            }
        """
        W.inline(code, ['F', 'M', 'rs', 'rcutcond', 'dV', 'ddV', 'rijhat', 'n', 'neighborlist']) 

        d = np.zeros((N,3), dtype='double')
        ifail = np.zeros(1, dtype='int')
        # Use above to find d. Solve Ax=b for x.
        code = """
            for(int i=0; i<N; i++) {
                double *A = M+i*9;
                double *b = F+i*3;
                double *x = d+i*3;
                double det=A[0]*A[4]*A[8]+A[1]*A[5]*A[6] +A[3]*A[7]*A[2]-A[2]*A[4]*A[6] -A[0]*A[5]*A[7]-A[8]*A[1]*A[3];
                if (abs(det)<1.0e-14) *ifail = 1;
                *(x++)=(b[0]*A[4]*A[8]+A[1]*A[5]*b[2]+b[1]*A[7]*A[2]-A[2]*A[4]*b[2]-b[0]*A[5]*A[7]-A[8]*A[1]*b[1])/det;
                *(x++)=(A[0]*b[1]*A[8]+b[0]*A[5]*A[6]+A[3]*b[2]*A[2]-A[2]*b[1]*A[6]-A[0]*A[5]*b[2]-A[8]*b[0]*A[3])/det;
                *(x++)=(A[0]*A[4]*b[2]+A[1]*b[1]*A[6]+A[3]*A[7]*b[0]-b[0]*A[4]*A[6]-A[0]*b[1]*A[7]-b[2]*A[1]*A[3])/det;
            }
        """
        W.inline(code, ['F', 'M', 'd', 'N', 'ifail']);
        return d

if __name__ == '__main__':
    model = pp_model('pp.tab', 7.0, 0.9)
    box, coords, ityp, ityp2, typnam = readfiles.readxyz('1_1-1.xyz')
    # dirty change
    model.set_atoms(box, coords, ityp)
    print model.get_energy()
    force = model.get_force()
    print force
    print "Pressure", model.get_pressure(), "eV/Angstrom^3"
    print "Pressure", model.get_pressure(GPa=True), "GPa"
    #force_norm = np.sqrt((force*force).sum(axis=1))
