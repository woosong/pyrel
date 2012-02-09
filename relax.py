from eam import *
import numpy as np
import numpy.linalg as L
from scipy import optimize as O
import tetfix

def get_energy(coords, model):
    model.coords = coords.reshape(model.coords.shape)
    return model.get_energy()

def get_force(coords, model):
    model.coords = coords.reshape(model.coords.shape)
    return model.get_force().flatten()*model.box[0,0]

def get_hessian(coords, model):
    model.coords = coords.reshape(model.coords.shape)
    return model.get_hessian().reshape([coords.size, coords.size])

def get_hessian_directional(coords, d, model):
    model.coords = coords.reshape(model.coords.shape)
    return model.get_hessian_directional(d)*model.box[0,0]*model.box[0,0]

class relaxation:
    def __init__(self, model, xyzfile):
        self.model = model
        box, coords, ityp, ityp2, typnam = readfiles.readxyz(xyzfile)
        self.typnam = typnam
        # dirty change
        self.model.set_atoms(box, coords, ityp)
        self.constraints = []
    
    def addconstraint(self, constraint):
        self.constraints.append(constraint)

    def relax(self, steps=1000, hardconstraint=False):
        gamma = 0.4
        for i in range(steps):
            f = self.model.get_force()
            d = gamma*self.model.get_relaxation_step()/self.model.box[0,0]
            for const in self.constraints:
                d = const.constrain(self.model.coords, d)
            self.model.move_atoms(d)
            if hardconstraint and i%10==0:
                newcoord = self.model.coords.copy()
                for const in self.constraints:
                    const.fixorientation(newcoord)
                self.model.move_atoms(newcoord-self.model.coords)
            energy = self.model.get_energy()
            print "Step ", i, energy, (f*d).sum()/self.model.box[0,0]
            
    def run_cg(self, steps=1000, err=1.0e-10, eps=1.0e-3):
        # Run nonlinear CG
        x = self.model.coords.flatten()
        n = len(x)
        i = 0
        k = 0
        r = get_force(x, self.model)
        d = r
        dn = L.norm(r)
        d0 = dn
        jmax = 10
        while i < steps and dn > eps**2*d0:
            dd = L.norm(d)
            """
            results = O.line_search(get_energy, get_force, x, d, gfk=r, args=tuple([self.model]))
            alpha = results[0]
            x += alpha*d
            x += 0.5
            x %= 1.0
            x -= 0.5
            """
            for j in range(jmax):
                alpha = np.dot(get_force(x,self.model),d)/get_hessian_directional(x,d,self.model)
                x += alpha*d
                x += 0.5
                x %= 1.0
                x -= 0.5
                if (alpha**2 * dd > eps**2):
                    break
            r = get_force(x, self.model)
            do = dn
            dn = L.norm(r)
            beta = dn/do
            d = r + beta*d
            k += 1
            if k == n or np.dot(r,d) <= 0:
                d = r
                k = 0
            i += 1
            energy = get_energy(x, self.model)
            print "Step ", i, energy, dn, d0

if __name__ == '__main__':
    model = eam_model('eam.tab', 7.0, 0.9)
    relaxation = relaxation(model, '1-7.xyz')
    constraint = tetfix.tetrahedral_orientation_fix(relaxation.typnam, model.box)
    constraint.set_original_coords(model.coords)
    #relaxation.run_cg(steps=100)
    relaxation.addconstraint(constraint)
    relaxation.relax(steps=100, hardconstraint=True)
    #constraint.fixorientation(model.coords)
    #relaxation.relax(steps=1)
    #constraint.fixorientation(model.coords)
