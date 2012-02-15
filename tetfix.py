import numpy as np

def unbox(coords, box):
    #mxle3 returns unboxed for which box * unboxed = coords
    det = np.linalg.det(box)
    assert np.abs(det) >= 1.0e-14
    dx = np.zeros_like(coords)
    dx[:,0]=coords[:,0]*box[1,1]*box[2,2]+box[0,1]*box[1,2]*coords[:,2]+coords[:,1]*box[2,1]*box[0,2]-box[0,2]*box[1,1]*coords[:,2]-coords[:,0]*box[1,2]*box[2,1]-box[2,2]*box[0,1]*coords[:,1]
    dx[:,1]=box[0,0]*coords[:,1]*box[2,2]+coords[:,0]*box[1,2]*box[2,0] +box[1,0]*coords[:,2]*box[0,2]-box[0,2]*coords[:,1]*box[2,0] -box[0,0]*box[1,2]*coords[:,2] -box[2,2]*coords[:,0]*box[1,0]
    dx[:,2]=box[0,0]*box[1,1]*coords[:,2] +box[0,1]*coords[:,1]*box[2,0] +box[1,0]*box[2,1]*coords[:,0] -coords[:,0]*box[1,1]*box[2,0] -box[0,0]*coords[:,1]*box[2,1]-coords[:,2]*box[0,1]*box[1,0]
    dx /= det
    return dx

perm = np.zeros((3,3,3), dtype='int')
perm[0,1,2] = 1
perm[1,2,0] = 1
perm[2,0,1] = 1
perm[0,2,1] = -1
perm[2,1,0] = -1
perm[1,0,2] = -1

class tetrahedral_orientation_fix:
    def __init__(self, typnam, box):
        # feed initial coordinates 
        self.box = box
        tetcnt = 0
        self.ntet = 0
        pos = []
        newpos = None
        for (nm, cnt) in zip(typnam, range(len(typnam))):
            if nm == 'tet':
                tetcnt += 1
                if newpos is None:
                    newpos = cnt 
                if tetcnt == 4:
                    self.ntet += 1
                    tetcnt = 0
                    pos.append(newpos)
                    newpos = None
        self.tetcnt = np.array(pos)

    def set_original_coords(self, coord):
        self.org_coord = coord.copy()
        self.org_coord_ctr = []
        self.X = []
        for jtet in range(self.ntet):
            pos = self.tetcnt[jtet]*3
            coord_tet = coord.flat[pos:pos+12].reshape((4,3))
            # handle box
            # only works for cubic unit cells!
            coord_tet = np.matrix(coord_tet)*np.matrix(self.box.T) 
            coord_diff = coord_tet[1:]-coord_tet[0]
            xbox = self.box[0,0]
            coord_diff += ((coord_diff<-xbox/2)-(coord_diff>xbox/2))*xbox

            coord_tet[1:] = coord_diff+coord_tet[0]

            coord_ctr = coord_tet - np.average(coord_tet,axis=0)

            self.org_coord_ctr.append(coord_ctr)
            X = np.outer(coord_ctr.flat, coord_ctr.flat).reshape((4,3,4,3)).transpose([1,3,0,2])
            self.X.append(X)

    def fixorientation(self, coord):
        # Return orientation of the tetrahedra
        # Using deformation tensor polar decomposition
        coordorg = coord.copy()
        for jtet in range(self.ntet):
            pos = self.tetcnt[jtet]*3
            coord_tet = coord.flat[pos:pos+12].reshape((4,3))
            # handle box
            # only works for cubic unit cells!
            coord_tet = np.matrix(coord_tet)*np.matrix(self.box.T) 
            coord_diff = coord_tet[1:]-coord_tet[0]
            xbox = self.box[0,0]
            coord_diff = (coord_diff+xbox/2)%xbox-xbox/2

            coord_tet[1:] = coord_diff+coord_tet[0]

            coord_ctr = coord_tet - np.average(coord_tet,axis=0)

            # Calculate Y
            Y = np.array(np.dot(coord_ctr,self.org_coord_ctr[jtet].T)).reshape([16])
            X = self.X[jtet].reshape(9,16)

            # Find F
            import scipy.optimize as O
            F = np.zeros(9)
            def Axb(x,A,b):
                return (np.dot(A,x)-b)**2
            results = O.leastsq(Axb, F, args=(X.T,Y))
            F = results[0].reshape((3,3))
            # temp fix - dirty fix FIXME
            # This fixes det(R) becoming different from 1
            # Tested in numpy 1.4.x not in 1.6.x
            F *= (np.abs(F)>1.0e-12)
            F = np.matrix(F)
        
            # Polar decomposition
            import numpy.linalg as L
            w,v = L.eig(np.dot(F.T,F))
            P = v*np.diag(np.sqrt(w))*v.T
            R = F*L.inv(P)
            detR = L.det(R)
            #print detR

            # FIXME - Some gives a improper rotation which seems to screw
            # things up
            # Rotate back
            #print jtet, np.max(np.abs(R.T*R-np.diag([1,1,1])))
            #print jtet, R-np.diag([1,1,1])
            #print jtet, np.max(np.abs(R-np.diag([1,1,1])))
            new_coord_ctr = np.dot(coord_ctr,R.T)
            diff = unbox(new_coord_ctr - coord_ctr, self.box)
            coord.flat[pos:pos+12] += diff.flat
            coord.flat[pos:pos+12] += 0.5
            coord.flat[pos:pos+12] %= 1.0
            coord.flat[pos:pos+12] -= 0.5
            

    def constrain(self, coord, direction):
        for jtet in range(self.ntet):
            pos = self.tetcnt[jtet]*3
            coord_tet = coord.flat[pos:pos+12].reshape((4,3))
            # handle box
            # only works for cubic unit cells!
            coord_tet = np.matrix(coord_tet)*np.matrix(self.box.T) 
            coord_diff = coord_tet[1:]-coord_tet[0]
            xbox = self.box[0,0]
            coord_diff += ((coord_diff<-xbox/2)-(coord_diff>xbox/2))*xbox

            coord_tet[1:] = coord_diff+coord_tet[0]

            coord_ctr = coord_tet - np.average(coord_tet,axis=0)
            dir_tet = direction.flat[pos:pos+12].reshape((4,3))
            dir_tet = np.matrix(dir_tet)*np.matrix(self.box.T)
            dir_ctr = dir_tet - np.average(dir_tet,axis=0)

            for xyz in range(3):
                lui = -np.dot(coord_ctr,perm[:,:,xyz])
                lsq = (np.array(lui)**2).sum()
                direction.flat[pos:pos+12] -= unbox((lui*((np.array(lui)*np.array(dir_ctr)).sum())/lsq).reshape(4,3), self.box).flat
        return direction 
