import numpy as np
from scipy import weave as W

CUTOFF_RATIO = 1.31
print "cutoff_ratio = ", CUTOFF_RATIO

# Full Neighbor list
class neighborlist:
    def __init__(self, box, coords, cutoff, cutoff_ratio=CUTOFF_RATIO):
        self.box = box
        self.coords = (coords)
        self.N = len(self.coords)
        self.dr = np.zeros(len(coords))
        self.cutoff = cutoff
        self.cutoff_ratio = cutoff_ratio
        self.cutoff2 = (cutoff*cutoff_ratio)**2
        self.cutoff_dr = cutoff*(cutoff_ratio-1.)*0.5

    def build_fulllist(self):
        #build full list from neighborlist
        fulllist = []
        for i in self.neighborlist.keys().sorted():
            for j in self.neighborlist[i]:
                fulllist.append([i,j])
        self.fulllist = np.array(fulllist)

    def build(self):
        #FIXME does not work with PBC box
        assert False
        self.dr = np.zeros(len(self.coords))
        neighbors = {}
        half_list = []
        coords = self.coords
        for i in range(self.N):
            neighbors[i] = []
        for i in range(self.N):
            xi = coords[i]
            for j in range(i+1,self.N):
                xj = coords[j]
                dx = xi - xj
                dr = (dx*dx).sum()
                if (dr<self.cutoff2):
                    # tentative neighbor
                    neighbors[i].append(j)
                    neighbors[j].append(j)
                    half_list.append([i,j])
        self.neighborlist = neighbors
        self.halflist = np.array(half_list)
        #self.build_fulllist()

    def fastbuild(self):
        self.dr = np.zeros(len(self.coords))
        neighbors = {}
        half_list = []
        N = self.N
        coords = self.coords
        half_list = np.zeros((N*N, 2), dtype='int')
        cutoff2 = self.cutoff2
        support_code = """
        double inline pbc(double x) {
            return (x>0.5) ? x-1.0 : (x<-0.5) ? x+1.0 : x;
        };
        """
        code = """
            for(int i=0; i<N; i++) {
                double *xi = coords+i*3;
                for(int j=i+1; j<N; j++) {
                    double *xj = coords+j*3;
                    
                    double x = pbc(*(xi) - *(xj++));             
                    double y = pbc(*(xi+1) - *(xj++));             
                    double z = pbc(*(xi+2) - *(xj++));             
                    //FIXME - indicies right?
                    double realx = x*(*(box+0))+y*(*(box+3))+z*(*(box+6));
                    double realy = x*(*(box+1))+y*(*(box+4))+z*(*(box+7));
                    double realz = x*(*(box+2))+y*(*(box+5))+z*(*(box+8));
                    double r = realx*realx+realy*realy+realz*realz;
                    if (r<cutoff2) {
                        *(half_list++) = i;
                        *(half_list++) = j;
                        (*cnt)++;
                    }
                }
            }
        """
        box = self.box
        cnt = np.zeros(1, dtype='int')
        W.inline(code, ['box','coords', 'N', 'half_list', 'cnt', 'cutoff2'], support_code=support_code)
        half_list = half_list[:cnt[0]].astype('int')
        #self.neighborlist = neighbors
        self.halflist = half_list
        self.neighborlist = {}
        for i in range(self.N):
            self.neighborlist[i] = []
        for i,j in self.halflist:
            self.neighborlist[i].append(j)
            self.neighborlist[j].append(i)
        #self.build_fulllist()
    build = fastbuild

    def move(self, dx):
        self.dr += np.linalg.norm(dx)
        # if any atom moved by possibly longer than half the cutoff distance,
        # they need to be recalculated
        if (self.dr > self.cutoff_dr).any():
            #rebuild everything - simple but maybe slow for "fast" molecules
            #for solids at low T this might be good enough for a long time
            self.build()

    def get_neighbors(self, i):
        return self.neighborlist[i]

    def get_halflist(self):
        return self.halflist

