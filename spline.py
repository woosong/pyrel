import numpy as np
from scipy import weave as W

class spline:
    def __init__(self, x, y, b1=0, bn=0, rcut=None, rcutsm=0.9):
        n = len(x)
        if len(x.shape) == 1:
            x = x.reshape((1,n)) 
            x = np.outer([1]*(y.size/n), x)
        else:
            x = x.transpose()+0.
        # check if equidistant
        dx = x[:,1:]-x[:,:-1]
        diff = (dx.max(axis=1)-dx.min(axis=1))/np.average(dx, axis=1)
        y = (y.transpose()+0.).reshape([y.size/n, n])
        if (diff > 1.0e-5).any():
            print "Non-equidistant mesh not supported"
            assert False
        # apply rcut
        if rcut:
            r0 = rcut*rcutsm
            rc = rcut/r0 # rc = 1/rcutsm
            newx = x/r0
            z = (newx-rc)/(1.-rc)
            fi = 3.*z**4-8.*z**3+6.*z**2
            fi[(newx>=rc).nonzero()] = 0.
            fi[(newx<1.0).nonzero()] = 1.
            #FIXME seems like there is a logic bug? -- ?? what is this?
            # 10 in above fixed to 1.0
            newy = y*fi
            y = newy
        # make initial point
        y[:,0] = y[:,1]*2-y[:,2]

        self.n = n
        # x : array of points
        # y : array of function values
        # n : number of points
        # b1, bn : second derivative of the function at the end points
        # b1=bn=0 if no info available
        # return y1,y2,y3 : arrays of spline coefficients
        # f(x) = (((y3(k)*(x-x(k))+y2(k))*(x-x(k))+y1(k))*(x-x(k))+y(k)
        # where k is max k s.t. x(k) < x 
        y1 = np.zeros_like(y)   
        y2 = np.zeros_like(y)   
        y3 = np.zeros_like(y)   
        #FIXME change to allow multi-x
        c1=(y[:,1]-y[:,0])/(x[:,1]-x[:,0])
        c2=(y[:,2]-y[:,1])/(x[:,2]-x[:,1])
        b1=(c2-c1)/(x[:,2]-x[:,0])*2.
        c1=(y[:,n-1]-y[:,n-2])/(x[:,n-1]-x[:,n-2])
        c2=(y[:,n-2]-y[:,n-3])/(x[:,n-2]-x[:,n-3])
        bn=[c2-c1-1]/(x[:,n-1]-x[:,n-3])*2.
        s=0.
        for i in range(n-1):
            y1[:,i] = x[:,i+1]-x[:,i]
            r = (y[:,i+1]-y[:,i])/y1[:,i]
            y2[:,i] = r-s
            s=r
        s=0.
        r=0.
        y2[:,0]=b1
        y2[:,n-1]=bn
        for i in range(1,n-1):
            y2[:,i]=y2[:,i]+r*y2[:,i-1]
            y3[:,i]=2.0*(x[:,i-1]-x[:,i+1])-r*s
            s=y1[:,i]
            r=s/y3[:,i]
        for j in range(1,n-1):
            i=n-1-j
            y2[:,i]=(y1[:,i]*y2[:,i+1]-y2[:,i])/y3[:,i]
        for i in range(n-1):
            s=y1[:,i]
            r=y2[:,i+1]-y2[:,i]
            y3[:,i]=r/s
            y2[:,i]=3.0*y2[:,i]
            y1[:,i]=(y[:,i+1]-y[:,i])/s-(y2[:,i]+r)*s
        self.x = x
        self.y0, self.y1, self.y2, self.y3 = y,y1,y2,y3
        self.dx = x[:,1]-x[:,0]
       
    def eval_dy(self, xs):
        # FIXME? Only support equi-distant
        xi = ((xs-self.x[0,0])/self.dx[0]).astype('int')
        xi = (xi>=self.n)*(self.n-1)+xi*(xi<self.n)
        dx = xs-self.x[0,xi]
        y = self.y1[:,xi] + 2*self.y2[:,xi]*dx + 3*self.y3[:,xi]*dx**2
        return y

    def eval_y_multi(self, xs, ind, d=0):
        xi = ((xs-self.x[ind,0])/self.dx[ind]).astype('int')
        # FIXME - this is how the fortran code processes it... seems like an error
        xi0cond = (xi<0)
        xilargecond = (xi>=self.n)
        xi = (xi>=0)*((xi>=self.n)*(self.n-1)+xi*(xi<self.n)).astype('int')
        d_code = ["*(y++) = *(y0+s) + *(y1+s)*dx + *(y2+s)*dx*dx + *(y3+s)*dx*dx*dx;",
                "*(dy++) = *(y1+s) + 2*(*(y2+s))*dx + 3*(*(y3+s))*dx*dx;",
                "*(d2y++) = 2*(*(y2+s)) + 6*(*(y3+s))*dx;",]
        y_arr = ['y', 'dy', 'd2y']
        import collections
        if isinstance(d, collections.Iterable):
            dcode = "".join([d_code[s] for s in d])
            ddict = True
        else:
            dcode = d_code[d]
            d = [d]
            ddict = False
        y = 0
        dy = 0
        d2y = 0
        if 0 in d:
            y = np.zeros_like(xs)
        if 1 in d:
            dy = np.zeros_like(xs)
        if 2 in d:
            d2y = np.zeros_like(xs)
        ddictlist = [y_arr[delem] for delem in d]
        code = """
            for(int k=0; k<N; k++) {
                int xk = *(xi++);
                int indk = *(ind++);
                int s = indk*n+xk;
                double dx = *(xs++)-*(x+s);
                //printf("%d %f %f %f\\n", xk, dx, *(y0+s), *(y1+s)*dx);
        """ + dcode + """
                dx++;
            }
        """
        y0,y1,y2,y3 = self.y0, self.y1, self.y2, self.y3
        n = y1.shape[1]
        N = len(xs)
        x = self.x
        W.inline(code, ['N', 'n', 'ind', 'xi', 'y0', 'y1', 'y2', 'y3', 'xs', 'x'] + ddictlist)
        #y = self.y1[zip(ind,xi)] + 2*self.y2[zip(ind,xi)]*dx + 3*self.y3[zip(ind,xi)]*dx**2
        if 0 in d:
            y[xi0cond.nonzero()] = 0
            y[xilargecond.nonzero()] = 0
        if 1 in d:
            dy[xi0cond.nonzero()] = 0
            dy[xilargecond.nonzero()] = 0
        if 2 in d:
            d2y[xi0cond.nonzero()] = 0
            d2y[xilargecond.nonzero()] = 0
        if ddict:
            results = [y, dy, d2y]
            return [results[delem] for delem in d]
        else:
            if 0 in d:
                return y
            if 1 in d:
                return dy
            if 2 in d:
                return d2y

    def eval_d2y(self, xs):
        xi = ((xs-self.x[0,0])/self.dx[0]).astype('int')
        xi = (xi>=self.n)*(self.n-1)+xi*(xi<self.n)
        dx = xs-self.x[0,xi]
        dy = 2*self.y2[:,xi] + 6*self.y3[:,xi]*dx
        return dy

    def eval_d2y(self, xs):
        # Must be very inaccurate...
        xi = ((xs-self.x[0,0])/self.dx[0]).astype('int')
        xi = (xi>=self.n)*(self.n-1)++xi*(xi<self.n)
        dx = xs-self.x[0,xi]
        d2y = 6*self.y2[:,xi]
        return d2y
