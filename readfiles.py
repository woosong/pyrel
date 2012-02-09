import numpy as np
import re

def readelems(file):
    elems = re.split('\s+',file.readline())
    try:
        while True:
            elems.remove('')
    except ValueError, ve:
        pass
    return elems

def asis(x):
    return x
fmt_dict = {'d': int, 'f': float, 's': asis}
def elems_fmt(elems, fmt):
    return [fmt_dict[f](e) for f, e in zip(fmt, elems)]
       
def readelems_fmt(file, fmt):
    return elems_fmt(readelems(file), fmt)

def readxyz(xyzfile):
    file = open(xyzfile, 'rt')
    box = np.zeros((3,3))
    box[0] = readelems_fmt(file, 'fff')
    box[1] = readelems_fmt(file, 'fff')
    box[2] = readelems_fmt(file, 'fff')

    n = readelems_fmt(file, 'd')[0]
    coords = np.zeros((n,3), dtype='float')
    ityp = np.zeros(n, dtype='int')
    ityp2 = np.zeros(n, dtype='int')
    typename = []
    for i in range(n):
        x, y, z, ityp[i], ityp2[i], typnam = readelems_fmt(file, 'fffdds')
        coords[i] = [x,y,z]
        typename.append(typnam)
    return box, coords, ityp, ityp2, typename

def writexyz(xyzfile, box, coords, ityp, ityp2, typename):
    file = open(xyzfile, 'wt')
    file.write('%f\t%f\t%f\n' % tuple(box[0]))
    file.write('%f\t%f\t%f\n' % tuple(box[1]))
    file.write('%f\t%f\t%f\n' % tuple(box[2]))

    n = len(coords)
    file.write('%d\n' % n)
    for i in range(n):
        file.write('%.12f\t%.12f\t%.12f\t%d\t%d\t%s\n' % tuple(list(coords[i])+[ityp[i], ityp2[i], typename[i]]))
    file.close()

def fetchpairpot(file):
    # nvr : number of V(r) r points
    # scr : scale of r?
    # npp : number of pairs
    # sce : scale of e? 
    nvr, scr, npp, sce = readelems_fmt(file, 'dfdf')

    pairs = np.zeros((npp,2), dtype='int')
    pairs.flat[:] = readelems_fmt(file, 'dd'*npp)

    # read V
    rs = np.zeros(nvr, dtype='float')
    Vs = np.zeros((npp, nvr), dtype='float')
    for i in range(nvr):
        elems = readelems_fmt(file, 'f'*(npp+1))
        rs[i] = elems[0]
        Vs[:,i] = elems[1:]

    rs *= scr
    Vs *= sce

    return (nvr,scr,npp,sce,pairs), (rs,Vs)

def readpairpot(potfile):
    file = open(potfile, 'rt')
    return fetchpairpot(file)

def readeampot(eamfile):
    file = open(eamfile, 'rt')

    (nvr,scr,npp,sce,pairs), (rs,Vs) = fetchpairpot(file)

    elems = readelems(file)
    # number of element types
    mmele = int(elems[0])
    mele = np.array(elems_fmt(elems[1:], 'd'*mmele), dtype='int')
    # read rho
    er = np.zeros((mmele,nvr), dtype='float')
    for i in range(nvr):
        er[:,i] = readelems_fmt(file, 'f'+'f'*mmele)[1:]
   
    mmele2 = readelems_fmt(file, 'd')[0]
    assert mmele == mmele2
    io = np.zeros(mmele, dtype='int')
    rce0 = np.zeros(mmele)
    rce1 = np.zeros(mmele)
    nvre = np.zeros(mmele, dtype='int')
    dre = np.zeros(mmele)
    for i in range(mmele):
        io[i], rce0[i], rce1[i], nvre[i], dre[i] = readelems_fmt(file, 'dffdf')
    rce0 *= scr
    rce1 *= scr
    dre *= scr

    em = [np.zeros(nvre[i]) for i in range(mmele)]
    for i in range(mmele):
        for j in range(nvre[i]):
            em[i][j], dummy, num = readelems_fmt(file, 'fsd')
            assert num == io[i]
        em[i] *= sce

    return (nvr,scr,npp,sce,pairs), (rs,Vs,er), (mele,rce0,rce1,dre,em)

#print readxyz('test.xyz')
#print readpairpot('pp.tab')
#print readeampot('eam.tab')

