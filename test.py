from eam import *

prefix = '/a/woosong/cacd/3x3x3/tfilled/bbond_fixtet_relax/'

model = eam_model('eam.tab', 7.0, 0.9)
for bg in range(1,13):
    for i in range(1,13):
        for j in range(1,13):
            path = prefix + '%d/%d-%d/' % (bg,i,j)
            
            box, coords, ityp, ityp2, typnam = readfiles.readxyz(path + 'xyz.rel')
            # dirty change
            model.set_atoms(box, coords, ityp)
            me = model.get_energy()

            e = float("".join(open(path + 'energy', 'rt').readlines()))

            print me-e
