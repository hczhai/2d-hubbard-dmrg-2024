
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
import numpy as np

fid = 1

driver = DMRGDriver(scratch="/central/scratch/hczhai/h2d-su2-%02d" % fid,
    restart_dir="/central/scratch/hczhai/crit/2d-19/%02d-mps-rev" % fid,
    symm_type=SymmetryTypes.SU2, stack_mem=100 << 30,
    n_threads=64, n_mkl_threads=8, mpi=True)

bond_dims = [11000] * 4 + [10000] * 4 + [9000] * 4 + [8000] * 4
noises = [0] * 16
thrds = [1E-7] * 16
n_sweeps = 16

nx, ny = 16, 4
n = nx * ny
u = 8
nelec = 60

print("NELEC = %d" % nelec)
print("U = %d" % u)

driver.initialize_system(n_sites=n, n_elec=nelec, spin=0, orb_sym=None)

b = driver.expr_builder()

f = lambda i, j: i * ny + j if i % 2 == 0 else i * ny + ny - 1 - j

for i in range(0, nx):
    for j in range(0, ny):
        if driver.mpi.rank == j % driver.mpi.size:
            if i + 1 < nx:
                b.add_term("(C+D)0", [f(i, j), f(i + 1, j), f(i + 1, j), f(i, j)], 2 ** 0.5)
            if j + 1 < ny:
                b.add_term("(C+D)0", [f(i, j), f(i, j + 1), f(i, j + 1), f(i, j)], 2 ** 0.5)
            b.add_term("((C+(C+D)0)1+D)0", [f(i, j), f(i, j), f(i, j), f(i, j)], u)

mpo = driver.get_mpo(b.finalize(adjust_order=True), algo_type=MPOAlgorithmTypes.FastBipartite, iprint=2)
ket = driver.load_mps(tag='KET')
driver.mpi.barrier()
ket = driver.adjust_mps(ket, dot=2)[0]
driver.mpi.barrier()
energy = driver.dmrg(mpo, ket, n_sweeps=n_sweeps, bond_dims=bond_dims, noises=noises, thrds=thrds,
    lowmem_noise=True, twosite_to_onesite=None, tol=1E-12, cutoff=1E-24, iprint=2,
    dav_max_iter=50, dav_def_max_size=20)
print('DMRG energy = %20.15f' % energy)

dm = driver.get_1pdm(ket, iprint=2, max_bond_dim=12000)
if driver.mpi.rank == 0:
    np.save("%02d-1pdm-rev.npy" % fid, dm)

driver.mpi.barrier()
dm = driver.get_npdm(ket, pdm_type=2, npdm_expr='((C+D)2+(C+D)2)0', mask=(0, 0, 1, 1), iprint=2, max_bond_dim=12000)
dm = dm * (0.5 * -np.sqrt(3) / 2)
if driver.mpi.rank == 0:
    np.save("%02d-corr-rev.npy" % fid, dm)

driver.mpi.barrier()
dm = driver.get_npdm(ket, pdm_type=2, npdm_expr='((C+D)0+(C+D)0)0', mask=(0, 0, 1, 1), iprint=2, max_bond_dim=12000)
dm = dm * 2
if driver.mpi.rank == 0:
    np.save("%02d-nn-rev.npy" % fid, dm)
