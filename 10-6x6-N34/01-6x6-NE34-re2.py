from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
import numpy as np, os, shutil

fid = 0

driver = DMRGDriver(scratch="/scratch/global/hczhai/h2d-34-su2-%02d" % fid,
    restart_dir="/scratch/global/hczhai/h2d-34-%02d-mps" % fid,
    symm_type=SymmetryTypes.SU2, stack_mem=100 << 30,
    n_threads=28, n_mkl_threads=4, mpi=True)

bond_dims = [24000] * 8
noises = [1E-7] * 4 + [0] * 4
thrds = [1E-6] * 8
n_sweeps = 8

nx, ny = 6, 6
n = nx * ny
u = 8
nelec = 34

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

driver.mpi.barrier()
restart_dir = "/scratch/global/hczhai/h2d-34-%02d-mps" % fid
if driver.mpi.rank == 0 and restart_dir != driver.scratch:
    for k in os.listdir(restart_dir):
        if '.KET.' in k or k == 'KET-mps_info.bin':
            shutil.copy(restart_dir + "/" + k, driver.scratch + "/" + k)
driver.mpi.barrier()

ket = driver.load_mps(tag='KET')
driver.mpi.barrier()
ket = driver.adjust_mps(ket, dot=2)[0]
driver.mpi.barrier()

energy = driver.dmrg(mpo, ket, n_sweeps=n_sweeps, bond_dims=bond_dims, noises=noises, thrds=thrds,
    lowmem_noise=True, twosite_to_onesite=None, tol=1E-12, cutoff=1E-24, iprint=2,
    dav_max_iter=50, dav_def_max_size=20, sweep_start=1, forward=False)
print('DMRG energy = %20.15f' % energy)

dm = driver.get_1pdm(ket, iprint=2, max_bond_dim=24000)
if driver.mpi.rank == 0:
    np.save("%02d-1pdm.npy" % fid, dm)

driver.mpi.barrier()
dm = driver.get_npdm(ket, pdm_type=2, npdm_expr='((C+D)2+(C+D)2)0', mask=(0, 0, 1, 1), iprint=2, max_bond_dim=24000)
dm = dm * (0.5 * -np.sqrt(3) / 2)
if driver.mpi.rank == 0:
    np.save("%02d-corr.npy" % fid, dm)

driver.mpi.barrier()
dm = driver.get_npdm(ket, pdm_type=2, npdm_expr='((C+D)0+(C+D)0)0', mask=(0, 0, 1, 1), iprint=2, max_bond_dim=24000)
dm = dm * 2
if driver.mpi.rank == 0:
    np.save("%02d-nn.npy" % fid, dm)
