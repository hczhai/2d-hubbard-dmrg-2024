
def get_dmrg_energies(fns, n_sites, n_sweeps):
    sw_eners = [[] for _ in range(n_sweeps)]
    sw_dws = [[] for _ in range(n_sweeps)]
    sw_bdims = [0 for  _ in range(n_sweeps)]
    isw = -1
    for fn in fns:
        for l in open(fn, 'r').readlines():
            if l.startswith('Sweep ='):
                isw = int(l.split()[2])
                sw_bdims[isw] = int(l.split('Bond dimension = ')[1].split('|')[0].strip())
            elif l.startswith('Time elapsed'):
                isw = -1
            elif isw != -1 and (l.startswith(' --> Site =') or l.startswith(' <-- Site =')):
                if l.strip().split('..')[1] == '':
                    continue
                ee, dw = [float(x.strip()) for x in l.split('E =')[1].split('FLOPS')[0].split('Error =')]
                sw_eners[isw].append(ee)
                sw_dws[isw].append(dw)
    for x, y in zip(sw_eners, sw_dws):
        assert len(x) == n_sites - 1 and len(y) == n_sites - 1
    import numpy as np
    eners, dws, bds = np.array([min(x) for x in sw_eners]), np.array([max(x) for x in sw_dws]), sw_bdims
    return eners, dws, bds

def get_properties(fn_dm1, fn_corr, fn_nn, fn_dm1_rev, shape):
    import numpy as np
    hd = np.diag(1 - np.load(fn_dm1))
    hd = np.copy(hd.reshape(shape))
    hd[1::2, :] = hd[1::2, ::-1]

    hdr = np.diag(1 - np.load(fn_dm1_rev))
    hdr = np.copy(hdr.reshape(shape))
    hdr[1::2, :] = hdr[1::2, ::-1]

    cr = np.copy(np.load(fn_corr).reshape(shape + shape))
    cr[1::2, :] = cr[1::2, ::-1]
    cr[:, :, 1::2, :] = cr[:, :, 1::2, ::-1]

    nn = np.copy(np.load(fn_nn).reshape(shape + shape))
    nn[1::2, :] = nn[1::2, ::-1]
    nn[:, :, 1::2, :] = nn[:, :, 1::2, ::-1]

    return hd, hdr, cr, nn

def write_properties(hd, fn, title):
    if hd.ndim == 2:
        with open(fn, 'w') as f:
            f.write("%6s %6s %10s\n" % ("X", "Y", title))
            for x in range(hd.shape[0]):
                for y in range(hd.shape[1]):
                    f.write("%6d %6d %10.6f\n" % (x, y, hd[x, y]))
    else:
        with open(fn, 'w') as f:
            f.write("%6s %6s %6s %6s %10s\n" % ("X1", "Y1", "X2", "Y2", title))
            for x1 in range(hd.shape[0]):
                for y1 in range(hd.shape[1]):
                    for x2 in range(hd.shape[2]):
                        for y2 in range(hd.shape[3]):
                            f.write("%6d %6d %6d %6d %10.6f\n" % (x1, y1, x2, y2, hd[x1, y1, x2, y2]))

def plot_properties(hd, fn, title, ref=(0, 0)):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rc('font', size=12)
    plt.clf()
    plt.cla()
    if hd.ndim == 2:
        plt.matshow(hd.T, cmap='summer_r')
    else:
        plt.matshow(hd[ref[0], ref[1]].T, cmap='summer_r')
    if hd.shape == (16, 4):
        plt.colorbar(anchor=(-6.8, 0.5), shrink=0.8, aspect=15)
    elif hd.shape == (8, 8) or hd.shape == (8, ) * 4:
        plt.colorbar(anchor=(-11.4, 0.5), shrink=0.8, aspect=15)
    elif hd.shape[-1] == 6:
        plt.colorbar(anchor=(-7.4, 0.5), shrink=0.8, aspect=15)
    else:
        plt.colorbar(anchor=(-8.2, 0.5), shrink=0.8, aspect=15)
    plt.subplots_adjust(left=-5, right=1.5, top=1.0, bottom=0, wspace=0.0)
    plt.title(title, loc='left')
    plt.savefig(fn, dpi=600, bbox_inches='tight')

def rev_energy_extrapolation(eners, dws, bds, n_sites, xener, xdw, xbd, fn, titles):
    import scipy.stats
    import numpy as np
    reg = scipy.stats.linregress(dws, eners)
    emin, emax = min(eners), max(eners)
    eex, rex = reg.intercept, abs(reg.intercept - emin) / 5
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    import matplotlib
    matplotlib.rc('font', size=12)
    de = emax - emin
    x_reg = np.array([0, dws[-1] + dws[0]])
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1E}"))
    colors = ['#330A0D', "#9A1D26"]
    plt.clf()
    plt.cla()
    plt.grid(which='both', alpha=0.5)
    print(dws, eners)
    print(reg.intercept, reg.slope)
    plt.plot(x_reg, reg.intercept + reg.slope * x_reg, '--', linewidth=1, color=colors[0])
    plt.plot(dws, eners, ' ', marker='s', mfc='white', mec=colors[0], color=colors[0], markersize=5)
    plt.plot([xdw], [xener], ' ', marker='x', mfc='white', mec=colors[1], color=colors[1], markersize=6)
    plt.text(dws[0] * 0.025, emax + de * 0.0, "$E(D=\\infty) = %.6f \\pm %.6f$" %
        (reg.intercept, abs(reg.intercept - emin) / 5), color=colors[0], fontsize=13)
    plt.text(dws[0] * 0.025, emax - de * 0.2, "$R^2 = %.6f$" % (reg.rvalue ** 2),
        color=colors[0], fontsize=13)
    for it, tit in enumerate(titles):
        plt.text(dws[-1] + dws[0] * 0.075, reg.intercept - de * (0.0 - it * 0.2), tit,
            color=colors[0], horizontalalignment='right', fontsize=13)
    plt.xlim((0, dws[-1] + dws[0] / 10))
    plt.ylim((reg.intercept - de * 0.1, emax + de * 0.2))
    plt.xlabel("Largest discarded weight")
    plt.ylabel("Sweep energy per site")
    for dw, ee, bd in zip(dws, eners, bds):
        plt.text(dw - dws[0] * 0.05, ee, "$D=%d$" % bd, verticalalignment='center', horizontalalignment='right', fontsize=12, color=colors[0])
    plt.text(xdw - dws[0] * 0.05, xener, "$D=%d$" % xbd, verticalalignment='center', horizontalalignment='right', fontsize=12, color=colors[1])
    plt.subplots_adjust(left=0.17, bottom=0.1, right=0.95, top=0.95)
    plt.savefig(fn, dpi=600)
    return eex * n_sites, rex * n_sites

def analysis(dmrg_fns, rev_fns, prop_fns, n_rev_swps=16, shape=(16, 6), doping='1/16', fid=7, rev=True):
    import os
    n_sites = shape[0] * shape[1]
    n_elec = n_sites - n_sites // int(doping.split('/')[1])
    title = "$%s\\times %s\\ U=8\\ N_e = %s\\ $(%s doping) SU(2) DMRG" % (*shape, n_elec, doping)
    eners, dws, bds = get_dmrg_energies(dmrg_fns, n_sites, 88 if shape == (16, 4) else (14 if shape == (8, 8) else 10))
    xener, xdw, xbd = eners[-1], dws[-1], bds[-1]
    prefix = "%02d-%dx%d-N%d" % (fid, shape[1], shape[0], n_elec)
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    print('\n*** %s %s NE = %s PREFIX = %s ***' % (shape, doping, n_elec, prefix))
    print('D = %8d E = %13.8f (total) %12.8f (per-site) DW = %8.2e' % (xbd, xener, xener / n_sites, xdw))
    eners, dws, bds = get_dmrg_energies(rev_fns, n_sites, n_rev_swps)
    eners, dws, bds = eners[3::4], dws[3::4], bds[3::4]
    for ee, dw, bd in zip(eners, dws, bds):
        print('D = %8d E = %13.8f (total) %12.8f (per-site) DW = %8.2e' % (bd, ee, ee / n_sites, dw))
    if rev:
        eex, rex = rev_energy_extrapolation(eners / n_sites, dws, bds, n_sites, xener / n_sites, xdw, xbd,
            prefix + '/00-energy-extrapolation.png', titles=[title])
    else:
        eex, rex = 0, 0
    print('D = %8s E = %13.8f (+/- %10.8f, total)' % ("infinity", eex, rex))
    print('    %8s     %13.8f (+/- %10.8f, per-site)' % ("", eex / n_sites, rex / n_sites))
    with open(prefix + '/00-energies.txt', 'w') as f:
        f.write(title.replace('\\times ', 'x').replace('\\', '').replace('$', '') + '\n\n')
        f.write('D = %8d E = %13.8f (total) %12.8f (per-site) DW = %8.2e\n' % (xbd, xener, xener / n_sites, xdw))
        if rev:
            for ee, dw, bd in zip(eners, dws, bds):
                f.write('D = %8d E = %13.8f (total) %12.8f (per-site) DW = %8.2e\n' % (bd, ee, ee / n_sites, dw))
            f.write('D = %8s E = %13.8f (+/- %10.8f, total)\n' % ("infinity", eex, rex))
            f.write('    %8s     %13.8f (+/- %10.8f, per-site)\n' % ("", eex / n_sites, rex / n_sites))

    hd, hdr, cr, nn = get_properties(*prop_fns, shape=shape)
    ref = {(8, 8): (4, 4), (16, 6): (7, 3), (16, 8): (7, 4), (16, 4): (7, 2)}[shape]
    write_properties(hd, prefix + "/01-hole-density-D32k.txt", "hole density (D=32k)")
    write_properties(hdr, prefix + "/02-hole-density-D16k.txt", "hole density (D=16k)")
    write_properties(cr, prefix + "/03-spin-correlation-D32k.txt", "<S(X1,Y1).S(X2,Y2)> (D=32k)")
    write_properties(nn, prefix + "/04-charge-correlation-D32k.txt", "<N(X1,Y1).N(X2,Y2)> (D=32k)")
    plot_properties(hd, prefix + "/01-hole-density-D32k.png",
        title="$\\mathbf{Hole\\ Density}$ $%s\\times %s\\ (U=8, %s\\ \\mathrm{doping})$ SU(2) DMRG $D=32k$" % (*shape, doping))
    plot_properties(hdr, prefix + "/02-hole-density-D16k.png",
        title="$\\mathbf{Hole\\ Density}$ $%s\\times %s\\ (U=8, %s\\ \\mathrm{doping})$ SU(2) DMRG $D=16k$" % (*shape, doping))
    plot_properties(cr, prefix + "/03-spin-correlation-D32k.png", ref=ref,
        title="$\\mathbf{\\langle S_{%s,%s}\\cdot S_{x,y}\\rangle}$ $%s\\times %s\\ (U=8, %s\\ \\mathrm{doping})$ SU(2) DMRG $D=32k$" % (*ref, *shape, doping))
    plot_properties(nn, prefix + "/04-charge-correlation-D32k.png", ref=ref,
        title="$\\mathbf{\\langle N_{%s,%s}\\cdot N_{x,y}\\rangle}$ $%s\\times %s\\ (U=8, %s\\ \\mathrm{doping})$ SU(2) DMRG $D=32k$" % (*ref, *shape, doping))

def analysis_00():
    dmrg_fns = [
        '../00-4x16-N62/00-4x16-N62.out'
    ]
    rev_fns = [
        '../00-4x16-N62/00-4x16-N62-rev.out'
    ]
    prop_fns = [
        "../00-4x16-N62/00-1pdm.npy",
        "../00-4x16-N62/00-corr.npy",
        "../00-4x16-N62/00-nn.npy",
        "../00-4x16-N62/00-1pdm-rev.npy"
    ]
    analysis(dmrg_fns, rev_fns, prop_fns, shape=(16, 4), doping='1/32', fid=0)

def analysis_01():
    dmrg_fns = [
        '../01-4x16-N60/01-4x16-N60.out'
    ]
    rev_fns = [
        '../01-4x16-N60/01-4x16-N60-rev.out'
    ]
    prop_fns = [
        "../01-4x16-N60/01-1pdm.npy",
        "../01-4x16-N60/01-corr.npy",
        "../01-4x16-N60/01-nn.npy",
        "../01-4x16-N60/01-1pdm-rev.npy"
    ]
    analysis(dmrg_fns, rev_fns, prop_fns, shape=(16, 4), doping='1/16', fid=1)

def analysis_02():
    dmrg_fns = [
        '../02-8x8-N62/02-8x8-N62.out',
        '../02-8x8-N62/02-8x8-N62-re1.out'
    ]
    rev_fns = [
        '../02-8x8-N62/02-8x8-N62-rev.out',
        '../02-8x8-N62/02-8x8-N62-rev-re1.out',
        '../02-8x8-N62/02-8x8-N62-rev-re2.out'
    ]
    prop_fns = [
        "../02-8x8-N62/02-1pdm.npy",
        "../02-8x8-N62/02-corr.npy",
        "../02-8x8-N62/02-nn.npy",
        "../02-8x8-N62/02-1pdm-rev.npy"
    ]
    analysis(dmrg_fns, rev_fns, prop_fns, shape=(8, 8), doping='1/32', fid=2)

def analysis_03():
    dmrg_fns = [
        '../03-8x8-N56/03-8x8-N56.out',
        '../03-8x8-N56/03-8x8-N56-re1.out'
    ]
    rev_fns = [
        '../03-8x8-N56/03-8x8-N56-rev.out',
        '../03-8x8-N56/03-8x8-N56-rev-re1.out',
        '../03-8x8-N56/03-8x8-N56-rev-re2.out'
    ]
    prop_fns = [
        "../03-8x8-N56/03-1pdm.npy",
        "../03-8x8-N56/03-corr.npy",
        "../03-8x8-N56/03-nn.npy",
        "../03-8x8-N56/03-1pdm-rev.npy"
    ]
    analysis(dmrg_fns, rev_fns, prop_fns, shape=(8, 8), doping='1/8', fid=3)

def analysis_04():
    dmrg_fns = [
        '../04-4x16-N56/04-4x16-N56.out'
    ]
    rev_fns = [
        '../04-4x16-N56/04-4x16-N56-rev.out'
    ]
    prop_fns = [
        "../04-4x16-N56/04-1pdm.npy",
        "../04-4x16-N56/04-corr.npy",
        "../04-4x16-N56/04-nn.npy",
        "../04-4x16-N56/04-1pdm-rev.npy"
    ]
    analysis(dmrg_fns, rev_fns, prop_fns, shape=(16, 4), doping='1/8', fid=4)

def analysis_05():
    dmrg_fns = [
        '../05-8x16-N120/05-8x16-N120-x2.out',
        '../05-8x16-N120/05-8x16-N120-x2-re1.out'
    ]
    rev_fns = [
        '../05-8x16-N120/05-8x16-N120-x2-rev.out',
        '../05-8x16-N120/05-8x16-N120-x2-rev-re1.out'
    ]
    prop_fns = [
        "../05-8x16-N120/05-1pdm-x2.npy",
        "../05-8x16-N120/05-x2-corr.npy",
        "../05-8x16-N120/05-x2-nn.npy",
        "../05-8x16-N120/05-1pdm-x2-rev.npy"
    ]
    analysis(dmrg_fns, rev_fns, prop_fns, shape=(16, 8), doping='1/16', fid=5, rev=True)

def analysis_06():
    dmrg_fns = [
        '../06-8x16-N112/06-8x16-N112-x1.out',
        '../06-8x16-N112/06-8x16-N112-x1-re1.out',
        '../06-8x16-N112/06-8x16-N112-x1-re2.out'
    ]
    rev_fns = [
        '../06-8x16-N112/06-8x16-N112-x1-rev.out',
        '../06-8x16-N112/06-8x16-N112-x1-rev-re1.out',
        '../06-8x16-N112/06-8x16-N112-x1-rev-re2.out',
        '../06-8x16-N112/06-8x16-N112-x1-rev-re3.out',
        '../06-8x16-N112/06-8x16-N112-x1-rev-re4.out'
    ]
    prop_fns = [
        "../06-8x16-N112/06-1pdm-x1.npy",
        "../06-8x16-N112/06-x1-corr.npy",
        "../06-8x16-N112/06-x1-nn.npy",
        "../06-8x16-N112/06-1pdm-x1-rev.npy"
    ]
    analysis(dmrg_fns, rev_fns, prop_fns, shape=(16, 8), doping='1/8', fid=6, rev=True)

def analysis_07():
    dmrg_fns = [
        '../07-6x16-N90/07-6x16-N90-x1.out',
        '../07-6x16-N90/07-6x16-N90-x1-re1.out',
        '../07-6x16-N90/07-6x16-N90-x1-re2.out'
    ]
    rev_fns = [
        '../07-6x16-N90/07-6x16-N90-x1-rev.out',
        '../07-6x16-N90/07-6x16-N90-x1-rev-re1.out',
        '../07-6x16-N90/07-6x16-N90-x1-rev-re2.out'
    ]
    prop_fns = [
        "../07-6x16-N90/07-1pdm-x1.npy",
        "../07-6x16-N90/07-x1-corr.npy",
        "../07-6x16-N90/07-x1-nn.npy",
        "../07-6x16-N90/07-1pdm-x1-rev.npy"
    ]
    analysis(dmrg_fns, rev_fns, prop_fns, shape=(16, 6), doping='1/16', fid=7)

def analysis_08():
    dmrg_fns = [
        '../08-6x16-N84/08-6x16-N84-x1.out',
        '../08-6x16-N84/08-6x16-N84-x1-re1.out',
        '../08-6x16-N84/08-6x16-N84-x1-re2.out'
    ]
    rev_fns = [
        '../08-6x16-N84/08-6x16-N84-x1-rev.out',
        '../08-6x16-N84/08-6x16-N84-x1-rev-re1.out',
        '../08-6x16-N84/08-6x16-N84-x1-rev-re2.out'
    ]
    prop_fns = [
        "../08-6x16-N84/08-1pdm-x1.npy",
        "../08-6x16-N84/08-x1-corr.npy",
        "../08-6x16-N84/08-x1-nn.npy",
        "../08-6x16-N84/08-1pdm-x1-rev.npy"
    ]
    analysis(dmrg_fns, rev_fns, prop_fns, shape=(16, 6), doping='1/8', fid=8)


if __name__ == "__main__":
    analysis_00()
    analysis_01()
    analysis_02()
    analysis_03()
    analysis_04()
    analysis_05()
    analysis_06()
    analysis_07()
    analysis_08()
    
