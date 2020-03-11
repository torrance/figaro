from __future__ import print_function, division

import argparse
import os

from astropy.cosmology import Planck15, z_at_value
from astropy.io import fits
from astropy.units import Quantity
from numba import njit, prange
import numpy as np


def main(args):
    print("Variation %s" % args.variation)
    print("FOV %d" % args.fov)

    boxsize_Mpc = 100
    N = 2400
    slicewidth = 0.05

    # Save redshift slices in 0.1 slices
    nslices = np.int(np.ceil(args.z_max / slicewidth))

    # Create catalogue image
    catalogue_image = np.zeros((
        nslices + 1,
        (args.fov * 3600) // args.catalogue_resolution,
        (args.fov * 3600) // args.catalogue_resolution,
    ))
    # ...and a halo image to compare visually
    halo_image = np.zeros((
        nslices + 1,
        (args.fov * 3600) // args.halo_resolution,
        (args.fov * 3600) // args.halo_resolution,
    ))
    halo_catalogue = np.zeros((0, 6))

    np.random.seed(args.variation)

    sim_dict = create_sim_dict()

    halos_dict = dict()
    halos_dict[0.025] = np.load("halosCM-z-0.025-m-0.004-c-0.035.npy")
    halos_dict[0.203] = np.load("halosCM-z-0.203-m-0.004-c-0.035.npy")
    halos_dict[0.309] = np.load("halosCM-z-0.309-m-0.004-c-0.035.npy")
    halos_dict[0.6]   = np.load("halosCM-z-0.6-m-0.004-c-0.035.npy")

    box_count = -1
    while True:
        box_count += 1

        # Break loop if we've exceeded z_max
        z = z_at_value(Planck15.comoving_distance, Quantity((box_count + 1) * boxsize_Mpc, "Mpc"))
        if z < args.z_min:
            continue
        if z > args.z_max:
            break

        z_snapshot = nearest_snapshot(z)
        print("Using snapshot:", z_snapshot)

        # Calculate ntiles here rather than in 25 Mpc loop, or else we might retile the 25 Mpc slices
        # differently
        fov_Mpc = np.radians(args.fov) * Planck15.comoving_transverse_distance(z).to_value("Mpc")
        ntiles = np.int(np.ceil(fov_Mpc / boxsize_Mpc))
        print("fov_Mpc:", fov_Mpc)
        print("ntiles:", ntiles)

        # Random offset
        x_offset, y_offset = np.random.randint(0, N, size=2)
        x_offset_Mpc, y_offset_Mpc = (x_offset / N) * boxsize_Mpc, (y_offset / N) * boxsize_Mpc
        print("x_offset:", x_offset, "x_offset_Mpc:", x_offset_Mpc)
        print("y_offset:", y_offset, "y_offset_Mpc:", y_offset_Mpc)

        # Process 25 Mpc slices for flux
        for slice_idx, offset_Mpc in enumerate([12.5, 37.5, 62.5, 87.5]):
            # if slice_idx != 0: continue ## REMOVE

            DC_Mpc = box_count * boxsize_Mpc + offset_Mpc  # radial comoving distance
            z = z_at_value(Planck15.comoving_distance, Quantity(DC_Mpc, "Mpc"))

            print("Redshift z", z, " DC_Mpc", DC_Mpc)

            lums_154MHz, alphas = sim_dict[z_snapshot][slice_idx]
            fluxes = (lums_154MHz * (1 + z)**(1 + alphas)) / (4 * np.pi * Planck15.luminosity_distance(z).to_value('m')**2)
            fluxes *= 1E26  # [W /m^2 /Hz] -> [Jy]

            # Apply offset
            fluxes = np.roll(fluxes, (y_offset, x_offset), (0, 1))

            if ntiles > 1:
                _fluxes = np.zeros((N * ntiles, N * ntiles))
                for nx in range(ntiles):
                    for ny in range(ntiles):
                        _fluxes[ny * N:(ny + 1) * N, nx * N:(nx + 1) * N] = fluxes

                fluxes = _fluxes

            print("Fluxes map has shape:", fluxes.shape)

            comoving_transverse_distance_1deg_Mpc = np.radians(1) * Planck15.comoving_transverse_distance(z).to_value("Mpc")
            angular_width_of_box = boxsize_Mpc / comoving_transverse_distance_1deg_Mpc
            print("Angular width of box: ", angular_width_of_box)
            fluxres = angular_width_of_box / N


            painter(catalogue_image[np.int(z / slicewidth) + 1], args.catalogue_resolution / 3600, fluxes, fluxres)
            painter(catalogue_image[0], args.catalogue_resolution / 3600, fluxes, fluxres)

        # Now process halos
        halos = np.copy(halos_dict[z_snapshot])  # [mass, x, y, z, something, other]

        # Offset each
        halos[:, 1] += x_offset_Mpc
        halos[:, 1] %= boxsize_Mpc
        halos[:, 2] += y_offset_Mpc
        halos[:, 2] %= boxsize_Mpc
        halos[:, 3] += box_count * boxsize_Mpc

        # Retile
        if ntiles > 1:
            nhalos = halos.shape[0]
            _halos = np.zeros((nhalos * ntiles**2, halos.shape[1]))

            noffset = 0
            for nx in range(ntiles):
                for ny in range(ntiles):
                    _halos[noffset:noffset + nhalos, :] = halos
                    _halos[noffset:noffset + nhalos, 1] += nx * boxsize_Mpc
                    _halos[noffset:noffset + nhalos, 2] += ny * boxsize_Mpc
                    noffset += nhalos

            assert(noffset == len(_halos))
            halos = _halos

        # Center box
        halos[:, 1] -= (ntiles * boxsize_Mpc) / 2
        halos[:, 2] -= (ntiles * boxsize_Mpc) / 2
        print("Min/max x (Mpc):", halos[:, 1].min(), halos[:, 1].max())
        print("Min/max y (Mpc):", halos[:, 2].min(), halos[:, 2].max())
        print("Min/max z (Mpc):", halos[:, 3].min(), halos[:, 3].max())

        # Halos must be at least 1 Mpc away (z_at_value breaks for extremely close values)
        halos = halos[halos[:, 3] > 1]

        # Round DC value to aid computation
        halos[:, 3] = np.around(halos[:, 3], decimals=1)

        # Calculate angular position
        for DC_Mpc in np.unique(halos[:, 3]):

            z = z_at_value(Planck15.comoving_distance, Quantity(DC_Mpc, "Mpc"))

            comoving_transverse_distance_1deg_Mpc = np.radians(1) * Planck15.comoving_transverse_distance(z).to_value("Mpc")
            idxs = halos[:, 3] == DC_Mpc
            halos[idxs, 3] = z
            halos[idxs, 1] /= comoving_transverse_distance_1deg_Mpc  # -> degrees
            halos[idxs, 2] /= comoving_transverse_distance_1deg_Mpc

            # Now use column 4 to put in a pseudoflux
            halos[idxs, 4] = 1 / Planck15.luminosity_distance(z).to_value('Mpc')**2

        print("Min/max x (deg):", halos[:, 1].min(), halos[:, 1].max())
        print("Min/max y (deg):", halos[:, 2].min(), halos[:, 2].max())

        # Filter out values out of the FOV
        idx = np.all([halos[:, 1] >= -args.fov/2, halos[:, 1] <= args.fov/2, halos[:, 2] >= -args.fov/2, halos[:, 2] <= args.fov/2], axis=0)
        halos = halos[idx]

        halopainter(halo_image[0], args.halo_resolution / 3600, halos)
        for i, z in enumerate(np.arange(0, args.z_max, slicewidth)):
            idxs = np.all([halos[:, 3] > z, halos[:, 3] < z + slicewidth], axis=0)
            if len(idxs):
                halopainter(halo_image[i + 1], args.halo_resolution / 3600, halos[idxs])

        halo_catalogue = np.concatenate([halo_catalogue, halos], axis=0)

    np.save("halos-%d.npy" % args.variation, halo_catalogue)
    halo_catalogue = halo_catalogue[np.argsort(halo_catalogue[:, 0])]  # Order by mass
    try:
        os.mkdir("cones-%d" % args.variation)
    except OSError:
        pass
    zs = [0.01, 0.02] + list(np.arange(0.05, 1.06, 0.05))
    for i, z in enumerate(zs[:-1]):
        z_min = (zs[i - 1] + z) / 2
        z_max = (zs[i + 1] + z) / 2

        # Special case for z = 0.01
        if i == 0:
            z_min = 0

        idx = np.all([halo_catalogue[:, 3] >= z_min, halo_catalogue[:, 3] < z_max], axis=0)
        # [mass, redshift, latitude, longitude]
        np.savetxt("cones-%d/cone_5X5_z%.02f.txt_sort" % (args.variation, z), halo_catalogue[idx][:, [0, 3, 1, 2]], header="mass redshift latitude longitude")

    for data, name, res in [(catalogue_image, 'web', args.catalogue_resolution), (halo_image, 'myhalos', args.halo_resolution)]:
        hdu = fits.PrimaryHDU(data=data)
        hdu.header["BUNIT"] = "JY/PIXEL"
        hdu.header["CTYPE1"] = "RA---SIN"
        hdu.header["CRPIX1"] = 0
        hdu.header["CRVAL1"] = 0
        hdu.header["CDELT1"] = -res / 3600
        hdu.header["CUNIT1"] = "deg"
        hdu.header["CTYPE2"] = "DEC--SIN"
        hdu.header["CRPIX2"] = 0
        hdu.header["CRVAL2"] = 0
        hdu.header["CDELT2"] = res / 3600
        hdu.header["CUNIT2"] = "deg"
        hdu.writeto("%s-%d.fits" % (name, args.variation), overwrite=True)


@njit(parallel=True)
def painter(catalogue, catres, fluxes, fluxres):
    for y in prange(fluxes.shape[0]):
        for x in range(fluxes.shape[1]):
            # Offset wrt center of each image
            xstart_Mpc = (x - fluxes.shape[1] / 2) * fluxres
            xend_Mpc = (x + 1 - fluxes.shape[1] / 2) * fluxres
            ystart_Mpc = (y - fluxes.shape[0] / 2) * fluxres
            yend_Mpc = (y + 1 - fluxes.shape[0] / 2) * fluxres

            x_start = np.int(xstart_Mpc / catres + catalogue.shape[1] / 2)
            x_end = np.int(xend_Mpc / catres + catalogue.shape[1] / 2)
            y_start = np.int(ystart_Mpc / catres + catalogue.shape[0] / 2)
            y_end = np.int(yend_Mpc / catres + catalogue.shape[0] / 2)

            if 0 <= y_start <= catalogue.shape[0] and 0 <= x_start <= catalogue.shape[1]:
                x_end = min(x_end, catalogue.shape[1])
                y_end = min(y_end, catalogue.shape[0])

                N = catalogue[y_start:y_end, x_start:x_end].size
                catalogue[y_start:y_end, x_start:x_end] += fluxes[y, x] / N


@njit(parallel=True)
def halopainter(halo_image, res, halos):
    for i in prange(halos.shape[0]):
        _, x_deg, y_deg, _, pseudoflux, _ = halos[i]
        x_pix = np.int(x_deg / res + halo_image.shape[1] / 2)
        y_pix = np.int(y_deg / res + halo_image.shape[0] / 2)
        if 0 <= x_pix < halo_image.shape[0] and 0 <= y_pix < halo_image.shape[1]:
            halo_image[y_pix, x_pix] += pseudoflux



def nearest_snapshot(z):
    dist = 999
    z_nearest = 0
    for z_snap in [0.025, 0.203, 0.309, 0.6]:
        if abs(z_snap - z) < dist:
            z_nearest = z_snap
            dist = abs(z_snap - z)

    return z_nearest



def create_sim_dict():
    sim_dict = dict()
    for snapshot, z in [(188, 0.025), (166, 0.203), (156, 0.309), (124, 0.6)]:
        sim = np.zeros((4, 2, 2400, 2400))  # [4 x 25 Mpc slices, [lums, alphas], 2400 x 2400 cells]

        for i, suffix in enumerate(["A", "B", "C", "D"]):
            hdu = fits.open("../sk_model_100Mpc_zbins/map_radioD_%d_2400_ZnewHB_%s.fits" % (snapshot, suffix))[0]
            lums_1400MHz = hdu.data[5] * 1E-7  # [ergs /s /Hz] -> [Watts /Hz]
            machs = hdu.data[3]

            # Sanitise data
            machs[~np.isfinite(machs)] = 0
            lums_1400MHz[~np.isfinite(lums_1400MHz)] = 0

            # Set limits for machs
            machs[machs < 2] = 2
            machs[machs > 10] = 10

            # Scale luminosity down to 154 Mhz
            v_in = 1400
            v_out = 154
            alphas = machs**2 / (1 - machs**2)
            lums_154Mhz = lums_1400MHz * (v_out / v_in)**alphas

            sim[i, 0] = lums_154Mhz
            sim[i, 1] = alphas

        sim_dict[z] = sim

    return sim_dict





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variation", type=int, required=True)
    parser.add_argument("--fov", type=float, default=4, help="Degrees")
    parser.add_argument("--z_min", type=float, default=0)
    parser.add_argument("--z_max", type=float, default=0.8)
    parser.add_argument("--catalogue-resolution", type=float, default=1, help="Arcseconds")
    parser.add_argument("--halo-resolution", type=float, default=20, help="Arcseconds")
    args = parser.parse_args()
    main(args)