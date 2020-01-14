from __future__ import print_function, division

import argparse

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
    cellresolution_Mpc = boxsize_Mpc / N
    slicewidth = 0.1

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

    np.random.seed(args.variation)

    sim_dict = create_sim_dict()

    halos_dict = dict()
    for snapshot, z in [(188, 0.025)]:
        halos = np.loadtxt("halos/gal_cat_radio188.dat")
        halos = halos[np.argsort(halos[:, 3])]

        # # REMOVE
        # for DC in [12.5, 37.5, 62.5, 87.5]:
        #     halos = np.concatenate([halos, [[0, 0, i, DC, 0, 0] for i in np.arange(100, step=0.01)]], axis=0)
        #     halos = np.concatenate([halos, [[0, 99.9, i, DC, 0, 0] for i in np.arange(100, step=0.01)]], axis=0)
        #     halos = np.concatenate([halos, [[0, i, 0, DC, 0, 0] for i in np.arange(100, step=0.01)]], axis=0)
        #     halos = np.concatenate([halos, [[0, i, 99.9, DC, 0, 0] for i in np.arange(100, step=0.01)]], axis=0)

        halos_dict[z] = halos

    box_count = -1
    while True:
        box_count += 1

        # Break loop if we've exceeded z_max
        z = z_at_value(Planck15.comoving_distance, Quantity((box_count + 1) * boxsize_Mpc, "Mpc"))
        if z > args.z_max:
            break

        # # # REMOVE
        # if box_count < 25:
        #     continue
        # if box_count > 25:
        #     break

        # Calculate ntiles here rather than in 25 Mpc loop, or else we might retile the 25 Mpc slices
        # differently
        fov_Mpc = np.radians(args.fov) * Planck15.comoving_transverse_distance(z).to_value("Mpc")
        ntiles = np.int(np.ceil(fov_Mpc / boxsize_Mpc))

        # Random offset
        x_offset, y_offset = np.random.randint(0, N, size=2)
        x_offset_Mpc, y_offset_Mpc = (x_offset / N) * boxsize_Mpc, (y_offset / N) * boxsize_Mpc

        # Process 25 Mpc slices for flux
        for slice_idx, offset_Mpc in enumerate([12.5, 37.5, 62.5, 87.5]):
            # if slice_idx != 0: continue ## REMOVE

            DC_Mpc = box_count * boxsize_Mpc + offset_Mpc  # radial comoving distance
            z = z_at_value(Planck15.comoving_distance, Quantity(DC_Mpc, "Mpc"))

            print("Redshift z", z, " DC_Mpc", DC_Mpc)

            # TODO: get nearest redshift
            lums_154MHz, alphas = sim_dict[0.025][slice_idx]
            fluxes = (lums_154MHz * (1 + z)**(1 + alphas)) / (4 * np.pi * Planck15.luminosity_distance(z).to_value('m')**2)
            fluxes *= 1E26  # [W /m^2 /Hz] -> [Jy]
            # # REMOVE
            # fluxes[:, 0:5] = 1
            # fluxes[:, -5:-1] = 1
            # fluxes[0:5, :] = 1
            # fluxes[-5:-1, :] = 1

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
            catalogue_image[0] += catalogue_image[np.int(z / slicewidth) + 1]

        # Now process halos
        # TODO: get nearest redshift
        halos = np.copy(halos_dict[0.025])  # [mass, x, y, z, something, other]
        # halos = halos[np.all([halos[:, 3] >= 0, halos[:, 3] < 25], axis=0)] # REMOVE

        # Offset each
        halos[:, 1] += x_offset_Mpc
        halos[:, 1] %= 100
        halos[:, 2] += y_offset_Mpc
        halos[:, 2] %= 100
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

            halos = _halos

        # Center box
        halos[:, 1] -= (ntiles * boxsize_Mpc) / 2
        halos[:, 2] -= (ntiles * boxsize_Mpc) / 2

        # Remove z = 0 values
        halos = halos[halos[:, 3] != 0]

        # Calculate angular position
        for DC_Mpc in np.unique(halos[:, 3]):
            z = z_at_value(Planck15.comoving_distance, Quantity(DC_Mpc, "Mpc"))
            comoving_transverse_distance_1deg_Mpc = np.radians(1) * Planck15.comoving_transverse_distance(z).to_value("Mpc")
            idxs = halos[:, 3] == DC_Mpc
            halos[idxs, 0] = z
            halos[idxs, 1] /= comoving_transverse_distance_1deg_Mpc  # -> degrees
            halos[idxs, 2] /= comoving_transverse_distance_1deg_Mpc

            # Now use column 4 to put in a pseudoflux
            halos[idxs, 4] = 1 / Planck15.luminosity_distance(z).to_value('Mpc')**2

        # Filter out values out of the FOV
        idx = np.all([halos[:, 1] >= -args.fov/2, halos[:, 1] <= args.fov/2, halos[:, 2] >= -args.fov/2, halos[:, 2] <= args.fov/2], axis=0)
        halos = halos[idx]

        for i, z in enumerate(np.arange(0, args.z_max, slicewidth)):
            idxs = np.all([halos[:, 0] > z, halos[:, 0] < z + slicewidth], axis=0)
            if len(idxs):
                halopainter(halo_image[i + 1], args.halo_resolution / 3600, halos[idxs])
                halo_image[0] += halo_image[i + 1]


    for data, name, res in [(catalogue_image, 'web', args.catalogue_resolution), (halo_image, 'halos', args.halo_resolution)]:
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
        if x_pix < halo_image.shape[0] and y_pix < halo_image.shape[1]:
            halo_image[y_pix, x_pix] += pseudoflux







def create_sim_dict():
    sim_dict = dict()
    for snapshot, z in [(188, 0.025)]:
        sim = np.zeros((4, 2, 2400, 2400))  # [4 x 25 Mpc slices, [lums, alphas], 2400 x 2400 cells]

        for i, suffix in enumerate(["A", "B", "C", "D"]):
            hdu = fits.open("first_100Mpc/map_radioD_%d_2400_ZnewHB_%s.fits" % (snapshot, suffix))[0]
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
            alphas = (0.5 * (machs**2 + 1) / (machs**2 - 1) + 0.5) * -1  # Times by -1 to match convention s**alpha not s**-alpha
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