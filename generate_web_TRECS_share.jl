using Cosmology: cosmology, luminosity_dist, angular_diameter_dist
using DelimitedFiles: readdlm, writedlm
using FITSIO
using NPZ: npzwrite
using Unitful
using UnitfulAstro
using Random: Random, rand


# Simulation parameters
const lbox = 100u"Mpc"       # 1D box size in Mpc
const n0 = 2400        # 1D box size in cells
const res = lbox / n0  # Spatial resolution
const nt = 15           # Maximum index of redshift slice
const fov = 2
const beam = 15
const nima = convert(Int64,trunc(fov * 3600 / beam))
const box_number = 1

ima_gal = zeros(nima, nima, nt)
catalogue = Array{Float64, 2}[]

# Set of available snapshots
snaps = ["188","166","156"]

# Set cosmology to Planck 2015 (https://arxiv.org/pdf/1502.01589.pdf)
# TODO: verify w0, wa and Tcmw
c = cosmology(
    h=0.6774,
    OmegaK=0.0008,
    OmegaM=0.3089,
    Neff=3.04,
    w0=-1,
    wa=0,
)

#...sequence of redshift snapshots as in TRECS catalog
zz = vcat([0.01, 0.02], range(0.05, stop=1, step=0.05), range(1.2, stop=10, step=0.2))

# Loop over redshift slices until zz[nt]
for ii in 1:nt - 1
    # We define the range of a redshift slice to be halfway between
    # between the last and next redshift slice value, with a special case for
    # the first redshift slice.
    if ii == 1
        z_min = 0
    else
        z_min = (zz[ii - 1] + zz[ii]) / 2
    end
    z_max = (zz[ii] + zz[ii+1]) / 2
    z_mean = (z_min + z_max) / 2

    println("Calculating flux contribution from redshift slice ", z_mean)

    # Calculate cosmology for this redshift
    ldist = luminosity_dist(c, z_mean)
    transverse_dist = deg2rad(fov) * angular_diameter_dist(c, z_mean)

    if transverse_dist > lbox
        println("Box is too large for field of view. Skipping...")
        continue
    end

    # ncloud determines the 'oversampling' of each pixel to avoid pixellation effects at a 15'' resolution
    if zz[ii] == 0.01
        ncloud = 18
    elseif zz[ii] == 0.02
        ncloud = 12
    elseif zz[ii] == 0.05
        ncloud = 6
    elseif zz[ii] == 0.1
        ncloud = 2
    else
        ncloud = 1  # Default case
    end

    # In principle, here we choose which snapshots slice and projection along the LOS to use in order
    # to minimize repeating structures. In this case however just one LOS and redshift, as example.
    if 0 <= zz[ii] <= 0.1
        los = "Y"
        filefits = string(pwd(), "/map_allD_", snaps[1], "_1024_", los, "newHB2.fits")
        pthr = 1e23   # Lower limit on pixels to use (i.e. pixels with Flux<pthr are not used) in [erg/s/Hz]
    elseif 0.1 < zz[ii] <= 0.2
        los = "Y"
        filefits = string(pwd(), "/map_allD_", snaps[1], "_1024_", los, "newHB2.fits")
        pthr = 1e23
    else
        los = "Y"
        filefits = string(pwd(), "/map_allD_", snaps[1], "_1024_", los, "newHB2.fits")
        pthr = 1e24
    end

    # Read in from the appropriate sky model with the cosmic web emission
    imaf = FITS(filefits, "r")
    ima = read(imaf[1])
    radio = ima[:, :, 9]  # Radio emission in axis 9 [erg/s/Hz]
    mach = ima[:, :, 5]   # Weighted distribution of Mach number in axis 5
    close(imaf)
    ima = nothing

    # Set limits for too small or too large Mach numbers
    mach[mach .< 2] .= 2
    mach[mach .> 10] .= 10

    # Adjust flux of radio output from 1400 MHz down to 154 MHz
    ν_in = 1400    # Input frequency in the sky model [MHz]
    ν_out = 154    # Desired output frequency  [MHz]
    α = @. 0.5 * (mach^2 + 1 ) / (mach^2 - 1)
    radio = @. radio * (ν_in / ν_out)^α

    # Restrict only to sources above threshold pthr
    # We also calculate the position as the resolution element * the cell coordinates
    idx = findall(radio .> pthr)    #  Find index of all pixels brighter than p_thr [erg/s/Hz]
    xs = map(x -> (x[1] - 1) * res, idx)
    ys = map(x -> (x[2] - 1) * res, idx)
    lum = radio[idx]

    # Convert luminosity to flux in Jy
    @views lum /= 1E7                         # [erg/s/Hz] -> [Watts/Hz]
    fluxes = lum / (4 * π * (ldist / u"m")^2) # [Watts / m^2 / Hz]
    @views fluxes *= 1E26                     # [Watts / m^2 / Hz] -> [Jy]

    # Randomly offset the flattened radio volume (wrap around)
    Random.seed!(ii) # Seeding with ii produces reproducible offsets
    shift1 = lbox * rand(2)
    xs = (xs .+ lbox * rand()) .% lbox
    ys = (ys .+ lbox * rand()) .% lbox

    for i in eachindex(fluxes)
        if xs[i] > transverse_dist || ys[i] > transverse_dist
            continue
        end

        # For low redshifts, we oversample cells to avoid pixellation effects
        flux = fluxes[i] / ncloud^2
        for ll in 1:ncloud
            for jj in 1:ncloud
                x = (xs[i] + ll * (res / ncloud)) / transverse_dist
                y = (ys[i] + jj * (res / ncloud)) / transverse_dist

                # Recheck that we're still inside the FOV - the cloud may have pushed us out
                if x > 1 || y > 1
                    continue
                end

                push!(catalogue, [deg2rad(x) deg2rad(y) 154E6 flux -1 z_mean])  # [RA, Dec, reference freq, flux, spectral index, redshift]

                xu = floor(Int, x * nima + 1)
                yu = floor(Int, y * nima + 1)
                ima_gal[xu, yu, ii] += flux
            end
        end
    end

    ima_gal[:, :, nt] += ima_gal[:, :, ii]
    println("Total flux in redshift slice: ", sum(@view ima_gal[:, :, ii]))
end

# Write catalogue out to numpy file
catalogue = vcat(catalogue...)
@views catalogue[:, 2] .-= deg2rad(27)  # Centre on EoR0 field at (0, -27)
npzwrite(string("web-", box_number, ".npy"), catalogue)

# Write temporary fits file for quick sanity check
filep4 = string(pwd(), "/map_web", box_number, ".fits")
f = FITS(filep4, "w");
write(f, ima_gal)
close(f)