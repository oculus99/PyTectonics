
# world_sim.py
import numpy as np
import random
from math import pi, sqrt, sin, cos, asin, atan2, log
from scipy.constants import golden
from scipy.spatial import cKDTree, ConvexHull
from numba import njit, prange
import matplotlib.pyplot as plt
import pyvista as pv
from plyfile import PlyData, PlyElement

# ==================== VAKIOT ====================
phi = golden
sqrt5 = sqrt(5)

# ==================== APUFUNKTIOT (Python) ====================
def bound(x, floor, ceil):
    return max(min(x, ceil), floor)

def toCartesian(spherical):
    lat, lon = spherical
    return np.array([
        cos(lat) * cos(lon),
        sin(lat),
        -cos(lat) * sin(lon)
    ], dtype=np.float32)

def toSpherical(cartesian):
    y = cartesian[1]
    z = cartesian[2]
    x = cartesian[0]
    return asin(bound(y, -1.0, 1.0)), atan2(-z, x)

# ==================== FIBGRID (Python, precompute neighbors) ====================
class FibGrid:
    def __init__(self, avgDistance):
        self.avgDistance = avgDistance
        self.pointNum = self.getPointNum(avgDistance)
        self.totalPointNum = 2 * self.pointNum + 1
        self.zIncrement = 2.0 / self.totalPointNum

        # points: dictionary index -> cartesian (float32)
        self.points = {}
        for i in range(-self.pointNum, self.pointNum + 1):
            self.points[i] = toCartesian(self._getSpherical(i))
        # neighbor list precomputed as a dict
        self.neighbor_dict = {i: self.getCellNeighborIds(i) for i in self.points.keys()}

    @staticmethod
    def getPointNum(avgAngleDistance):
        faceArea = sqrt(3) / 4 * avgAngleDistance**2
        sphereArea = 4 * pi
        totalPointNum = sphereArea / faceArea
        pointNum = int((totalPointNum - 1) / 2)
        return pointNum

    def _getZ(self, index):
        return bound(index * self.zIncrement, -1.0, 1.0)

    def _getLon(self, index):
        return (index * 2 * pi / phi) % (2 * pi)

    def _getSpherical(self, index):
        index = min(index, self.pointNum) if index > 0 else max(index, -self.pointNum)
        return (asin(self._getZ(index)), self._getLon(index))

    def getCellNeighborIds(self, index):
        lat, lon = self._getSpherical(index)
        zone = int(round(log(self.totalPointNum * pi * sqrt5 * cos(lat)**2, phi) / 2.0)) if cos(lat) > 0 else 0

        neighbors = []
        for sign_val in [-1, 1]:
            for zone_offset in [-1, 0, 1]:
                z = zone + zone_offset
                fib_val = fib(abs(z)) if z != 0 else 1
                neighbor_idx = index + sign_val * fib_val
                if -self.pointNum <= neighbor_idx <= self.pointNum:
                    neighbors.append(neighbor_idx)
        return list(set(neighbors))

def fib(n):
    # small safe version (n could be up to grid size)
    return int(round(phi**n / sqrt5)) if n >= 0 else int(round(phi**(-n) / sqrt5))

# ==================== NUMBA-OPTIMOIDUT FUNKTIOT ====================
@njit
def njit_normalize3(v):
    x, y, z = v[0], v[1], v[2]
    mag = (x*x + y*y + z*z) ** 0.5
    if mag == 0.0:
        return np.array([0.0,0.0,0.0], dtype=np.float32)
    return np.array([x/mag, y/mag, z/mag], dtype=np.float32)

@njit
def njit_rotate_vector(v, angle, axis):
    # Rodrigues' rotation formula (float32)
    ax = njit_normalize3(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    # cross
    cx = ax[1]*v[2] - ax[2]*v[1]
    cy = ax[2]*v[0] - ax[0]*v[2]
    cz = ax[0]*v[1] - ax[1]*v[0]
    dot = ax[0]*v[0] + ax[1]*v[1] + ax[2]*v[2]
    res0 = v[0]*cos_a + cx*sin_a + ax[0]*dot*(1-cos_a)
    res1 = v[1]*cos_a + cy*sin_a + ax[1]*dot*(1-cos_a)
    res2 = v[2]*cos_a + cz*sin_a + ax[2]*dot*(1-cos_a)
    return np.array([res0, res1, res2], dtype=np.float32)

@njit
def njit_moment_arm(point, axis):
    # returns magnitude of cross(point, axis_norm)
    ax = njit_normalize3(axis)
    cx = point[1]*ax[2] - point[2]*ax[1]
    cy = point[2]*ax[0] - point[0]*ax[2]
    cz = point[0]*ax[1] - point[1]*ax[0]
    return (cx*cx + cy*cy + cz*cz) ** 0.5




@njit(parallel=True)
def gradient_erosion(thickness, neighbors, neighbors_len, erosion_factor=0.05):
    """
    Eroosio, joka perustuu topografian gradienttiin.
    thickness: (N,) float32
    neighbors: (N, M) int32, -1 padded
    neighbors_len: int
    erosion_factor: float, 0..1
    """
    N = thickness.shape[0]
    new_thickness = thickness.copy()
    
    for i in prange(N):
        t_i = thickness[i]
        max_diff = 0.0
        for k in range(neighbors_len):
            nb = neighbors[i, k]
            if nb < 0:
                break
            diff = t_i - thickness[nb]
            if diff > max_diff:
                max_diff = diff
        # jyrkimmän gradientin mukaan päivitys
        new_thickness[i] = t_i - erosion_factor * max_diff
    
    # kopioi takaisin
    for i in prange(N):
        thickness[i] = new_thickness[i]




@njit(parallel=True)
def realistic_erosion(thickness, neighbors, neighbors_len, gradient_factor=0.05, smoothing_factor=0.02):
    """
    Yhdistetty eroosio:
      - gradient_factor: kuinka nopeasti jyrkät huiput tasoittuvat
      - smoothing_factor: kuinka paljon tasataan naapurien keskiarvon mukaan
    """
    N = thickness.shape[0]
    new_thickness = thickness.copy()
    
    for i in prange(N):
        t_i = thickness[i]
        max_diff = 0.0
        neighbor_sum = 0.0
        neighbor_count = 0
        
        for k in range(neighbors_len):
            nb = neighbors[i, k]
            if nb < 0:
                break
            diff = t_i - thickness[nb]
            if diff > max_diff:
                max_diff = diff
            neighbor_sum += thickness[nb]
            neighbor_count += 1
        
        # Gradient-pohjainen eroosio
        t_i -= gradient_factor * max_diff
        # Naapurien keskiarvoon perustuva tasoitus
        if neighbor_count > 0:
            t_i = (1.0 - smoothing_factor) * t_i + smoothing_factor * (neighbor_sum / neighbor_count)
        
        new_thickness[i] = t_i
    
    # Kopioi takaisin
    for i in prange(N):
        thickness[i] = new_thickness[i]

@njit(parallel=True)
def erosion_with_sea_level(thickness, neighbors, neighbors_len, sea_level,
                           gradient_factor=0.05, smoothing_factor=0.02):
    """
    Realistinen eroosio, huomioi merenpinnan: solmut < sea_level eivät kasva yli.
    """
    N = thickness.shape[0]
    new_thickness = thickness.copy()
    
    for i in prange(N):
        t_i = thickness[i]
        max_diff = 0.0
        neighbor_sum = 0.0
        neighbor_count = 0
        
        for k in range(neighbors_len):
            nb = neighbors[i, k]
            if nb < 0:
                break
            diff = t_i - thickness[nb]
            if diff > max_diff:
                max_diff = diff
            neighbor_sum += thickness[nb]
            neighbor_count += 1
        
        # Gradient-pohjainen eroosio
        t_i -= gradient_factor * max_diff
        
        # Naapurien keskiarvoon perustuva tasoitus
        if neighbor_count > 0:
            t_i = (1.0 - smoothing_factor) * t_i + smoothing_factor * (neighbor_sum / neighbor_count)
        
        # Merenpinnan rajoitus
        if t_i < sea_level:
            t_i = sea_level
        
        new_thickness[i] = t_i
    
    # Kopioi takaisin
    for i in prange(N):
        thickness[i] = new_thickness[i]




@njit(parallel=True)
def sediment_flow(thickness, neighbors, neighbors_len, sea_level,
                  flow_factor=0.1, min_slope=1.0):
    """
    Yksinkertainen sedimentin virtaus simulaatio.
    thickness: (N,) float32
    neighbors: (N,M) int32, -1 padding
    sea_level: float32, rajoittaa virtauksen
    flow_factor: float32, kuinka paljon siirtyy kerralla
    min_slope: float32, minimierotus virtausta varten
    """
    N = thickness.shape[0]
    # kerätään muutokset ensin, jotta ei aiheuta ketjureaktioita samalla kierroksella
    delta = np.zeros(N, dtype=np.float32)
    
    for i in prange(N):
        t_i = thickness[i]
        for k in range(neighbors_len):
            nb = neighbors[i, k]
            if nb < 0:
                break
            t_nb = thickness[nb]
            slope = t_i - t_nb
            if slope > min_slope:
                flow = slope * flow_factor
                delta[i] -= flow
                delta[nb] += flow
    
    # sovelletaan muutokset
    for i in prange(N):
        thickness[i] += delta[i]
        # rajoitetaan merenpinnan alapuolelle
        if thickness[i] < sea_level:
            thickness[i] = sea_level





from numba import njit, prange

@njit(parallel=True)
def erosion_diffusion(thickness, neighbors, neighbors_len, diffusion_factor=0.1):
    """
    Kevyt Laplacian-tyylinen diffuusio / eroosio crust-thicknessille.
    thickness: (N,) float32
    neighbors: (N, M) int32, -1 padded
    neighbors_len: int
    diffusion_factor: float, 0..1, pienempi = hitaampi diffuusio
    """
    N = thickness.shape[0]
    new_thickness = thickness.copy()
    
    for i in prange(N):
        t_sum = thickness[i]
        count = 1
        for k in range(neighbors_len):
            nb = neighbors[i, k]
            if nb < 0:
                break
            t_sum += thickness[nb]
            count += 1
        avg = t_sum / count
        # päivitys: siirrä nykyinen hieman kohti naapureiden keskiarvoa
        new_thickness[i] = thickness[i] + diffusion_factor * (avg - thickness[i])
    
    # kopioi takaisin alkuperäiseen taulukkoon
    for i in prange(N):
        thickness[i] = new_thickness[i]




@njit(parallel=True)
def gravity_relaxation(thickness, density, mantleDensity, neighbors, neighbors_len, relaxation_rate=0.05):
    """
    Yksinkertainen numba-optimoitu painovoima + eroosio step.
    - thickness: (N,) float32
    - density: (N,) float32
    - mantleDensity: float32
    - neighbors: (N, M) int32, naapurit (-1 padded)
    - neighbors_len: int
    - relaxation_rate: float32, kuinka nopeasti tasataan
    """
    N = thickness.shape[0]

    for i in prange(N):
        if thickness[i] <= 0:
            continue

        # laske ympäröivien solmujen keskiarvo
        neighbor_count = 0
        neighbor_sum = 0.0
        for k in range(neighbors_len):
            nb = neighbors[i, k]
            if nb < 0:
                break
            neighbor_sum += thickness[nb]
            neighbor_count += 1

        if neighbor_count == 0:
            continue

        avg_neighbor = neighbor_sum / neighbor_count

        # tavoitekorkeus isostasian kautta
        equilibrium_thick = mantleDensity / density[i] * avg_neighbor

        # paksuuden säätö
        thickness[i] -= (thickness[i] - equilibrium_thick) * relaxation_rate


@njit(parallel=True)
def isostacy_update(thickness, density, mantleDensity, displacement_out):
    # thickness, density: (N,)
    for i in prange(thickness.shape[0]):
        max_thickness = 20000.0 ## 40000
        thickness[i] = min(thickness[i], max_thickness)

        root = thickness[i] * density[i] / mantleDensity
        displacement_out[i] = thickness[i] - root

@njit
def simple_erupt(top_idx, thickness, density, world_continent_density):
    """Lisää top-crustin paksuutta ja päivittää tiheyden."""
    add = 0.1 * thickness[top_idx] + 100.0
    new_thick = thickness[top_idx] + add
    new_density = (density[top_idx]*thickness[top_idx] + world_continent_density*add) / new_thick
    thickness[top_idx] = new_thick
    density[top_idx] = new_density

@njit
def process_collision_logic(i, j,
                            positions,
                            thickness, density,
                            subductedBy, subducts, first_sub_by,
                            continent_mask,
                            world_radius, max_mountain_width,
                            world_continent_density):
    """
    Numba-ystävällinen yksittäinen törmäyslogiikka.
    Molempien mantereiden törmäys kasvattaa paksuutta.
    Muuten subduktiot / eruptiot.
    """
    # Määritä top ja bottom densiteetin mukaan
    if density[i] < density[j]:
        top = i
        bottom = j
    else:
        top = j
        bottom = i

    # Molemmat mantereita -> vuoristo
    if continent_mask[top] == 1 and continent_mask[bottom] == 1:
        uplift = 1000.0  # metriä, säädettävissä
        thickness[top] += uplift
        thickness[bottom] += uplift
    else:
        # bottom manneri, top ei -> eruptio
        if continent_mask[bottom] == 1 and continent_mask[top] == 0:
            simple_erupt(top, thickness, density, world_continent_density)
        else:
            # normaali subduction
            if first_sub_by[bottom] == -1:
                first_sub_by[bottom] = top
            fb = first_sub_by[bottom]
            dx = positions[bottom,0] - positions[fb,0]
            dy = positions[bottom,1] - positions[fb,1]
            dz = positions[bottom,2] - positions[fb,2]
            moved = (dx*dx + dy*dy + dz*dz)**0.5 * world_radius
            if moved > max_mountain_width:
                simple_erupt(top, thickness, density, world_continent_density)
                subductedBy[bottom] = top
                subducts[top] = -1
            else:
                subducts[top] = bottom
                subductedBy[bottom] = top

@njit(parallel=True)
def collide_core(indices,
                 positions,
                 thickness, density,
                 plate_id,
                 neighbors, neighbors_len,
                 subductedBy, subducts, first_sub_by,
                 continent_mask,
                 world_radius,
                 max_mountain_width,
                 world_continent_density,
                 collision_distance):
    """
    Numba-ystävällinen törmäyssilmukka.
    Tarkistaa annettujen indeksi-kandidaattien naapureita.
    """
    n_idx = indices.shape[0]
    for ii in prange(n_idx):
        i = indices[ii]
        if plate_id[i] < 0:
            continue
        for k in range(neighbors_len):
            j = neighbors[i, k]
            if j < 0:
                break
            if plate_id[j] < 0:
                continue
            if plate_id[i] == plate_id[j]:
                continue
            # euklidinen etäisyys yksikköpallon koordinaateissa
            dx = positions[i,0] - positions[j,0]
            dy = positions[i,1] - positions[j,1]
            dz = positions[i,2] - positions[j,2]
            dist2 = dx*dx + dy*dy + dz*dz
            if dist2 < collision_distance*collision_distance:
                process_collision_logic(i, j,
                                        positions,
                                        thickness, density,
                                        subductedBy, subducts, first_sub_by,
                                        continent_mask,
                                        world_radius, max_mountain_width,
                                        world_continent_density)

# ==================== NUMBA-OPTIMOIDUT SEED & RIFT ====================
# seed_plates_and_crusts_numba ja create_rifts_numba on integroitu NumPyWorldiin

@njit
def hash_noise(i):
    v = i * 374761393 + 668265263
    v = (v ^ (v >> 13)) * 1274126177
    v = v ^ (v >> 16)
    return (v & 0xffffffff) / 2147483648.0 - 1.0



@njit(parallel=True)
def seed_plates_and_crusts_numba(positions,
                                 plate_centers,
                                 continent_mask_indices,
                                 plate_id,
                                 thickness,
                                 density,
                                 is_continent,
                                 world_continent_density,
                                 world_continent_thickness,
                                 world_ocean_density,
                                 world_ocean_thickness,
                                 rng_seed_int=42):

    def lcg_rand(seed):
        a = 1664525
        c = 1013904223
        ns = (a * seed + c) & 0xFFFFFFFF
        return ns, (ns / 4294967296.0)

    seed = rng_seed_int

    N = positions.shape[0]
    P = plate_centers.shape[0]

    for i in prange(N):

        # ---- find nearest plate center ----
        px0 = positions[i,0]; px1 = positions[i,1]; px2 = positions[i,2]
        best = 0
        bd = (px0 - plate_centers[0,0])**2 + (px1 - plate_centers[0,1])**2 + (px2 - plate_centers[0,2])**2

        for c in range(1, P):
            d = (px0 - plate_centers[c,0])**2 + (px1 - plate_centers[c,1])**2 + (px2 - plate_centers[c,2])**2
            if d < bd:
                bd = d
                best = c

        plate_id[i] = best

        # ---- continent or ocean ----
        is_cont = continent_mask_indices[i]
        is_continent[i] = is_cont

        # ---- Box-Muller gaussian noise ----
        z0 = hash_noise(i) * 40.0

        if is_cont == 1:
            thickness[i] = world_continent_thickness + z0 * 40.0
            density[i]   = world_continent_density   + z0 * 40.0
        else:
            thickness[i] = world_ocean_thickness + z0 * 40.0
            density[i]   = world_ocean_density  + z0 * 40.0



@njit(parallel=True)
def create_rifts_numba(positions,    # (N,3) float32
                       plate_id,     # (N,) int32 (input/output)
                       neighbors,    # (N, M) int32 (neighbors as -1 padded)
                       neighbors_len,# int (M)
                       is_continent, # (N,) int8
                       thickness,    # (N,) float32
                       density,      # (N,) float32
                       world_radius, # float (meters)
                       minDistance,  # float (meters) threshold for safety
                       default_ocean_thickness,
                       default_ocean_density):
    """
    Finds neighbour slots that are currently -1 (vacant) and attempts to create new crust there
    if the nearest existing crust is farther than minDistance.
    Brute-force nearest implemented in numba (parallelized).
    """
    N = positions.shape[0]
    M = neighbors.shape[1]

    # precompute list of existing crust indices
    exists = np.empty(N, dtype=np.int32)
    ecount = 0
    for i in range(N):
        if plate_id[i] >= 0:
            exists[ecount] = i
            ecount += 1

    if ecount == 0:
        return

    # candidate mask: vacant cells with at least one neighbor that has a plate
    is_candidate = np.zeros(N, dtype=np.int8)
    for i in range(N):
        if plate_id[i] >= 0:
            continue
        for k in range(M):
            nb = neighbors[i,k]
            if nb < 0:
                break
            if plate_id[nb] >= 0:
                is_candidate[i] = 1
                break

    for i in prange(N):
        if is_candidate[i] == 0:
            continue
        # compute nearest existing crust distance
        px = positions[i,0]; py = positions[i,1]; pz = positions[i,2]
        min_d2 = 1.0e9
        for eidx in range(ecount):
            j = exists[eidx]
            dx = px - positions[j,0]; dy = py - positions[j,1]; dz = pz - positions[j,2]
            d2 = dx*dx + dy*dy + dz*dz
            if d2 < min_d2:
                min_d2 = d2
        chord = (min_d2 ** 0.5) * world_radius
        if chord >= minDistance:
            # find first neighbor plate id to assign
            assigned_plate = -1
            for k in range(M):
                nb = neighbors[i,k]
                if nb < 0:
                    break
                if plate_id[nb] >= 0:
                    assigned_plate = plate_id[nb]
                    break
            if assigned_plate >= 0:
                plate_id[i] = assigned_plate
                is_continent[i] = 0
                thickness[i] = default_ocean_thickness
                density[i] = default_ocean_density

# ==================== HIGH-LEVEL WRAPPER: maailma-taulukoiden rakentaminen (Python) ====================
class NumPyWorld:
    """
    Rakentaa NumPy-taulukot grid:stä ja initialisoi crust/plate data.
    Taulukko indeksi vastaa FibGrid:n avainta järjestyksessä sorted(keys).
    """
    def __init__(self, world_radius_m, grid: FibGrid, plates_count, seaLevel=3790):
        self.grid = grid
        keys = sorted(list(grid.points.keys()))
        self.keys = keys
        self.index_of_key = {k:i for i,k in enumerate(keys)}
        self.N = len(keys)
        self.positions = np.vstack([grid.points[k] for k in keys]).astype(np.float32)  # unit sphere positions
        # initialize arrays
        self.plate_id = np.full(self.N, -1, dtype=np.int32)   # -1 = no crust placed
        self.thickness = np.zeros(self.N, dtype=np.float32)
        self.density = np.zeros(self.N, dtype=np.float32)
        self.subductedBy = np.full(self.N, -1, dtype=np.int32)
        self.subducts = np.full(self.N, -1, dtype=np.int32)
        self.first_sub_by = np.full(self.N, -1, dtype=np.int32)
        self.displacement = np.zeros(self.N, dtype=np.float32)
        self.is_continent = np.zeros(self.N, dtype=np.int8)
        self.world_radius = float(world_radius_m)
        self.seaLevel = seaLevel
        # neighbor arrays: convert neighbor dict -> fixed-width numpy array
        max_n = max(len(v) for v in grid.neighbor_dict.values())
        neighbors = np.full((self.N, max_n), -1, dtype=np.int32)
        for i, k in enumerate(keys):
            nbs = grid.neighbor_dict[k]
            for j, nb in enumerate(nbs):
                neighbors[i,j] = self.index_of_key[nb]
        self.neighbors = neighbors
        self.neighbors_len = neighbors.shape[1]  # used by numba
        # plates metadata
        self.plates_count = plates_count
        # placeholder for plate colors etc in Python if needed

    def seed_plates_and_crusts(self, plate_centers_spherical, continentMaskKeys=None,
                                ocean_thickness=7100.0, ocean_density=2890.0,
                                continent_thickness=36900.0, continent_density=2700.0):
        """
        place initial crusts: choose nearest plate center for each grid point
        uses numba-accelerated seed function
        plate_centers_spherical: list of spherical coords (lat, lon)
        continentMaskKeys: set of keys that should become continents (optional)
        """
        centers_cart = np.vstack([toCartesian(s) for s in plate_centers_spherical]).astype(np.float32)
        # continent mask array per index
        cont_mask = np.zeros(self.N, dtype=np.int8)
        if continentMaskKeys is not None:
            for i, k in enumerate(self.keys):
                if k in continentMaskKeys:
                    cont_mask[i] = 1

        # call numba seeder
        seed_plates_and_crusts_numba(self.positions.astype(np.float32),
                                     centers_cart.astype(np.float32),
                                     cont_mask,
                                     self.plate_id,
                                     self.thickness,
                                     self.density,
                                     self.is_continent,
                                     continent_density,
                                     continent_thickness,
                                     ocean_density,
                                     ocean_thickness,
                                     rng_seed_int=123456)

        # initial displacement (isostacy)
        isostacy_update(self.thickness, self.density, World.mantleDensity, self.displacement)

    def build_crust_indices_list(self):
        # return numpy array of indices which have crust (plate_id >=0)
        mask = (self.plate_id >= 0)
        return np.where(mask)[0].astype(np.int32)


from numba import njit, prange
import numpy as np

# ==================== Numba-optimoitu laattapyöritys ====================
@njit(parallel=True)
def rotate_positions_by_plate(positions, plate_id, plate_omega, dt):
    """
    Rotoi jokaisen solmun aseman oman laatan pyörimisvektorin mukaan.
    positions: (N,3) float32, yksikköpallon koordinaatit
    plate_id: (N,) int32
    plate_omega: (P,3) float32, jokaiselle laatalle rotaatiovektori rad/s
    dt: float32, timestep sekunneissa
    """
    N = positions.shape[0]
    for i in prange(N):
        pid = plate_id[i]
        if pid < 0:
            continue
        omega = plate_omega[pid]
        angle = np.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2) * dt
        if angle == 0.0:
            continue
        axis = omega / np.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
        # Rodrigues rotation
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        v = positions[i]
        cx = axis[1]*v[2] - axis[2]*v[1]
        cy = axis[2]*v[0] - axis[0]*v[2]
        cz = axis[0]*v[1] - axis[1]*v[0]
        dot = axis[0]*v[0] + axis[1]*v[1] + axis[2]*v[2]
        positions[i,0] = v[0]*cos_a + cx*sin_a + axis[0]*dot*(1-cos_a)
        positions[i,1] = v[1]*cos_a + cy*sin_a + axis[1]*dot*(1-cos_a)
        positions[i,2] = v[2]*cos_a + cz*sin_a + axis[2]*dot*(1-cos_a)

from numba import njit, prange

@njit(parallel=True)
def erosion_step(thickness, neighbors, neighbors_len, erosion_rate=0.01):
    """
    Simple Laplace-like erosion:
      - thickness: (N,) float32
      - neighbors: (N, M) int32, neighbor indices or -1
      - neighbors_len: int (M)
      - erosion_rate: fraction of difference to neighbors removed per timestep
    """
    N = thickness.shape[0]
    delta = np.zeros(N, dtype=np.float32)

    for i in prange(N):
        t_i = thickness[i]
        for k in range(neighbors_len):
            j = neighbors[i, k]
            if j < 0:
                break
            t_j = thickness[j]
            diff = t_i - t_j
            if diff > 0:
                delta[i] -= erosion_rate * diff / neighbors_len
                delta[j] += erosion_rate * diff / neighbors_len

    for i in prange(N):
        thickness[i] += delta[i]
        if thickness[i] < 0.0:
            thickness[i] = 0.0


from numba import njit, prange
import numpy as np

# ==================== Numba-optimoitu laattapyöritys ====================
@njit(parallel=True)
def rotate_positions_by_plate(positions, plate_id, plate_omega, dt):
    """
    Rotoi jokaisen solmun aseman oman laatan pyörimisvektorin mukaan.
    positions: (N,3) float32, yksikköpallon koordinaatit
    plate_id: (N,) int32
    plate_omega: (P,3) float32, jokaiselle laatalle rotaatiovektori rad/s
    dt: float32, timestep sekunneissa
    """
    N = positions.shape[0]
    for i in prange(N):
        pid = plate_id[i]
        if pid < 0:
            continue
        omega = plate_omega[pid]
        angle = np.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2) * dt
        if angle == 0.0:
            continue
        axis = omega / np.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
        # Rodrigues rotation
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        v = positions[i]
        cx = axis[1]*v[2] - axis[2]*v[1]
        cy = axis[2]*v[0] - axis[0]*v[2]
        cz = axis[0]*v[1] - axis[1]*v[0]
        dot = axis[0]*v[0] + axis[1]*v[1] + axis[2]*v[2]
        positions[i,0] = v[0]*cos_a + cx*sin_a + axis[0]*dot*(1-cos_a)
        positions[i,1] = v[1]*cos_a + cy*sin_a + axis[1]*dot*(1-cos_a)
        positions[i,2] = v[2]*cos_a + cz*sin_a + axis[2]*dot*(1-cos_a)

# ==================== Numba-optimoitu kokonais-update ====================
def update_world_numba(world, dt=1.0):
    npw = world.npworld

    # 1️⃣ Rotoi pisteet laatan pyörimisvektorin mukaan
    rotate_positions_by_plate(npw.positions, npw.plate_id, world.plate_omega, dt)

    # 2️⃣ Isostacy
    isostacy_update(npw.thickness, npw.density, world.mantleDensity, npw.displacement)

    # ---- gravity + erosion step ----
    gravity_relaxation(npw.thickness, npw.density, World.mantleDensity, npw.neighbors, npw.neighbors_len, relaxation_rate=0.05)
    
    #diffusion_erosion(npw.thickness, npw.neighbors, npw.neighbors_len, diffusion_rate=0.02)
    erosion_diffusion(npw.thickness, npw.neighbors, npw.neighbors_len, diffusion_factor=0.1)
    # 3️⃣ Valitse collidable indices
    collidable_mask = np.zeros(npw.N, dtype=np.int8)
    for i in range(npw.N):
        if npw.plate_id[i] < 0:
            continue
        for nb in npw.neighbors[i]:
            if nb < 0:
                break
            if npw.plate_id[nb] != npw.plate_id[i]:
                collidable_mask[i] = 1
                break
    collidable_indices = np.where(collidable_mask == 1)[0].astype(np.int32)

    # 4️⃣ Collision core
    collision_distance = world.minDistance * 1.5
    collide_core(collidable_indices,
                 npw.positions,
                 npw.thickness, npw.density,
                 npw.plate_id,
                 npw.neighbors, npw.neighbors_len,
                 npw.subductedBy, npw.subducts, npw.first_sub_by,
                 npw.is_continent,
                 npw.world_radius,
                 world.maxMountainWidth,
                 world.continentCrustDensity,
                 collision_distance)

    # 5️⃣ Rift
    create_rifts_numba(npw.positions.astype(np.float32),
                       npw.plate_id,
                       npw.neighbors.astype(np.int32),
                       npw.neighbors_len,
                       npw.is_continent,
                       npw.thickness,
                       npw.density,
                       npw.world_radius,
                       world.minDistance,
                       7100.0,
                       world.oceanCrustDensity)
    
    #erosion_step(npw.thickness, npw.neighbors, npw.neighbors_len, erosion_rate=0.01)
    #gradient_erosion(npw.thickness, npw.neighbors, npw.neighbors_len, erosion_factor=0.05)
    # Käytetään merenpinnan arvoa World-luokasta
    #realistic_erosion(npw.thickness, npw.neighbors, npw.neighbors_len,
    #              gradient_factor=0.05, smoothing_factor=0.02)
    erosion_with_sea_level(npw.thickness, npw.neighbors, npw.neighbors_len,
                       sea_level=npw.seaLevel,
                       gradient_factor=0.05, smoothing_factor=0.02)
                       
    sediment_flow(npw.thickness, npw.neighbors, npw.neighbors_len,
              sea_level=npw.seaLevel,
              flow_factor=0.1, min_slope=1.0)
    # 6️⃣ Päivitä ikä
    world.age += dt



# ==================== WORLD (Python wrapper metadata) ====================
class World:
    mantleDensity = 3300.0
    waterDensity = 1026.0
    oceanCrustDensity = 2890.0
    continentCrustDensity = 2700.0

    def __init__(self, radius=6367, resolution=72, plateNum=5,
                 continentNum=3, continentSize=1250):
        self.age = 0.0
        self.radius = radius * 1000.0
        self.resolution = resolution
        avgPointDistance = 2 * pi / resolution
        self.grid = FibGrid(avgPointDistance)
        self.plate_omega = np.random.randn(plateNum, 3).astype(np.float32) * 1e-8
        # create simple random plate centers
        self.plate_centers = [self.randomPoint() for _ in range(plateNum)]

        # initialize numpy world
        self.npworld = NumPyWorld(self.radius, self.grid, plateNum, seaLevel=3790.0)

        # pick some continent shields
        shields = [toCartesian(self.randomPoint()) for _ in range(continentNum)]
        continentSizeRad = continentSize * 1000.0 / self.radius
        continent_keys = set()
        for i, k in enumerate(sorted(self.grid.points.keys())):
            cart = self.grid.points[k]
            if any(np.linalg.norm(cart - s) < continentSizeRad for s in shields):
                continent_keys.add(k)

        # seed crusts using numba seeder
        self.npworld.seed_plates_and_crusts(self.plate_centers, continentMaskKeys=continent_keys)
        # compute minDistance similar to prior code
        area = 4 * pi / self.grid.totalPointNum
        d = sqrt(area / sqrt(5))
        self.minDistance = sqrt(2) * d

        # mountain width parameter
        self.maxMountainWidth = 300000.0

    def randomPoint(self):
        return (asin(2 * random.random() - 1),
                2 * pi * random.random())

    def update(self, timestep=1.0):
        update_world_numba(self, timestep) 
        self.age += timestep



import pyvista as pv
import numpy as np

def build_mesh(npw, field="thickness"):
    verts = npw.positions.astype(np.float32)
    cells = []
    cell_types = []

    for i in range(npw.N):
        row = npw.neighbors[i]
        valid = row[row >= 0]
        nv = len(valid)
        if nv < 3:
            continue

        # sort neighbors around point i
        p = verts[i]
        nbrs = verts[valid]

        # local coordinate frame
        n = p / np.linalg.norm(p)
        if abs(n[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        t1 = np.cross(n, ref)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)

        dx = nbrs - p
        u = dx @ t1
        v = dx @ t2

        angles = np.arctan2(v, u)
        order = np.argsort(angles)
        sorted_valid = valid[order]

        # triangulate sorted ring
        for k in range(len(sorted_valid) - 1):
            cells.append([3, i, sorted_valid[k], sorted_valid[k+1]])
        # close the ring
        cells.append([3, i, sorted_valid[-1], sorted_valid[0]])
        cell_types.extend([pv.CellType.TRIANGLE] * len(sorted_valid))

    return verts, np.array(cells, dtype=np.int32), np.array(cell_types, dtype=np.uint8)


import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull

def visualize_world_pyvista(world, field="thickness", cmap="viridis"):
    """
    Visualisoi maailman karttapisteet PyVistalla.
    - world: World-luokan instanssi
    - field: pistekohtainen data, esim. "thickness" tai "density"
    """
    npw = world.npworld
    pts = npw.positions  # (N,3)

    # Luo convex hull pisteistä (trianguloitu pintamalli)
    hull = ConvexHull(pts)
    cells = np.hstack([np.full((len(hull.simplices),1),3), hull.simplices]).astype(np.int32)
    
    grid = pv.PolyData(pts, cells)

    # Pistekohtainen data (ei cell_data)
    scalars = getattr(npw, field)
    grid.point_data[field] = scalars.astype(np.float32)

    # PyVista-visualisointi
    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars=field, cmap=cmap, show_edges=False)
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()



# ==================== PIENTÄ TESTAUSTA / VISUALISOINNIN ALOITUS ====================
if __name__ == "__main__":
    import time
    print("Luo maailman sim (pieni resoluutio testausta varten)...")
    w = World(radius=6367, resolution=48, plateNum=6, continentNum=3, continentSize=1000)
    print("Pisteitä:", w.npworld.N)
    print("Ensimmäiset plate_id:t:", w.npworld.plate_id[:20])
    print("Aja 5 updatea ja tulosta tilastot")

    for step in range(25):
        t0 = time.time()
        w.update(timestep=1.0)
        t1 = time.time()
        print(f"Step {step+1}: age={w.age:.1f} duration={(t1-t0):.3f}s")
        print("  thickness stats: min %.1f max %.1f mean %.1f" %
              (w.npworld.thickness.min(), w.npworld.thickness.max(), w.npworld.thickness.mean()))
        print("  density stats: min %.1f max %.1f mean %.1f" %
              (w.npworld.density.min(), w.npworld.density.max(), w.npworld.density.mean()))

    print(w)
    visualize_world_pyvista(w, field="thickness")
    print("Valmista!")




