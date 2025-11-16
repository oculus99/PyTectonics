
# pytectonics_numba_full.py
"""
Täysi Numba-optimoitu ydinosio (C-tyyppinen data) yhdistettynä Python-UI/IO:hon.
- Grid / naapurit lasketaan Pythonissa (FibGrid)
- Crust- ja plate-attribuutit tallennetaan NumPy-taulukoihin
- Raskaat päivitys-operaatiot (isostacy, collide) ajetaan Numba:n avulla
- Rift-lisäykset tehdään kevyemmin Pythonilla (käyttäen KD-tree:ta)
HUOM: tämä on prototyyppi; logiikka on yksinkertaistettu ja tarkoitettu skaalattavaksi.
"""
import numpy as np
import random
from math import pi, sqrt, sin, cos, asin, atan2, log
from scipy.constants import golden
from scipy.spatial import cKDTree, ConvexHull
from numba import njit, prange
import matplotlib.pyplot as plt
import pyvista as pv
from plyfile import PlyData, PlyElement

import scipy
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

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
def isostacy_update(thickness, density, mantleDensity, displacement_out):
    # thickness, density: (N,)
    for i in prange(thickness.shape[0]):
        root = thickness[i] * density[i] / mantleDensity
        displacement_out[i] = thickness[i] - root

from numba import njit, prange
import numpy as np

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
        neighbors_len = np.zeros(max_n, dtype=np.int32)  # here we'll pass max_n as length
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
        plate_centers_spherical: list of spherical coords (lat, lon)
        continentMaskKeys: set of keys that should become continents (optional)
        """
        centers_cart = np.vstack([toCartesian(s) for s in plate_centers_spherical]).astype(np.float32)
        # for each grid point, find nearest center
        # brute-force: for N ~ few thousand it's fine; can be optimized with KD-tree if needed
        for i in range(self.N):
            p = self.positions[i]
            best = 0
            bd = np.linalg.norm(p - centers_cart[0])
            for c in range(1, centers_cart.shape[0]):
                d = np.linalg.norm(p - centers_cart[c])
                if d < bd:
                    bd = d; best = c
            # assign crust to plate 'best'
            self.plate_id[i] = best
            # continent mask
            is_cont = 0
            if continentMaskKeys is not None and self.keys[i] in continentMaskKeys:
                is_cont = 1
            self.is_continent[i] = is_cont
            if is_cont:
                self.thickness[i] = continent_thickness + random.gauss(0,40)
                self.density[i] = continent_density + random.gauss(0,40)
            else:
                self.thickness[i] = ocean_thickness + random.gauss(0,40)
                self.density[i] = ocean_density + random.gauss(0,40)
            # initial displacement
            root = self.thickness[i] * self.density[i] / World.mantleDensity
            self.displacement[i] = self.thickness[i] - root

    def build_crust_indices_list(self):
        # return numpy array of indices which have crust (plate_id >=0)
        mask = (self.plate_id >= 0)
        return np.where(mask)[0].astype(np.int32)

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

        self.npworld.seed_plates_and_crusts(self.plate_centers, continentMaskKeys=continent_keys)
        # compute minDistance similar to prior code
        area = 4 * pi / self.grid.totalPointNum
        d = sqrt(area / sqrt(5))
        self.minDistance = sqrt(2) * d

    def randomPoint(self):
        return (asin(2 * random.random() - 1),
                2 * pi * random.random())

    def update(self, timestep=1.0):
        """
        Full update:
          - move plates (lightweight: we rotate positions in python using numba rotate when possible)
          - isostacy update (numba)
          - collision (numba)
          - rift (python + KD-tree)
        """
        npw = self.npworld
        # ---- isostacy ----
        isostacy_update(npw.thickness, npw.density, World.mantleDensity, npw.displacement)
        # ---- prepare collidable indices: crusts with missing neighbors (edges) ----
        # simplified rule: any crust that has at least one neighbor cell empty (plate_id == -1) is "collidable"
        collidable_mask = np.zeros(npw.N, dtype=np.int8)
        # vectorized check: for each i, check neighbors
        for i in range(npw.N):
            if npw.plate_id[i] < 0:
                continue
            row = npw.neighbors[i]
            for nb in row:
                if nb < 0:
                    break
                if npw.plate_id[nb] != npw.plate_id[i]:
                    collidable_mask[i] = 1
                    break
        collidable_indices = np.where(collidable_mask == 1)[0].astype(np.int32)

        # ---- collisions (numba core) ----
        collision_distance = self.minDistance * 1.5
        collide_core(collidable_indices,
                     npw.positions,
                     npw.thickness, npw.density,
                     npw.plate_id,
                     npw.neighbors, npw.neighbors_len,
                     npw.subductedBy, npw.subducts, npw.first_sub_by,
                     npw.is_continent,
                     npw.world_radius,
                     self.maxMountainWidth if hasattr(self, 'maxMountainWidth') else 300000.0,
                     World.continentCrustDensity,
                     collision_distance)

        # ---- rift: create new crusts where neighbor slot empty and distance to other plates sufficient ----
        # We will use KD-tree of all existing crust positions (plate_id >=0)
        existing_positions = npw.positions[npw.plate_id >= 0]
        if existing_positions.shape[0] > 0:
            other_tree = cKDTree(existing_positions)
        else:
            other_tree = None

        rift_ids = []
        for i in range(npw.N):
            if npw.plate_id[i] < 0:
                continue
            # check neighbor slots - if there's a neighbor cell with plate_id == -1 -> candidate rift location
            for nb in npw.neighbors[i]:
                if nb < 0:
                    break
                if npw.plate_id[nb] == -1:
                    # candidate: ensure not too close to any existing crust from other plates
                    pos = npw.positions[nb]
                    if other_tree is None:
                        safe = True
                    else:
                        d, _ = other_tree.query(pos, k=1)
                        # convert chord distance to meters approx by * world_radius
                        safe = (d * npw.world_radius) >= self.minDistance
                    if safe:
                        rift_ids.append(nb)
        # Create new crusts at rift ids: assign to the plate of the neighbor that created it (approx)
        for rid in rift_ids:
            # find one neighbor plate id to assign
            assigned_plate = -1
            for nb in npw.neighbors[rid]:
                if nb < 0:
                    break
                if npw.plate_id[nb] >= 0:
                    assigned_plate = npw.plate_id[nb]
                    break
            if assigned_plate >= 0:
                npw.plate_id[rid] = assigned_plate
                npw.is_continent[rid] = 0
                npw.thickness[rid] = 7100.0 + random.gauss(0,40)
                npw.density[rid] = World.oceanCrustDensity + random.gauss(0,40)
                # displacement recomputed next step by isostacy

        self.age += timestep

# ==================== MESH BUILD / VISUALIZATION ====================
def build_planet_mesh_from_npworld(npw):
    # Build mesh using grid ordering (keys)
    verts = npw.positions * (npw.world_radius)  # scale by radius to get meters (approx)
    N = verts.shape[0]
    # elevation from displacement - seaLevel
    elevations = npw.displacement - npw.seaLevel
    displaced_scale = 1.0 + (elevations / (npw.world_radius * 0.02 + 1e-9))
    displaced_verts = verts * displaced_scale[:, np.newaxis]

    # Triangulate via ConvexHull (works for full sphere)
    hull = ConvexHull(displaced_verts)
    triangles = hull.simplices
    # fix winding
    tris_fixed = fix_triangle_winding(displaced_verts, triangles)

    # colors: plate-based simple palette (generate deterministic colors)
    plate_ids = npw.plate_id
    unique_plates = np.unique(plate_ids[plate_ids >= 0])
    palette = {}
    for p in unique_plates:
        # fixed color by seed
        random.seed(int(p))
        palette[p] = np.array([int(50 + 160*random.random()), int(50 + 160*random.random()), int(50 + 160*random.random())], dtype=np.uint8)
    colors = np.zeros((N,3), dtype=np.uint8)
    for i in range(N):
        pid = plate_ids[i]
        if pid >= 0:
            colors[i] = palette[pid]
        else:
            colors[i] = np.array([50,120,200], dtype=np.uint8)
    return displaced_verts, tris_fixed, colors, elevations

def fix_triangle_winding(verts, triangles):
    fixed = []
    for tri in triangles:
        p0 = verts[int(tri[0])]
        p1 = verts[int(tri[1])]
        p2 = verts[int(tri[2])]
        n = np.cross(p1 - p0, p2 - p0)
        centroid = (p0 + p1 + p2) / 3.0
        if np.dot(n, centroid) < 0:
            fixed.append([int(tri[0]), int(tri[2]), int(tri[1])])
        else:
            fixed.append([int(tri[0]), int(tri[1]), int(tri[2])])
    return np.array(fixed, dtype=np.int32)

def export_world_mesh_by_plate_npworld(npw, filename="planet_mesh_by_plate.ply"):
    verts, tris, colors, elevations = build_planet_mesh_from_npworld(npw)
    # faces and ply writing using plyfile
    n_faces = tris.shape[0]
    faces = []
    face_dtype = [('vertex_indices', 'i4', (3,)), ('red','u1'),('green','u1'),('blue','u1')]
    face_list = []
    min_elev = elevations.min()
    max_elev = elevations.max()
    for tri in tris:
        types = [1 if npw.is_continent[v] else 0 for v in tri]
        avg_type = round(sum(types)/3.0)
        avg_elev = (elevations[tri[0]] + elevations[tri[1]] + elevations[tri[2]]) / 3.0
        norm_elev = (avg_elev - min_elev) / (max_elev - min_elev + 1e-9)
        if avg_type == 1:
            r = int(150 + 105 * norm_elev)
            g = int(50 + 100 * (1 - norm_elev))
            b = int(50)
        else:
            r = int(50)
            g = int(50 + 50 * norm_elev)
            b = int(150 + 100 * (1 - norm_elev))
        face_list.append((list(tri), r, g, b))
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertices = np.empty(len(verts), dtype=vertex_dtype)
    vertices['x'] = verts[:,0].astype(np.float32)
    vertices['y'] = verts[:,1].astype(np.float32)
    vertices['z'] = verts[:,2].astype(np.float32)
    faces_array = np.array(face_list, dtype=face_dtype)
    el_verts = PlyElement.describe(vertices, 'vertex')
    el_faces = PlyElement.describe(faces_array, 'face')
    PlyData([el_verts, el_faces], text=True).write(filename)
    print(f"✓ Mesh tallennettu: {filename}")


# PyVista simple viewer
def visualize_npworld(npw, smooth=True):
    verts, tris, colors, elevations = build_planet_mesh_from_npworld(npw)
    n_faces = tris.shape[0]
    faces = np.hstack([np.full((n_faces,1), 3), tris]).astype(np.int64).flatten()
    mesh = pv.PolyData(verts, faces)
    mesh.point_data['RGB'] = colors
    mesh.compute_normals(cell_normals=False, auto_orient_normals=True, inplace=True)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='RGB', rgb=True, smooth_shading=smooth, show_edges=False)
    plotter.set_background('black')
    plotter.add_axes()
    plotter.show()

def world_to_2d_heightmap(world, resolution_lon=2048, resolution_lat=1024,
                          smoothing_sigma=2, filename="world_heightmap_gray.png"):
    """
    Luo saumattoman 2D-harmaaluvun (DEM) pallon pinnasta.
    Tukee sekä vanhaa World-iterointia että uutta NumPyWorld-mallia (world.npworld).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    # Hae pisteet ja korkeudet riippuen World-tyypistä
    if hasattr(world, "npworld"):
        npw = world.npworld
        verts = npw.positions.copy()            # (N,3) unit-sphere coords (float32)
        elevations = (npw.displacement - npw.seaLevel).astype(np.float32)
    else:
        verts = []
        elevations = []
        for crust in world:  # vanha tapa, jos World on iterable
            verts.append(crust.cartesian)
            elevations.append(crust.elevation)
        verts = np.array(verts, dtype=np.float32)
        elevations = np.array(elevations, dtype=np.float32)

    # Muunna pallokoordinaatteihin (lat, lon)
    norms = np.linalg.norm(verts, axis=1)
    # Vältä jakoa nollalla
    norms[norms == 0] = 1.0
    lats = np.arcsin(verts[:,1] / norms)
    lons = np.arctan2(-verts[:,2], verts[:,0])
    lons = (lons + 2*np.pi) % (2*np.pi)

    # Luo 2D-grid (lon: 0..2pi, lat: -pi/2..pi/2)
    Lon = np.linspace(0, 2*np.pi, resolution_lon)
    Lat = np.linspace(-np.pi/2, np.pi/2, resolution_lat)
    Long, Latt = np.meshgrid(Lon, Lat)

    # Interpoloi (nearest on usein riittävä ja vakaa)
    Elev = griddata(points=(lons, lats), values=elevations, xi=(Long, Latt), method='nearest')

    # Sumea rasteri (poistaa kohinat)
    if smoothing_sigma > 0:
        Elev = gaussian_filter(Elev, sigma=smoothing_sigma)

    # Normalisointi 0..1
    Elev_norm = (Elev - np.min(Elev)) / (np.ptp(Elev) + 1e-9)

    # Tallenna PNG
    plt.imsave(filename, Elev_norm, cmap='gray', origin='lower')
    print(f"✓ Harmaasävy DEM tallennettu: {filename}")

    return Elev_norm

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def world_to_climate_map(world, resolution_lon=2048, resolution_lat=1024,
                         smoothing_sigma=2, filename="world_climate_map.png"):
    """
    Piirtää 2D-kartan, jossa korkeus, merialueet ja ilmasto/kasvillisuus.
    - Vihreä = metsä / trooppinen
    - Ruskea = aavikko
    - Harmaa = tundra
    - Valkoinen = jää
    """
    # Hae pisteet ja korkeudet
    if hasattr(world, "npworld"):
        npw = world.npworld
        verts = npw.positions.copy()
        elevations = (npw.displacement - npw.seaLevel).astype(np.float32)
    else:
        verts = []
        elevations = []
        for crust in world:
            verts.append(crust.cartesian)
            elevations.append(crust.elevation)
        verts = np.array(verts, dtype=np.float32)
        elevations = np.array(elevations, dtype=np.float32)

    # Pallokoordinaatit
    norms = np.linalg.norm(verts, axis=1)
    norms[norms == 0] = 1.0
    lats = np.arcsin(verts[:,1] / norms)
    lons = np.arctan2(-verts[:,2], verts[:,0])
    lons = (lons + 2*np.pi) % (2*np.pi)

    # 2D-grid
    Lon = np.linspace(0, 2*np.pi, resolution_lon)
    Lat = np.linspace(-np.pi/2, np.pi/2, resolution_lat)
    Long, Latt = np.meshgrid(Lon, Lat)

    # Interpoloi korkeus
    Elev = scipy.interpolate.griddata(points=(lons, lats), values=elevations, xi=(Long, Latt), method='nearest')

    # Sumea rasteri
    if smoothing_sigma > 0:
        Elev = gaussian_filter(Elev, sigma=smoothing_sigma)

    # Normalisointi 0..1
    Elev_norm = (Elev - np.min(Elev)) / (np.ptp(Elev) + 1e-9)

    # Simuloidaan yksinkertaiset ilmastot:
    # Luodaan satunnainen keskilämpö ja sademäärä funktion lat mukaan
    temp = np.cos(Latt)  # lat: pohjoinen -> kylmä, päiväntasaaja -> kuuma
    precip = np.sin(Latt*2) * 0.5 + 0.5  # yksinkertainen sadanta-lat funktio

    # Luo RGB-kartta
    rgb = np.zeros((resolution_lat, resolution_lon, 3), dtype=np.uint8)

    # Meri (Elevation < 0) -> sininen
    sea_mask = Elev_norm < 0.5
    rgb[sea_mask] = np.array([50, 80, 150], dtype=np.uint8)

    # Maa
    land_mask = ~sea_mask
    for y in range(resolution_lat):
        for x in range(resolution_lon):
            if land_mask[y,x]:
                h = Elev_norm[y,x]
                t = temp[y,x]
                p = precip[y,x]

                # yksinkertainen päätös puusto/ilmasto
                if h > 0.85:            # vuoret/lumi
                    color = [255,255,255]
                elif t < 0.2:           # kylmä -> tundra
                    color = [150,150,150]
                elif p < 0.3:           # kuiva -> aavikko
                    color = [210,180,140]
                elif p > 0.7:           # kostea -> trooppinen metsä
                    color = [0,120,0]
                else:                   # muu -> havumetsä
                    color = [34,139,34]
                rgb[y,x] = np.array(color, dtype=np.uint8)

    # Tallenna kuva
    plt.imsave(filename, rgb)
    print(f"✓ Ilmasto/kasvisto-kartta tallennettu: {filename}")
    return rgb



def world_to_topomap(world, resolution_lon=2048, resolution_lat=1024,
                     smoothing_sigma=2, filename="world_topomap.png"):
    """
    Luo 2D topografisen kartan pallon pinnasta.
    - Meri = sininen
    - Matala maa = vaaleanvihreä
    - Keskikorkeus = tummempi vihreä
    - Vuoristo = ruskea/valkoinen
    """
    # Hae pisteet ja korkeudet
    if hasattr(world, "npworld"):
        npw = world.npworld
        verts = npw.positions.copy()
        elevations = (npw.displacement - npw.seaLevel).astype(np.float32)
    else:
        verts = []
        elevations = []
        for crust in world:
            verts.append(crust.cartesian)
            elevations.append(crust.elevation)
        verts = np.array(verts, dtype=np.float32)
        elevations = np.array(elevations, dtype=np.float32)

    # Pallokoordinaatit
    norms = np.linalg.norm(verts, axis=1)
    norms[norms == 0] = 1.0
    lats = np.arcsin(verts[:,1] / norms)
    lons = np.arctan2(-verts[:,2], verts[:,0])
    lons = (lons + 2*np.pi) % (2*np.pi)

    # 2D-grid
    Lon = np.linspace(0, 2*np.pi, resolution_lon)
    Lat = np.linspace(-np.pi/2, np.pi/2, resolution_lat)
    Long, Latt = np.meshgrid(Lon, Lat)

    # Interpoloi korkeus
    Elev = griddata(points=(lons, lats), values=elevations, xi=(Long, Latt), method='nearest')

    # Sumea rasteri
    if smoothing_sigma > 0:
        Elev = gaussian_filter(Elev, sigma=smoothing_sigma)

    # Normalisointi 0..1
    Elev_norm = (Elev - np.min(Elev)) / (np.ptp(Elev) + 1e-9)

    # Luo RGB-kartta topovärein
    rgb = np.zeros((resolution_lat, resolution_lon, 3), dtype=np.uint8)

    # Luodaan yksinkertainen topoväripaletti
    for y in range(resolution_lat):
        for x in range(resolution_lon):
            h = Elev_norm[y,x]
            if h < 0.5:  # meri
                rgb[y,x] = [30, 60, 180]  # sininen
            elif h < 0.55:  # matala maa
                rgb[y,x] = [160, 220, 120]  # vaaleanvihreä
            elif h < 0.7:  # keskikorkeus
                rgb[y,x] = [34, 139, 34]  # vihreä
            elif h < 0.85:  # korkea
                rgb[y,x] = [139, 69, 19]  # ruskea
            else:  # huippu
                rgb[y,x] = [255, 255, 255]  # lumi/valkoinen

    # Tallenna PNG
    plt.imsave(filename, rgb)
    print(f"✓ Topografinen kartta tallennettu: {filename}")
    return rgb

import pyvista as pv
import numpy as np

def visualize_npworld_topo(npw, smooth=True):
    """
    Visualisoi NumPyWorld-meshin PyVistalla topoväreillä.
    Topovärit: meri-sininen, matala maa-vaaleanvihreä, keskikorkeus-vihreä, korkea-ruskea, huippu-valkoinen
    """
    # Vertices
    verts = npw.positions * npw.world_radius  # mittayksikkö: m
    N = verts.shape[0]

    # Korkeus normalisoitu
    elevations = npw.displacement - npw.seaLevel
    h_norm = (elevations - elevations.min()) / (np.ptp(elevations) + 1e-9)

    # Luo RGB-taulukko
    colors = np.zeros((N,3), dtype=np.uint8)
    for i in range(N):
        h = h_norm[i]
        if h < 0.5:           # meri
            colors[i] = [30, 60, 180]
        elif h < 0.55:        # matala maa
            colors[i] = [160, 220, 120]
        elif h < 0.7:         # keskikorkeus
            colors[i] = [34, 139, 34]
        elif h < 0.85:        # korkea
            colors[i] = [139, 69, 19]
        else:                 # huippu
            colors[i] = [255, 255, 255]

    # Triangulointi convex hullilla (pallon pinnalla)
    from scipy.spatial import ConvexHull
    hull = ConvexHull(verts)
    triangles = hull.simplices
    # faces PyVistalle
    n_faces = triangles.shape[0]
    faces = np.hstack([np.full((n_faces,1),3), triangles]).astype(np.int64).flatten()

    # Luo PyVista mesh
    mesh = pv.PolyData(verts, faces)
    mesh.point_data['RGB'] = colors

    # Normaalit ja visualisointi
    mesh.compute_normals(cell_normals=False, auto_orient_normals=True, inplace=True)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='RGB', rgb=True, smooth_shading=smooth, show_edges=False)
    plotter.set_background('black')
    plotter.add_axes()
    plotter.show()

import pyvista as pv
import numpy as np

def visualize_npworld_topo_climate(npw, smooth=True):
    """
    Visualisoi NumPyWorld-meshin PyVistalla topoväreillä + ilmasto/biome.
    """
    verts = npw.positions * npw.world_radius
    N = verts.shape[0]

    # Korkeus normalisoitu
    elevations = npw.displacement - npw.seaLevel
    h_norm = (elevations - elevations.min()) / (np.ptp(elevations) + 1e-9)

    # Sijainti pallolla (lat, lon)
    norms = np.linalg.norm(npw.positions, axis=1)
    norms[norms==0] = 1.0
    lats = np.arcsin(npw.positions[:,1] / norms)
    lons = np.arctan2(-npw.positions[:,2], npw.positions[:,0])

    colors = np.zeros((N,3), dtype=np.uint8)
    for i in range(N):
        h = h_norm[i]
        lat = lats[i]
        # yksinkertainen ilmasto-logiikka
        # Korkeus > 0.85 -> jää / vuoriston huippu
        if h > 0.85:
            colors[i] = [255, 255, 255]
        # kylmä leveys -> tundra
        elif abs(lat) > np.pi*0.6:
            colors[i] = [180, 180, 180]
        # matala korkeus ja lämmin kuiva alue -> aavikko
        elif h < 0.3 and abs(lat) < np.pi*0.3:
            colors[i] = [210, 180, 120]
        # matala korkeus ja meri
        elif h < 0.45:
            colors[i] = [30, 60, 180]
        # muuten metsä / kasvillisuus
        else:
            # vihreän sävy korkeuden mukaan
            g = int(80 + 120 * (0.7 - min(h,0.7))/0.7)
            colors[i] = [34, g, 34]

    # Triangulointi (ConvexHull)
    from scipy.spatial import ConvexHull
    hull = ConvexHull(verts)
    triangles = hull.simplices
    n_faces = triangles.shape[0]
    faces = np.hstack([np.full((n_faces,1),3), triangles]).astype(np.int64).flatten()

    mesh = pv.PolyData(verts, faces)
    mesh.point_data['RGB'] = colors
    mesh.compute_normals(cell_normals=False, auto_orient_normals=True, inplace=True)

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='RGB', rgb=True, smooth_shading=smooth, show_edges=False)
    plotter.set_background('black')
    plotter.add_axes()
    plotter.show()

import pyvista as pv
import numpy as np

def visualize_npworld_biome(npw, smooth=True):
    """
    Visualisoi NumPyWorldin PyVistalla topografia + ilmastobiomit.
    Käyttää korkeutta + latitudia + yksinkertaista sademallia biomejen määrittämiseen.
    """
    verts = npw.positions * npw.world_radius
    N = verts.shape[0]

    # Korkeus
    elevations = npw.displacement - npw.seaLevel
    h_norm = (elevations - elevations.min()) / (np.ptp(elevations) + 1e-9)

    # Sijainti pallolla
    norms = np.linalg.norm(npw.positions, axis=1)
    norms[norms==0] = 1.0
    lats = np.arcsin(npw.positions[:,1] / norms)
    lons = np.arctan2(-npw.positions[:,2], npw.positions[:,0])

    # Yksinkertainen ilmasto: lämpö ja sademäärä
    # Lämpö: tropiikki 1.0, navat 0.0, korkeus vähentää lämpöä
    temp = 1.0 - np.abs(lats)/ (np.pi/2) - h_norm*0.5  # 0..1
    temp = np.clip(temp, 0.0, 1.0)
    # Sademäärä: tropiikki ja keskileveysasteet -> enemmän sadetta, kuivat alueet tropiikissa matala sademäärä
    rainfall = np.exp(-((lats/(np.pi/2))**2) * 3.0)  # yksinkertainen käyrä
    rainfall = np.clip(rainfall, 0.0, 1.0)

    # Värit
    colors = np.zeros((N,3), dtype=np.uint8)
    for i in range(N):
        h = h_norm[i]
        t = temp[i]
        r = rainfall[i]

        # Vuoret ja jää
        if h > 0.85:
            colors[i] = [255, 255, 255]
        # Tundra (kylmä, vähän sadetta)
        elif t < 0.3:
            colors[i] = [180, 180, 180]
        # Aavikko (kuuma, vähän sadetta)
        elif t > 0.5 and r < 0.3:
            colors[i] = [210, 180, 120]
        # Savanni (kuuma, keskimääräinen sade)
        elif t > 0.5 and r < 0.6:
            colors[i] = [190, 210, 120]
        # Metsä (riittävästi sadetta ja lämpöä)
        elif t > 0.3 and r > 0.3:
            g = int(100 + 155 * r)  # vihreän sävy sadearvon mukaan
            colors[i] = [34, g, 34]
        # Muu matala korkeus -> meri
        elif h < 0.45:
            colors[i] = [30, 60, 180]
        else:
            # fallback neutraali
            colors[i] = [150, 150, 150]

    # Triangulointi
    from scipy.spatial import ConvexHull
    hull = ConvexHull(verts)
    triangles = hull.simplices
    n_faces = triangles.shape[0]
    faces = np.hstack([np.full((n_faces,1),3), triangles]).astype(np.int64).flatten()

    mesh = pv.PolyData(verts, faces)
    mesh.point_data['RGB'] = colors
    mesh.compute_normals(cell_normals=False, auto_orient_normals=True, inplace=True)

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='RGB', rgb=True, smooth_shading=smooth, show_edges=False)
    plotter.set_background('black')
    plotter.add_axes()
    plotter.show()

import pyvista as pv
import numpy as np
from scipy.spatial import ConvexHull

def visualize_npworld_topo_biome(npw, smooth=True):
    """
    Visualisoi NumPyWorldin PyVistalla topografia + ilmasto sulavasti.
    Vuoret lumisina, rinteet kasvillisuuden/ilmaston mukaan, meri sinisenä.
    """
    verts = npw.positions * npw.world_radius
    N = verts.shape[0]

    # Korkeus normalisoitu 0..1
    elevations = npw.displacement - npw.seaLevel
    h_norm = (elevations - elevations.min()) / (np.ptp(elevations) + 1e-9)

    # Latitudit + yksinkertainen ilmasto
    norms = np.linalg.norm(npw.positions, axis=1)
    norms[norms==0] = 1.0
    lats = np.arcsin(npw.positions[:,1] / norms)

    temp = 1.0 - np.abs(lats) / (np.pi/2) - h_norm*0.5
    temp = np.clip(temp, 0.0, 1.0)
    rainfall = np.exp(-((lats/(np.pi/2))**2) * 3.0)
    rainfall = np.clip(rainfall, 0.0, 1.0)

    # Luo biome-värit (0..1 float)
    biome_colors = np.zeros((N,3), dtype=np.float32)
    for i in range(N):
        t = temp[i]
        r = rainfall[i]
        # meri
        if h_norm[i] < 0.45:
            biome_colors[i] = np.array([30, 60, 180], dtype=np.float32)/255.0
        else:
            # tundra
            if t < 0.3:
                biome_colors[i] = np.array([180,180,180], dtype=np.float32)/255.0
            # aavikko
            elif t > 0.5 and r < 0.3:
                biome_colors[i] = np.array([210,180,120], dtype=np.float32)/255.0
            # savanni
            elif t > 0.5 and r < 0.6:
                biome_colors[i] = np.array([190,210,120], dtype=np.float32)/255.0
            # metsä
            else:
                g = 100 + 155*r
                biome_colors[i] = np.array([34, g, 34], dtype=np.float32)/255.0

    # Topo-värit: lumi huipuilla, ruskea keskikorkeilla
    topo_colors = np.zeros((N,3), dtype=np.float32)
    for i in range(N):
        h = h_norm[i]
        # lumi
        if h > 0.85:
            topo_colors[i] = np.array([1.0,1.0,1.0])
        # keskikorkeus: ruskea
        elif h > 0.55:
            v = (h - 0.55)/0.3
            topo_colors[i] = np.array([139,69,19], dtype=np.float32)/255.0 * v + biome_colors[i]*(1-v)
        else:
            # matala maasto: biome väri
            topo_colors[i] = biome_colors[i]

    # Muunna uint8 RGB
    colors = np.clip(topo_colors*255, 0, 255).astype(np.uint8)

    # Triangulointi
    hull = ConvexHull(verts)
    triangles = hull.simplices
    n_faces = triangles.shape[0]
    faces = np.hstack([np.full((n_faces,1),3), triangles]).astype(np.int64).flatten()

    mesh = pv.PolyData(verts, faces)
    mesh.point_data['RGB'] = colors
    mesh.compute_normals(cell_normals=False, auto_orient_normals=True, inplace=True)

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='RGB', rgb=True, smooth_shading=smooth, show_edges=False)
    plotter.set_background('black')
    plotter.add_axes()
    plotter.show()


# ==================== PÄÄOHJELMA / TESTI ====================
def main():
    print("="*60)
    print("NUMBA-OPTIMOITU PYTECTONICS PROTOTYYPPI (Täysi Numba-ydin)")
    print("="*60)
    # luodaan maailma
    world = World(radius=6367, resolution=256, plateNum=12, continentNum=4, continentSize=10000)
    print(f"Luotu maailma: grid-pisteitä = {world.grid.totalPointNum}, crust-paikkoja = {world.npworld.N}")
    # simulate a few steps
    steps = 50
    for step in range(steps):
        print(f"-- step {step+1}/{steps}, age={world.age:.1f}")
        world.update(timestep=1.0)

    dem=world_to_2d_heightmap(world, resolution_lon=2048, resolution_lat=1024, smoothing_sigma=2, filename="world_heightmap_gray.png")
    #world_to_2d_paletted(world, resolution_lon=2048, resolution_lat=1024, smoothing_sigma=2, filename="world_heightmap_paletted.png")
    world_to_climate_map(world, resolution_lon=2048, resolution_lat=1024,
                         smoothing_sigma=2, filename="world_climate_map.png")
    world_to_topomap(world, resolution_lon=2048, resolution_lat=1024,
                     smoothing_sigma=2, filename="world_topomap.png")                         
    # export mesh
    export_world_mesh_by_plate_npworld(world.npworld, filename="planet_mesh_by_plate_numba.ply")
    # visualize (popup)
    #visualize_npworld(world.npworld)
    #visualize_npworld_topo(world.npworld, smooth=True)
    #visualize_npworld_topo_climate(world.npworld, smooth=True) ## NOK
    #visualize_npworld_biome(world.npworld, smooth=True)
    visualize_npworld_topo_biome(world.npworld, smooth=True)
    print("Valmis.")

if __name__ == "__main__":
    main()
