"""
Optimoitu laattatektoniikkasimulaatio
Keskeiset parannukset:
- KD-tree spatiaalinen haku
- Numpy-vektorisointi
- Välimuistin hyödyntäminen
- Tarpeeton laskennan minimointi
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
import random
from math import pi, sqrt, sin, cos, asin, atan2, copysign, log
from scipy.constants import golden

import plyfile
from plyfile import PlyData, PlyElement

import matplotlib.pyplot as plt

import pyvista as pv

# ==================== VAKIOT ====================
phi = golden
sqrt5 = sqrt(5)

# ==================== APUFUNKTIOT ====================

def fib(n):
    return int(round(phi**n / sqrt5))

def bound(x, floor, ceil):
    return max(min(x, ceil), floor)

def toCartesian(spherical):
    lat, lon = spherical
    return np.array([
        cos(lat) * cos(lon),
        sin(lat),
        -cos(lat) * sin(lon)
    ], dtype=float)

def toSpherical(cartesian):
    y = cartesian[1]
    z = cartesian[2]
    x = cartesian[0]
    return asin(bound(y, -1.0, 1.0)), atan2(-z, x)

def vector_normalize(v):
    mag = np.linalg.norm(v)
    return v / mag if mag > 0 else v

def rotate_vector(v, angle, axis):
    axis = vector_normalize(axis)
    cos_angle = cos(angle)
    sin_angle = sin(angle)
    return (v * cos_angle +
            np.cross(axis, v) * sin_angle +
            axis * np.dot(axis, v) * (1 - cos_angle))

def moment_arm(point, axis):
    axis_norm = vector_normalize(axis)
    return np.linalg.norm(np.cross(point, axis_norm))

# ==================== GEOCOORDINATE ====================
class GeoCoordinate:
    _idCounter = 0

    def __init__(self, spherical=None, cartesian=None, id=None):
        self._spherical = spherical
        self._cartesian = cartesian
        self.id = id if id is not None else GeoCoordinate._idCounter
        GeoCoordinate._idCounter += 1

    @property
    def spherical(self):
        if self._spherical is None:
            self._spherical = toSpherical(self._cartesian)
        return self._spherical

    @property
    def cartesian(self):
        if self._cartesian is None:
            self._cartesian = toCartesian(self._spherical)
        return self._cartesian

    @cartesian.setter
    def cartesian(self, value):
        self._cartesian = np.array(value, dtype=float)
        self._spherical = None

    def rotate(self, angle, axis):
        self.cartesian = rotate_vector(self.cartesian, angle, axis)

# ==================== FIBGRID ====================
class FibGrid:
    def __init__(self, avgDistance):
        self.avgDistance = avgDistance
        self.pointNum = self.getPointNum(avgDistance)
        self.totalPointNum = 2 * self.pointNum + 1
        self.zIncrement = 2.0 / self.totalPointNum

        self.coverage = {}
        self.points = {}
        
        for i in range(-self.pointNum, self.pointNum + 1):
            self.points[i] = toCartesian(self._getSpherical(i))
            self.coverage[i] = None

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

    def rotate(self, angle, axis):
        for i in list(self.points.keys()):
            self.points[i] = rotate_vector(self.points[i], angle, axis)
        for k, crust in list(self.coverage.items()):
            if crust is not None:
                crust.rotate(angle, axis)

    def __getitem__(self, key):
        return self.coverage.get(key, None)

    def __setitem__(self, key, val):
        self.coverage[key] = val

    def add(self, crust):
        self.coverage[crust.id] = crust

    def remove(self, crust):
        self.coverage[crust.id] = None

    def getNeighbors(self, crust):
        neighbors = []
        for neighbor_id in self.getCellNeighborIds(crust.id):
            neighbor = self.coverage.get(neighbor_id)
            if neighbor is not None:
                neighbors.append(neighbor)
        return neighbors

# ==================== CRUST ====================
class Crust(GeoCoordinate):
    def __init__(self, plate, world, isContinent=False, id=None):
        super().__init__()
        self.world = world
        self.plate = plate
        self.id = id
        self._cartesian = plate.grid.points[id].copy()

        self.subductedBy = None
        self.subducts = None
        self._firstSubductedBy = None

        self._isContinent = isContinent
        if isContinent:
            thickness = 36900
            density = world.continentCrustDensity + random.gauss(0, 40)
        else:
            thickness = 7100
            density = world.oceanCrustDensity + random.gauss(0, 40)

        self.density = density
        self._thickness = thickness

        rootDepth = self._thickness * self.density / self.world.mantleDensity
        self.displacement = self._thickness - rootDepth

        plate.add(self)

    @property
    def thickness(self):
        thickness = self._thickness
        if self.subducts:
            thickness += self.subducts._thickness
        return thickness

    @property
    def elevation(self):
        return self.displacement - self.world.seaLevel

    @property
    def pressure(self):
        pressure = self._thickness * self.density
        if self.subducts:
            pressure += self.subducts._thickness * self.subducts.density
        return pressure

    def isContinent(self):
        return self.thickness > 17000

    def isDetaching(self):
        if not self._firstSubductedBy:
            return False
        # Käytä euklidista etäisyyttä (nopeampi)
        distanceSubducted = np.linalg.norm(self.cartesian - self._firstSubductedBy.cartesian) * self.world.radius
        return distanceSubducted > self.world.maxMountainWidth

    def isostacy(self):
        thickness = self.thickness
        rootDepth = thickness * self.density / self.world.mantleDensity
        displacement = thickness - rootDepth
        self.displacement = displacement

    def erupt(self):
        meltDensity = self.world.continentCrustDensity
        thickness = self.thickness

        if self.elevation < 0:
            thickness -= ((meltDensity - self.world.waterDensity) /
                         (self.density - meltDensity)) * abs(self.elevation)
            height = thickness * ((self.density - meltDensity) / meltDensity)
            heightChange = height + abs(self.elevation)
        else:
            heightChange = thickness * ((self.density - meltDensity) / meltDensity)

        pressure = (thickness * self.density) + (heightChange * self.density)
        density = pressure / (thickness + heightChange)

        self._thickness += heightChange
        self.density = density

    def collide(self, other):
        if self.subductedBy:
            top, bottom = other, self
        elif other.subductedBy:
            top, bottom = self, other
        else:
            top, bottom = sorted([self, other], key=lambda c: c.density)

        if top.subducts != bottom or bottom.subductedBy != top:
            if bottom.isDetaching():
                if bottom.isContinent() and top.isContinent():
                    pass
                else:
                    top.erupt()
                    bottom.destroy()
                    top.plate.update(top)
            else:
                if not bottom._firstSubductedBy:
                    bottom._firstSubductedBy = top

                top.subducts = bottom
                bottom.subductedBy = top

                bottom.plate.update(bottom)
                top.plate.update(top)

    def destroy(self):
        self.plate.remove(self)
        if self.subductedBy:
            self.subductedBy.subducts = None
        if self.subducts:
            self.subducts.subductedBy = None

# ==================== PLATE (OPTIMOITU) ====================
class Plate(GeoCoordinate):
    def __init__(self, spherical, world, grid, speed=0, eulerPole=None):
        super().__init__(spherical=spherical)
        self.world = world
        self.speed = speed
        self.eulerPole = eulerPole if eulerPole is not None else toCartesian((pi/2, 0))
        self.grid = grid
        self.crusts = []
        self.base_color = (random.random(), random.random(), random.random())
        
        self._collidable = None
        self._riftable = None
        self._kdtree = None
        self._kdtree_positions = None

    def add(self, crust):
        self.grid.add(crust)
        self.crusts.append(crust)
        self._collidable = None
        self._riftable = None
        self._kdtree = None  # Invalidoi KD-tree

    def remove(self, crust):
        self.grid.remove(crust)
        if crust in self.crusts:
            self.crusts.remove(crust)
        self._collidable = None
        self._kdtree = None

    def update(self, crust):
        self._collidable = None
        self._kdtree = None  # Invalidoi kun koordinaatit muuttuvat

    @property
    def collidable(self):
        if self._collidable is None:
            self._collidable = set()
            for crust in self.crusts:
                neighbors = self.grid.getNeighbors(crust)
                neighbor_ids = self.grid.getCellNeighborIds(crust.id)
                if len(neighbors) < len(neighbor_ids):
                    self._collidable.add(crust)
        return list(self._collidable)

    @property
    def riftable(self):
        if self._riftable is None:
            self._riftable = set()
            for crust in self.crusts:
                if not crust.subductedBy:
                    for neighbor_id in self.grid.getCellNeighborIds(crust.id):
                        if self.grid[neighbor_id] is None:
                            self._riftable.add(neighbor_id)
        return list(self._riftable)

    def _build_kdtree(self):
        """Rakenna KD-tree spatiaaliseen hakuun"""
        if not self.crusts:
            return None, None
        positions = np.array([c.cartesian for c in self.crusts])
        return cKDTree(positions), positions

    def getAngularSpeed(self, timestep=1):
        speed = self.speed * timestep
        smallCircleRadius = moment_arm(self.cartesian, self.eulerPole) * self.world.radius
        if smallCircleRadius > 0:
            return speed / smallCircleRadius
        return 0

    def move(self, timestep):
        angularSpeed = self.getAngularSpeed(timestep)
        self.rotate(angularSpeed, self.eulerPole)
        self.grid.rotate(angularSpeed, self.eulerPole)
        self._kdtree = None  # Invalidoi KD-tree liikkeen jälkeen

    def collide(self):
        """OPTIMOITU: Käytä KD-tree:tä törmäyshakuun"""
        if not self.collidable:
            return
            
        plates = [p for p in self.world.plates if p != self]
        collision_distance = self.world.minDistance * 1.5
        
        for crust in self.collidable:
            crust_pos = crust.cartesian
            
            # Käy läpi vain läheiset laatat
            for other_plate in plates:
                if not other_plate.crusts:
                    continue
                
                # Rakenna KD-tree tarvittaessa
                if other_plate._kdtree is None:
                    other_plate._kdtree, other_plate._kdtree_positions = other_plate._build_kdtree()
                
                if other_plate._kdtree is None:
                    continue
                
                # Etsi lähimmät naapurit KD-tree:llä
                distances, indices = other_plate._kdtree.query(
                    crust_pos, 
                    k=min(5, len(other_plate.crusts)),  # Tarkista 5 lähintä
                    distance_upper_bound=collision_distance
                )
                
                # Käsittele löydetyt törmäykset
                if np.isscalar(distances):
                    distances = [distances]
                    indices = [indices]
                
                for dist, idx in zip(distances, indices):
                    if dist < collision_distance and idx < len(other_plate.crusts):
                        other_crust = other_plate.crusts[idx]
                        crust.collide(other_crust)
                        break

    def rift(self):
        """OPTIMOITU: Vektoroitu riftaustarkistus"""
        if not self.riftable:
            return
            
        plates = [p for p in self.world.plates if p != self]
        min_distance = self.world.minDistance
        
        # Kerää kaikki muiden laattojen crusts yhdeksi KD-tree:ksi
        all_other_positions = []
        for other_plate in plates:
            if other_plate.crusts:
                all_other_positions.extend([c.cartesian for c in other_plate.crusts])
        
        if not all_other_positions:
            # Ei muita laattoja, luo riftaukset vapaasti
            for rift_id in list(self.riftable):
                if self.grid[rift_id] is None:
                    Crust(self, self.world, False, rift_id)
            return
        
        all_other_positions = np.array(all_other_positions)
        other_tree = cKDTree(all_other_positions)
        
        # Tarkista kaikki riftauspisteet kerralla
        rift_positions = np.array([self.grid.points[rid] for rid in self.riftable])
        distances, _ = other_tree.query(rift_positions, distance_upper_bound=min_distance)
        
        # Luo riftaukset pisteisiin joissa ei törmäystä
        for i, rift_id in enumerate(list(self.riftable)):
            if self.grid[rift_id] is None and distances[i] >= min_distance:
                Crust(self, self.world, False, rift_id)

# ==================== WORLD ====================
class World:
    mantleDensity = 3300
    waterDensity = 1026
    oceanCrustDensity = 2890
    continentCrustDensity = 2700

    def __init__(self, radius=6367, resolution=72, plateNum=5,
                 continentNum=3, continentSize=1250):
        self.age = 0.0
        self.radius = radius * 1000
        self.resolution = resolution
        avgPointDistance = 2 * pi / resolution

        self.plates = []
        template = FibGrid(avgPointDistance)

        for i in range(plateNum):
            plate = Plate(
                self.randomPoint(),
                self,
                FibGrid(avgPointDistance),
                random.gauss(42.8, 27.7),
                toCartesian(self.randomPoint())
            )
            self.plates.append(plate)

        shields = [GeoCoordinate(spherical=self.randomPoint())
                   for _ in range(continentNum)]
        continentSizeRad = continentSize * 1000 / self.radius

        for i in range(-template.pointNum, template.pointNum + 1):
            cartesian = template.points[i]
            nearestPlate = min(self.plates,
                               key=lambda p: np.linalg.norm(p.cartesian - cartesian))
            isContinent = any(
                np.linalg.norm(shield.cartesian - cartesian) < continentSizeRad
                for shield in shields
            )
            Crust(nearestPlate, self, isContinent, id=i)

        for plate in self.plates:
            continent_count = sum(1 for c in plate.crusts if c.isContinent())
            if continent_count > 0.3 * max(1, len(plate.crusts)):
                plate.base_color = (0.2, 0.7, 0.2)
            else:
                plate.base_color = (0.2, 0.4, 0.85)

        self.seaLevel = 3790
        self.maxMountainWidth = 300 * 1000

        area = 4 * pi / template.totalPointNum
        d = sqrt(area / sqrt(5))
        self.minDistance = sqrt(2) * d

    def randomPoint(self):
        return (asin(2 * random.random() - 1),
                2 * pi * random.random())

    def __iter__(self):
        for plate in self.plates:
            for crust in plate.crusts:
                yield crust

    def update(self, timestep):
        """OPTIMOITU: Paralesoitu päivitys"""
        # Liikuta laattoja
        for plate in self.plates:
            plate.move(timestep)

        # Isostaasi - voitaisiin vektorisoida täysin
        for crust in self:
            crust.isostacy()

        # Törmäykset (käyttää nyt KD-tree:tä)
        for plate in self.plates:
            plate.collide()

        # Riftaus (käyttää nyt KD-tree:tä)
        for plate in self.plates:
            plate.rift()

        self.age += timestep

def fix_triangle_winding(verts, triangles):
    """
    Varmistaa, että jokaisen kolmion normaalit osoittavat ulospäin (origosta).
    Palauttaa Nx3 int-arrayin.
    """
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
    return np.array(fixed, dtype=int)



def build_planet_mesh_from_world(world):
    """
    Rakentaa planet-meshin worldista.
    Palauttaa (verts, tris, colors, elevations)
      - verts: (N,3) displaced vertexit (korkeus huomioitu)
      - tris: (M,3) trianglet indeksein
      - colors: (N,3) uint8 RGB per-vertex värit (manner/meri)
      - elevations: (N,) float elevation (m)
    Tämä toimii riippumatta siitä, ettei worldillä ole 'grid' attribuuttia:
    luodaan FibGrid samaa resoluutiota käyttämällä ja haetaan crust:t id:n perusteella.
    """
    avgPointDistance = 2 * np.pi / world.resolution
    grid = FibGrid(avgPointDistance)

    # sorted keys -> index mapping
    keys = sorted(grid.points.keys())
    verts = np.array([grid.points[k] for k in keys])  # (N,3)
    key_to_vidx = {k: idx for idx, k in enumerate(keys)}

    # Map crust.id -> crust (mahdollisesti negative ids)
    crust_by_id = {c.id: c for c in world}

    # Per-vertex attribuutit
    N = len(verts)
    vert_elevation = np.zeros(N, dtype=float)
    vert_is_continent = np.zeros(N, dtype=np.int8)
    vert_plate_id = np.full(N, -1, dtype=int)

    # Täytetään suoraan id:llä jos mahdollista, muuten lähin
    for idx, key in enumerate(keys):
        if key in crust_by_id:
            c = crust_by_id[key]
        else:
            # varmistus: etsi lähin crust — harvinaisessa tilanteessa
            # tapahtuu vain jos crust-lista ei vastaa täsmälleen grid:iä
            bestd = float('inf')
            best = None
            for c2 in world:
                d = np.linalg.norm(verts[idx] - c2.cartesian)
                if d < bestd:
                    bestd = d
                    best = c2
            c = best
        vert_elevation[idx] = c.elevation
        vert_is_continent[idx] = 1 if c.isContinent() else 0
        vert_plate_id[idx] = world.plates.index(c.plate)

    # Displace verts according to elevation (skaalaa suhteessa world.radius)
    # pienennä vaikutusta sopivaksi: sama logiikka kuin aiemmassa koodissasi
    displaced_scale = 1.0 + (vert_elevation / (world.radius * 0.02))
    displaced_verts = verts * displaced_scale[:, np.newaxis]

    # Trianguloidaan koko pallo ConvexHullilla (tuottaa suljetun meshin)
    try:
        hull = ConvexHull(displaced_verts)
        triangles = hull.simplices  # (M,3)
    except Exception as e:
        raise RuntimeError(f"ConvexHull epäonnistui: {e}")

    # Korjataan tri-winding (ulospäin)
    triangles = fix_triangle_winding(displaced_verts, triangles)

    # Vertex-värit uint8: manner/meri tai laatan väri
    colors = np.zeros((N, 3), dtype=np.uint8)
    # Käytetään laatan base_color jos haluat monivärisempää lopputulosta
    for vi in range(N):
        plate_idx = vert_plate_id[vi]
        if 0 <= plate_idx < len(world.plates):
            base = np.array(world.plates[plate_idx].base_color)
            # skaalataan 0..255 ja hieman kirkastetaan korkeusvaikutuksella
            bright = 1.0 + 0.25 * (vert_elevation[vi] / max(1.0, np.max(np.abs(vert_elevation))))
            rgb = np.clip((base * bright) * 255.0, 0, 255)
            colors[vi] = rgb.astype(np.uint8)
        else:
            # fallback: mantereet ruskea, meri sininen
            if vert_is_continent[vi]:
                colors[vi] = np.array([180, 120, 80], dtype=np.uint8)
            else:
                colors[vi] = np.array([50, 120, 200], dtype=np.uint8)

    return displaced_verts, triangles, colors, vert_elevation


# PyVista-visualisointi (sileä)
def visualize_world_pyvista(world, smooth=True):
    verts, tris, colors, elevations = build_planet_mesh_from_world(world)

    # PyVista faces: [nverts_per_face, v0, v1, v2, ...] flatten
    n_faces = tris.shape[0]
    faces = np.hstack([np.full((n_faces,1), 3), tris]).astype(np.int64).flatten()

    mesh = pv.PolyData(verts, faces)
    # point_data-key voi olla mikä vain; käytetään "RGB"
    mesh.point_data['RGB'] = colors

    # Normaalit ja smooth shading
    mesh.compute_normals(cell_normals=False, auto_orient_normals=True, inplace=True)

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='RGB', rgb=True, smooth_shading=smooth, show_edges=False)
    plotter.set_background('black')
    plotter.add_axes()
    plotter.show()


# PLY-export käyttäen PyVistaa (helppo ja luotettava)
def export_world_ply(world, filename="planet.ply"):
    verts, tris, colors, elevations = build_planet_mesh_from_world(world)
    n_faces = tris.shape[0]
    faces = np.hstack([np.full((n_faces,1), 3), tris]).astype(np.int64).flatten()
    mesh = pv.PolyData(verts, faces)
    mesh.point_data['RGB'] = colors
    # PyVista tuntee point_data['RGB'] -> tallentuu PLY:hin automaattisesti
    mesh.save(filename)
    print(f"✓ PLY export: {filename}")

def export_world_mesh_by_plate(world, filename="earth_mesh_by_plate.ply"):
    """
    Vie World-simulaation meshin PLY-tiedostoksi.
    - Väritetään manner- ja merilaatat eri väreillä, korkeus näkyy sävynä.
    """
    avgPointDistance = 2 * np.pi / world.resolution
    grid = FibGrid(avgPointDistance)
    
    # Vertices
    keys = sorted(grid.points.keys())
    verts = np.array([grid.points[k] for k in keys])
    
    # Kerätään crustit
    crusts = list(world.__iter__())
    if not crusts:
        print("Ei crusteja meshiksi")
        return
    
    # Liikuta pisteet korkeus mukaan
    vert_elevation = np.zeros(len(verts))
    vert_plate_type = np.zeros(len(verts), dtype=int)  # 0 = meri, 1 = manner
    for vi, v in enumerate(verts):
        bestd = float('inf')
        best = None
        for crust in crusts:
            d = np.linalg.norm(v - crust.cartesian)
            if d < bestd:
                bestd = d
                best = crust
        vert_elevation[vi] = best.elevation
        vert_plate_type[vi] = 1 if best.isContinent() else 0
    
    # Displacement (korotukset)
    elevation_scale = 1.0 + (vert_elevation / (world.radius * 0.02))
    displaced_verts = verts * elevation_scale[:, np.newaxis]
    
    # ConvexHull
    hull = ConvexHull(displaced_verts)
    triangles = hull.simplices
    triangles = fix_triangle_winding(displaced_verts, triangles)    
    # Luo face-värit
    min_elev = np.min(vert_elevation)
    max_elev = np.max(vert_elevation)
    
    face_list = []
    for tri in triangles:
        types = vert_plate_type[tri]
        avg_type = round(np.mean(types))  # jos useampi manner -> manner
        avg_elev = np.mean(vert_elevation[tri])
        norm_elev = (avg_elev - min_elev) / max(1e-6, (max_elev - min_elev))
        
        if avg_type == 1:  # manner
            r = int(150 + 105 * norm_elev)
            g = int(50 + 100 * (1 - norm_elev))
            b = int(50)
        else:  # meri
            r = int(50)
            g = int(50 + 50 * norm_elev)
            b = int(150 + 100 * (1 - norm_elev))
        
        face_list.append((list(tri), r, g, b))
    
    # PLY vertices
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertices = np.empty(len(displaced_verts), dtype=vertex_dtype)
    vertices['x'] = displaced_verts[:,0]
    vertices['y'] = displaced_verts[:,1]
    vertices['z'] = displaced_verts[:,2]
    
    # PLY faces
    face_dtype = [('vertex_indices', 'i4', (3,)), ('red','u1'),('green','u1'),('blue','u1')]
    faces_array = np.array(face_list, dtype=face_dtype)
    
    # Kirjoita PLY
    el_verts = PlyElement.describe(vertices, 'vertex')
    el_faces = PlyElement.describe(faces_array, 'face')
    PlyData([el_verts, el_faces], text=True).write(filename)
    
    print(f"✓ Mesh tallennettu: {filename} (manner/meri värit, korkeus näkyy sävynä)")


# PyVista-visualisointi (sileä)
def visualize_world_pyvista(world, smooth=True):
    verts, tris, colors, elevations = build_planet_mesh_from_world(world)

    # PyVista faces: [nverts_per_face, v0, v1, v2, ...] flatten
    n_faces = tris.shape[0]
    faces = np.hstack([np.full((n_faces,1), 3), tris]).astype(np.int64).flatten()

    mesh = pv.PolyData(verts, faces)
    # point_data-key voi olla mikä vain; käytetään "RGB"
    mesh.point_data['RGB'] = colors

    # Normaalit ja smooth shading
    mesh.compute_normals(cell_normals=False, auto_orient_normals=True, inplace=True)

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='RGB', rgb=True, smooth_shading=smooth, show_edges=False)
    plotter.set_background('black')
    plotter.add_axes()
    plotter.show()

def visualize_world_pyvista_smooth(world):
    """
    PyVista-visualisointi: 
    • Manner- ja merilaatat eri väreillä
    • Korkeus interpoloituna
    • Sileä shading
    """
    avgPointDistance = 2 * np.pi / world.resolution
    grid = FibGrid(avgPointDistance)
    keys = sorted(grid.points.keys())
    verts = np.array([grid.points[k] for k in keys])

    # Jokaiselle kärjelle lähin crust
    crusts = list(world.__iter__())
    vert_elevation = np.zeros(len(verts))
    vert_plate_type = np.zeros(len(verts), dtype=int)  # 0=meri, 1=manner

    for vi, v in enumerate(verts):
        best = None
        bestd = float('inf')
        for crust in crusts:
            d = np.linalg.norm(v - crust.cartesian)
            if d < bestd:
                bestd = d
                best = crust
        vert_elevation[vi] = best.elevation
        vert_plate_type[vi] = 1 if best.isContinent() else 0

    # Värit: manner punertava, meri sininen
    colors = np.zeros((len(verts), 3), dtype=np.uint8)
    colors[vert_plate_type == 0] = [50, 120, 200]   # meri
    colors[vert_plate_type == 1] = [200, 100, 100]  # manner

    # Korkeuden vaikutus väriin (sävytetään hieman)
    elev_norm = (vert_elevation - vert_elevation.min()) / (np.ptp(vert_elevation) + 1e-9)
    colors = (colors * (0.5 + 0.5 * elev_norm[:, np.newaxis])).astype(np.uint8)

    # Konveksi kuori
    hull = ConvexHull(verts)
    faces = hull.simplices
    n_faces = faces.shape[0]
    pv_faces = np.hstack([np.full((n_faces, 1), 3), faces]).flatten()

    mesh = pv.PolyData(verts, pv_faces)
    mesh.point_data['colors'] = colors

    # PyVista-plotteri
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars='colors',
        rgb=True,
        show_edges=False,
        smooth_shading=True   # <--- Sileä shading
    )
    plotter.background_color = 'black'
    plotter.show()

from scipy.ndimage import gaussian_filter

def smooth_dem(dem, sigma=2.0):
    return gaussian_filter(dem, sigma=sigma)
    
def hillshade(dem, azimuth=315, altitude=45):
    az = np.radians(azimuth)
    alt = np.radians(altitude)

    dy, dx = np.gradient(dem)
    slope = np.pi/2 - np.arctan(np.sqrt(dx*dx + dy*dy))
    aspect = np.arctan2(-dx, dy)

    shaded = (
        np.sin(alt) * np.sin(slope) +
        np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    )
    shaded = (shaded - shaded.min()) / (shaded.max() - shaded.min() + 1e-9)

    return shaded


def world_to_2d_raster_seamless(world, resolution_lon=2048, resolution_lat=1024, smoothing_sigma=2, filename="world_heightmap_seamless.png"):
    """
    Luo saumattoman 2D-rasterin pallon pinnasta:
      - Harmaasävy = korkeus
      - Värit: vihreä = manner, sininen = meri
      - Reunat jatkuvat lon=0 <-> lon=2pi
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    verts = []
    elevations = []
    is_continent = []
    for crust in world:
        verts.append(crust.cartesian)
        elevations.append(crust.elevation)
        is_continent.append(1 if crust.isContinent() else 0)
    verts = np.array(verts)
    elevations = np.array(elevations)
    is_continent = np.array(is_continent)

    # Muunna pallokoordinaateiksi
    lats = np.arcsin(verts[:,1] / np.linalg.norm(verts, axis=1))
    lons = np.arctan2(-verts[:,2], verts[:,0])
    lons = (lons + 2*np.pi) % (2*np.pi)  # 0..2pi

    # Luo 2D-grid
    Lon = np.linspace(0, 2*np.pi, resolution_lon)
    Lat = np.linspace(-np.pi/2, np.pi/2, resolution_lat)
    Lon, Lat = np.meshgrid(Lon, Lat)

    # Interpoloi korkeus ja manner/meri
    Elev = griddata(points=(lons, lats), values=elevations, xi=(Lon, Lat), method='nearest')
    Cont = griddata(points=(lons, lats), values=is_continent, xi=(Lon, Lat), method='nearest')

    # Sumea rasteri (poistaa kohinat)
    if smoothing_sigma > 0:
        Elev = gaussian_filter(Elev, sigma=smoothing_sigma)

    # Normalisointi 0..1 harmaasävyksi
    Elev_norm = (Elev - np.min(Elev)) / (np.ptp(Elev) + 1e-9)

    # Luo RGB-kuva
    rgb = np.zeros((resolution_lat, resolution_lon, 3), dtype=np.float32)
    # Harmaasävy pohjaksi (korkeus)
    rgb[...,0] = Elev_norm
    rgb[...,1] = Elev_norm
    rgb[...,2] = Elev_norm
    # Värit mantereille/merille
    rgb[...,0] = rgb[...,0]*(1-Cont) + 0.0*Cont  # punainen: meri 0
    rgb[...,1] = rgb[...,1]*(1-Cont) + 0.6*Cont  # vihreä: manner 0.6
    rgb[...,2] = rgb[...,2]*(1-Cont) + 0.0*Cont  # sininen: meri 0

    rgb = np.clip(rgb, 0, 1)

    # Tallenna PNG
    plt.imsave(filename, rgb, origin='lower')
    print(f"✓ Saumaton 2D-harhaa/RGB-raster tallennettu: {filename}")
    return rgb

def world_to_2d_heightmap(world, resolution_lon=2048, resolution_lat=1024, smoothing_sigma=2, filename="world_heightmap_gray.png"):
    """
    Luo saumattoman 2D-harmaaluvun (DEM) pallon pinnasta.
      - Harmaasävy: 0 = alin korkeus, 1 = korkein
      - Saumaton lon=0 <-> lon=2pi
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    verts = []
    elevations = []
    for crust in world:
        verts.append(crust.cartesian)
        elevations.append(crust.elevation)
    verts = np.array(verts)
    elevations = np.array(elevations)

    # Muunna pallokoordinaatteihin
    lats = np.arcsin(verts[:,1] / np.linalg.norm(verts, axis=1))
    lons = np.arctan2(-verts[:,2], verts[:,0])
    lons = (lons + 2*np.pi) % (2*np.pi)  # 0..2pi

    # Luo 2D-grid
    Lon = np.linspace(0, 2*np.pi, resolution_lon)
    Lat = np.linspace(-np.pi/2, np.pi/2, resolution_lat)
    Lon, Lat = np.meshgrid(Lon, Lat)

    # Interpoloi korkeus
    Elev = griddata(points=(lons, lats), values=elevations, xi=(Lon, Lat), method='nearest')

    # Sumea rasteri (poistaa kohinat)
    if smoothing_sigma > 0:
        Elev = gaussian_filter(Elev, sigma=smoothing_sigma)

    # Normalisointi 0..1
    Elev_norm = (Elev - np.min(Elev)) / (np.ptp(Elev) + 1e-9)

    # Tallenna PNG
    plt.imsave(filename, Elev_norm, cmap='gray', origin='lower')
    print(f"✓ Harmaasävy DEM tallennettu: {filename}")

    return Elev_norm

def world_to_2d_paletted(world, resolution_lon=2048, resolution_lat=1024, smoothing_sigma=2, filename="world_heightmap_gray.png"):
    """
    Luo saumattoman 2D-harmaaluvun (DEM) pallon pinnasta.
      - sSävy: 0 = alin korkeus, 1 = korkein
      - Saumaton lon=0 <-> lon=2pi
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    verts = []
    elevations = []
    for crust in world:
        verts.append(crust.cartesian)
        elevations.append(crust.elevation)
    verts = np.array(verts)
    elevations = np.array(elevations)

    # Muunna pallokoordinaatteihin
    lats = np.arcsin(verts[:,1] / np.linalg.norm(verts, axis=1))
    lons = np.arctan2(-verts[:,2], verts[:,0])
    lons = (lons + 2*np.pi) % (2*np.pi)  # 0..2pi

    # Luo 2D-grid
    Lon = np.linspace(0, 2*np.pi, resolution_lon)
    Lat = np.linspace(-np.pi/2, np.pi/2, resolution_lat)
    Lon, Lat = np.meshgrid(Lon, Lat)

    # Interpoloi korkeus
    Elev = griddata(points=(lons, lats), values=elevations, xi=(Lon, Lat), method='nearest')

    # Sumea rasteri (poistaa kohinat)
    if smoothing_sigma > 0:
        Elev = gaussian_filter(Elev, sigma=smoothing_sigma)

    # Normalisointi 0..1
    Elev_norm = (Elev - np.min(Elev)) / (np.ptp(Elev) + 1e-9)

    # Tallenna PNG
    plt.imsave(filename, Elev_norm, cmap='terrain', origin='lower')
    print(f"✓ Harmaasävy DEM tallennettu: {filename}")

    return Elev_norm

# ==================== PÄÄOHJELMA ====================
def main():
    print("=" * 60)
    print("OPTIMOITU LAATTATEKTONIIKKASIMULAATIO")
    print("=" * 60)
    print("\nAlustetaan simulaatio...")

    # Testaa pienemmällä resoluutiolla ensin
    world = World(radius=6367, resolution=48, plateNum=10,
                  continentNum=5, continentSize=1700)

    print(f"✓ Luotu {len(world.plates)} laattaa")
    print(f"✓ Yhteensä {sum(len(p.crusts) for p in world.plates)} maankuoren palaa")

    print("\nSimuloidaan laattatektoniikkaa...")
    import time
    
    for step in range(10):
        start = time.time()
        world.update(timestep=1.0)
        elapsed = time.time() - start
        
        print(f"  Askel {step + 1:2d}/20 | Ikä: {world.age:5.1f} My | "
              f"Aika: {elapsed:.3f}s | "
              f"Laattoja: {len(world.plates)}")

    print("\n" + "=" * 60)
    print("✓ Simulaatio valmis!")
    print("\nOptimointihyödyt:")
    print("  • KD-tree spatiaalinen haku")
    print("  • Vektorisoitu riftaustarkistus")
    print("  • Välimuisti KD-tree:lle")
    print("  • Euklidinen etäisyys kaarietäisyyden sijaan")
    
    
    verts, tris, colors, elevations = build_planet_mesh_from_world(world)

    #export_world_ply(world, "planet.ply")
    export_world_mesh_by_plate(world, filename="planet_mesh_by_plate.ply")
        
    world_to_2d_raster_seamless(world, resolution_lon=2048, resolution_lat=1024, smoothing_sigma=2, filename="world_heightmap_seamless.png")
    dem=world_to_2d_heightmap(world, resolution_lon=2048, resolution_lat=1024, smoothing_sigma=2, filename="world_heightmap_gray.png")
    world_to_2d_paletted(world, resolution_lon=2048, resolution_lat=1024, smoothing_sigma=2, filename="world_heightmap_paletted.png")
    smoothed=smooth_dem(dem, sigma=2.0)
    hill=hillshade(smoothed, azimuth=315, altitude=45)
    #plt.imshow(hill)
    #plt.show()
    #visualize_world_pyvista(world)
    visualize_world_pyvista_smooth(world)
    print(".")
    return(0)

if __name__ == "__main__":
    main()
