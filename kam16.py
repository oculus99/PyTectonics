"""
Laattatektoniikkasimulaatio — korjattu kokonaisversio
Python 3

Riippuvuudet:
 pip install numpy scipy plotly

Suoritus: python3 laattatektoniikka_simulaatio_fixed.py
"""

import numpy as np
import plotly.graph_objects as go
from math import pi, sqrt, sin, cos, asin, atan2, copysign, log
from scipy.constants import golden
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
import random
import plyfile
from plyfile import PlyData, PlyElement
import pyvista as pv

# ==================== VAKIOT ====================
RIGHT_ANGLE = pi / 2
SEMICIRCLE = pi
CIRCLE = 2 * pi
phi = golden
sqrt5 = sqrt(5)

# ==================== APUFUNKTIOT ====================

def fib(n):
    """Palauttaa n:nnen Fibonacci-luvun approksimaation"""
    return int(round(phi**n / sqrt5))


def sign(n):
    """Palauttaa 1 positiiviselle, -1 negatiiviselle, 0 nollalle"""
    return 0 if n == 0 else copysign(1, n)


def bound(x, floor, ceil):
    """Rajaa arvon annettujen rajojen sisään"""
    return max(min(x, ceil), floor)


def toCartesian(spherical):
    """Muuntaa lat/lon koordinaatit karteesisiksi
    Spherical = (lat, lon) radiaaneina
    Palauttaa numpy-vektorin [x, y, z]
    """
    lat, lon = spherical
    return np.array([
        cos(lat) * cos(lon),
        sin(lat),
        -cos(lat) * sin(lon)
    ], dtype=float)


def toSpherical(cartesian):
    """Muuntaa karteesiset koordinaatit lat/lon:iksi (lat, lon)"""
    if isinstance(cartesian, np.ndarray):
        y = cartesian[1]
        z = cartesian[2]
        x = cartesian[0]
    else:
        x, y, z = cartesian
    return asin(bound(y, -1.0, 1.0)), atan2(-z, x)


def vector_magnitude(v):
    """Laskee vektorin pituuden"""
    return np.linalg.norm(v)


def vector_normalize(v):
    """Normalisoi vektorin"""
    mag = vector_magnitude(v)
    return v / mag if mag > 0 else v


def rotate_vector(v, angle, axis):
    """Rotoi vektoria akselin ympäri Rodriguesin kaavalla"""
    axis = vector_normalize(axis)
    cos_angle = cos(angle)
    sin_angle = sin(angle)
    return (v * cos_angle +
            np.cross(axis, v) * sin_angle +
            axis * np.dot(axis, v) * (1 - cos_angle))


def moment_arm(point, axis):
    """Laskee momentin varren: etäisyys pisteestä akseliin"""
    axis_norm = vector_normalize(axis)
    return vector_magnitude(np.cross(point, axis_norm))

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

    @spherical.setter
    def spherical(self, value):
        self._spherical = value
        self._cartesian = None

    @property
    def cartesian(self):
        if self._cartesian is None:
            self._cartesian = toCartesian(self._spherical)
        return self._cartesian

    @cartesian.setter
    def cartesian(self, value):
        self._cartesian = np.array(value, dtype=float)
        self._spherical = None

    def getArcDistance(self, other):
        """Laskee kaarietäisyyden kahden pisteen välillä (radiaaneina)"""
        lat1, lon1 = self.spherical
        lat2, lon2 = other.spherical
        latChange = abs(lat1 - lat2)
        lonChange = abs(lon1 - lon2)
        return 2 * asin(sqrt(sin(latChange/2)**2 +
                            cos(lat1) * cos(lat2) * sin(lonChange/2)**2))

    def getDistance(self, other):
        """Laskee euklidisen etäisyyden"""
        return vector_magnitude(self.cartesian - other.cartesian)

    def rotate(self, angle, axis):
        """Rotoi pistettä akselin ympäri"""
        self.cartesian = rotate_vector(self.cartesian, angle, axis)

# ==================== FIBGRID ====================
class FibGrid:
    """Fibonacci-verkko tasaisesti jakautuneille pisteille pallolla,
       tallentaa pisteet ja coverage dict:eihin jotta negatiiviset indeksi toimivat odotetusti."""

    def __init__(self, avgDistance):
        self.avgDistance = avgDistance
        self.pointNum = self.getPointNum(avgDistance)
        self.totalPointNum = 2 * self.pointNum + 1
        self.zIncrement = 2.0 / self.totalPointNum

        # Käytetään dict:iä, avaimina indeksejä -pointNum..pointNum
        self.coverage = {}
        self.points = {}
        self.rotation_angle = 0.0
        self.rotation_axis = np.array([0.0, 1.0, 0.0])

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
        return (index * CIRCLE / phi) % CIRCLE

    def _getSpherical(self, index):
        index = min(index, self.pointNum) if index > 0 else max(index, -self.pointNum)
        return (asin(self._getZ(index)), self._getLon(index))

    def getCellNeighborIds(self, index):
        """Palauttaa naapurisolujen indeksit (avaimet -pointNum..pointNum)"""
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
        """Rotoi koko verkkoa"""
        self.rotation_angle += angle
        for i in list(self.points.keys()):
            self.points[i] = rotate_vector(self.points[i], angle, axis)
        # päivitä mahdollisten crust:ien koordinaatit
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
        """Palauttaa naapurisolut (Crust-objektit)"""
        neighbors = []
        for neighbor_id in self.getCellNeighborIds(crust.id):
            neighbor = self.coverage.get(neighbor_id)
            if neighbor is not None:
                neighbors.append(neighbor)
        return neighbors

# ==================== CRUST ====================
class Crust(GeoCoordinate):
    """Maankuoren pala"""

    def __init__(self, plate, world, isContinent=False, id=None):
        super().__init__()
        self.world = world
        self.plate = plate
        self.id = id
        # kopioijataan karteesiset koordinaatit laatan gridin vastaavasta
        self._cartesian = plate.grid.points[id].copy()

        self.subductedBy = None
        self.subducts = None
        self._firstSubductedBy = None

        self._isContinent = isContinent
        if isContinent:
            thickness = 36900  # m
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

    @property
    def smallCircleRadius(self):
        """Momenttivarsi Eulerin napaan nähden"""
        return moment_arm(self.cartesian, self.plate.eulerPole)

    @property
    def inertialMoment(self):
        """Hitausmomentti"""
        return self.pressure * self.smallCircleRadius ** 2

    def isContinent(self):
        return self.thickness > 17000

    def isDetaching(self):
        """Tarkistaa pitäisikö subduktoituvan kuoren irrota"""
        if not self._firstSubductedBy:
            return False
        distanceSubducted = self.getArcDistance(self._firstSubductedBy) * self.world.radius
        return distanceSubducted > self.world.maxMountainWidth

    def isostacy(self):
        """Laskee korkeuden tiheyden funktiona (isostaasi)"""
        thickness = self.thickness
        rootDepth = thickness * self.density / self.world.mantleDensity
        displacement = thickness - rootDepth
        self.displacement = displacement

    def erupt(self):
        """Purkaus: tulivuori"""
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
        """Törmäys toisen kuoren kanssa"""
        if self.subductedBy:
            top, bottom = other, self
        elif other.subductedBy:
            top, bottom = self, other
        else:
            top, bottom = sorted([self, other], key=lambda c: c.density)

        if top.subducts != bottom or bottom.subductedBy != top:
            if bottom.isDetaching():
                if bottom.isContinent() and top.isContinent():
                    # Mantereet sulautuvat — yksinkertaistettu
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
        """Tuhoaa kuoren palan"""
        self.plate.remove(self)
        if self.subductedBy:
            self.subductedBy.subducts = None
        if self.subducts:
            self.subducts.subductedBy = None

# ==================== PLATE ====================
class Plate(GeoCoordinate):
    """Laattatektoninen laatta"""

    def __init__(self, spherical, world, grid, speed=0, eulerPole=None):
        super().__init__(spherical=spherical)
        self.world = world
        self.speed = speed  # km/Myr
        self.eulerPole = eulerPole if eulerPole is not None else toCartesian((pi/2, 0))
        self.grid = grid
        self.crusts = []
        # määrittele perusväri laatalle: manner vs meri
        # värimuoto 0..1
        self.base_color = (random.random(), random.random(), random.random())
        self.color = (random.random(), random.random(), random.random())

        self._collidable = None
        self._riftable = None
        self._collisions = [None] * self.grid.totalPointNum

    @property
    def velocity(self):
        """Nopeusvektori (suuntaa euler-polen mukaan)"""
        vel = self.eulerPole.copy()
        vel_mag = vector_magnitude(vel)
        if vel_mag > 0:
            return vel / vel_mag * self.speed
        return vel

    def add(self, crust):
        self.grid.add(crust)
        self.crusts.append(crust)
        self._collidable = None
        self._riftable = None

    def remove(self, crust):
        self.grid.remove(crust)
        if crust in self.crusts:
            self.crusts.remove(crust)
        self._collidable = None

    def update(self, crust):
        """Päiväyttää kuoren tilan"""
        self._collidable = None

    @property
    def collidable(self):
        """Kuoret jotka voivat törmätä"""
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
        """Paikat joihin voi syntyä uutta kuorta"""
        if self._riftable is None:
            self._riftable = set()
            for crust in self.crusts:
                if not crust.subductedBy:
                    for neighbor_id in self.grid.getCellNeighborIds(crust.id):
                        if self.grid[neighbor_id] is None:
                            self._riftable.add(neighbor_id)
        return list(self._riftable)

    def getAngularSpeed(self, timestep=1):
        """Kulmanopeus radiaaneina"""
        speed = self.speed * timestep  # km
        smallCircleRadius = moment_arm(self.cartesian, self.eulerPole) * self.world.radius
        if smallCircleRadius > 0:
            return speed / smallCircleRadius
        return 0

    def move(self, timestep):
        """Liikuttaa laattaa"""
        angularSpeed = self.getAngularSpeed(timestep)
        # Rotoi laatan keskipistettä
        self.rotate(angularSpeed, self.eulerPole)
        # Rotoi koko gridia
        self.grid.rotate(angularSpeed, self.eulerPole)

    def collide(self):
        """Törmäysten käsittely (yksinkertaistettu)"""
        plates = [p for p in self.world.plates if p != self]
        for crust in self.collidable:
            for other_plate in plates:
                for other_crust in other_plate.crusts:
                    distance = crust.getDistance(other_crust)
                    if distance < self.world.minDistance * 1.5:
                        crust.collide(other_crust)
                        break

    def rift(self):
        """Luo uutta kuorta riftauskohtiin"""
        plates = [p for p in self.world.plates if p != self]
        for rift_id in self.riftable[:]:
            if self.grid[rift_id] is not None:
                continue
            rift_pos = self.grid.points[rift_id]
            collision = False
            for other_plate in plates:
                for other_crust in other_plate.crusts:
                    if vector_magnitude(rift_pos - other_crust.cartesian) < self.world.minDistance:
                        collision = True
                        break
                if collision:
                    break
            if not collision:
                Crust(self, self.world, False, rift_id)

# ==================== WORLD ====================
class World:
    """Maailman simulaatio"""

    mantleDensity = 3300
    waterDensity = 1026
    oceanCrustDensity = 2890
    continentCrustDensity = 2700

    def __init__(self, radius=6367, resolution=72, plateNum=5,
                 continentNum=3, continentSize=1250):
        self.age = 0.0
        self.radius = radius * 1000  # km -> m
        self.resolution = resolution
        avgPointDistance = 2 * pi / resolution

        # Luo laatat
        self.plates = []
        template = FibGrid(avgPointDistance)

        for i in range(plateNum):
            # Suhtaudutaan plate.base_color määritykseen vasta, kun luodaan crustit
            plate = Plate(
                self.randomPoint(),
                self,
                FibGrid(avgPointDistance),
                random.gauss(42.8, 27.7),  # Nopeus km/Myr
                toCartesian(self.randomPoint())
            )
            self.plates.append(plate)

        # Luo mannerkilvet
        shields = [GeoCoordinate(spherical=self.randomPoint())
                   for _ in range(continentNum)]
        continentSizeRad = continentSize * 1000 / self.radius  # km -> radiaanit

        # Luo maankuori
        for i in range(-template.pointNum, template.pointNum + 1):
            cartesian = template.points[i]
            # Etsi lähin laatta
            nearestPlate = min(self.plates,
                               key=lambda p: vector_magnitude(p.cartesian - cartesian))
            # Tarkista onko manner
            isContinent = any(
                vector_magnitude(shield.cartesian - cartesian) < continentSizeRad
                for shield in shields
            )
            Crust(nearestPlate, self, isContinent, id=i)

        # Aseta plate.base_color selkeästi manner/meri
        for plate in self.plates:
            # Jos laatan sisältämissä crusteissa on enemmän mannerta kuin merta -> mannerväri
            continent_count = sum(1 for c in plate.crusts if c.isContinent())
            if continent_count > 0.3 * max(1, len(plate.crusts)):
                plate.base_color = (0.2, 0.7, 0.2)  # vihreäisen väri mantereille
            else:
                plate.base_color = (0.2, 0.4, 0.85)  # sininen merilaattoihin

        self.seaLevel = 3790
        self.maxMountainWidth = 300 * 1000  # m

        area = 4 * pi / template.totalPointNum
        d = sqrt(area / sqrt(5))
        self.minDistance = sqrt(2) * d

    def randomPoint(self):
        """Palauttaa satunnaisen pisteen pallolla (lat, lon)"""
        return (asin(2 * random.random() - 1),
                2 * pi * random.random())

    def __iter__(self):
        for plate in self.plates:
            for crust in plate.crusts:
                yield crust

    def update(self, timestep):
        """Päiväyttää simulaation: liikuta laattoja, isostaasi, törmäykset, riftaus"""
        for plate in self.plates:
            plate.move(timestep)

        for crust in self:
            crust.isostacy()

        for plate in self.plates:
            plate.collide()

        for plate in self.plates:
            plate.rift()

        self.age += timestep

# ==================== VISUALISOINTI ====================
class Visualization:
    """Plotly-pohjainen 3D mesh-visualisointi (yhtenäinen pallo)"""

    def __init__(self, world):
        self.world = world

    def fix_winding(self, verts, triangles):
        """
        Korjaa kolmiot niin, että normaalit osoittavat aina ulospäin pallon keskipisteestä.
        Tämä on kamerariippumaton ja turvallinen kaikille kolmioille.
        """
        fixed = []
        for t in triangles:
            # t = [i0, i1, i2]
            i0, i1, i2 = int(t[0]), int(t[1]), int(t[2])
            p1 = verts[i0]
            p2 = verts[i1]
            p3 = verts[i2]

            # normaalivektori (ei normalisoitu)
            n = np.cross(p2 - p1, p3 - p1)

            # kolmion keskipiste
            centroid = (p1 + p2 + p3) / 3.0

            # jos normaali osoittaa sisäänpäin (dot < 0), käännetään kolmio
            if np.dot(n, centroid) < 0:
                fixed.append([i0, i2, i1])
            else:
                fixed.append([i0, i1, i2])
        return np.array(fixed, dtype=int)

    def create_mesh(self):
        """Luo yhtenäinen pallon mesh ja ryhmittele kolmiot laatan mukaan"""
        avgPointDistance = 2 * pi / self.world.resolution
        grid = FibGrid(avgPointDistance)

        # Kärjet ja avain->index
        keys = sorted(grid.points.keys())
        verts = np.array([grid.points[k] for k in keys])
        key_to_vidx = {k: idx for idx, k in enumerate(keys)}

        # Konveksi kuori (kokonaispallon triangulaatio)
        try:
            hull = ConvexHull(verts)
            triangles = hull.simplices
            triangles = self.fix_winding(verts, triangles)
        except Exception as e:
            print("Konveksisen kuoren luonti epäonnistui:", e)
            return []

        # Kerää kaikki crustit
        crusts = list(self.world.__iter__())
        if not crusts:
            return []

        # Määritä jokaiselle kärjelle lähin crust (voidaan optimoida KD-tree:llä myöhemmin)
        vert_plate_idx = np.full(len(verts), -1, dtype=int)
        vert_elevation = np.zeros(len(verts))

        for vi, v in enumerate(verts):
            best = None
            bestd = float('inf')
            for crust in crusts:
                d = np.linalg.norm(v - crust.cartesian)
                if d < bestd:
                    bestd = d
                    best = crust
            vert_plate_idx[vi] = self.world.plates.index(best.plate)
            vert_elevation[vi] = best.elevation

        # Ryhmittele kolmioita laatan mukaan
        triangles_by_plate = {}
        for tri in triangles:
            plate_ids = [vert_plate_idx[idx] for idx in tri]
            if any(pid < 0 for pid in plate_ids):
                continue
            plate_id = max(set(plate_ids), key=plate_ids.count)
            triangles_by_plate.setdefault(plate_id, []).append(tri)

        meshes = []
        # Sovella korkeusvenytystä ennen meshien luontia (muodosta displaced_verts)
        # pienentävä / suurentava skaala riippuen elevation-arvosta
        max_abs_elev = max(1.0, np.max(np.abs(vert_elevation)))
        elevation_scale = 1.0 + (vert_elevation / (self.world.radius * 0.02))
        displaced_verts = verts * elevation_scale[:, np.newaxis]

        for plate_id, tris in triangles_by_plate.items():
            plate = self.world.plates[plate_id]
            tris = np.array(tris, dtype=int)
            if tris.size == 0:
                continue

            # Laske laatan keskikorkeus ja säädä hieman väriä sen mukaan
            plate_vertex_indices = np.unique(tris.flatten())
            mean_elev = np.mean(vert_elevation[plate_vertex_indices]) if plate_vertex_indices.size > 0 else 0.0

            # Perusväri (manner/meri)
            base = np.array(plate.base_color)
            # Säädetään kirkkaus mean_elev:n mukaan (pieni vaikutus)
            bright_factor = 0.75 + 0.25 * (1.0 + mean_elev / max_abs_elev)
            bright_factor = float(bound(bright_factor, 0.4, 1.4))
            rgb = (base * bright_factor * 255.0).astype(int)
            rgb = np.clip(rgb, 0, 255)

            mesh = go.Mesh3d(
                x=displaced_verts[:, 0],
                y=displaced_verts[:, 1],
                z=displaced_verts[:, 2],
                i=tris[:, 0],
                j=tris[:, 1],
                k=tris[:, 2],
                color='rgb({},{},{})'.format(int(rgb[0]), int(rgb[1]), int(rgb[2])),
                opacity=0.98,
                name=f'Laatta {plate_id + 1}',
                hoverinfo='skip',
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
                lightposition=dict(x=100, y=200, z=0)
            )
            meshes.append(mesh)

        return meshes

    def plot(self):
        meshes = self.create_mesh()
        if not meshes:
            print("Ei meshejä piirrettäväksi")
            return

        fig = go.Figure(data=meshes)
        fig.update_layout(
            title=f'Laattatektoniikka - Ikä: {self.world.age:.1f} My<br>' +
                  f'Laattoja: {len(self.world.plates)} | ' +
                  f'Maankuoren paloja: {sum(len(p.crusts) for p in self.world.plates)}',
            scene=dict(
                xaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
                zaxis=dict(showbackground=False, showgrid=False, zeroline=False, visible=False),
                aspectmode='data',
                bgcolor='rgb(10, 10, 30)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            showlegend=True,
            width=1400,
            height=900,
            margin=dict(l=0, r=0, t=60, b=0)
        )
        fig.show()

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull

def fix_triangle_winding(verts, triangles):
    """
    Varmistaa, että jokaisen kolmion normaalit osoittavat ulospäin origosta (tai pallon keskipisteestä).
    """
    fixed_tris = []
    for tri in triangles:
        p0, p1, p2 = verts[tri]
        n = np.cross(p1 - p0, p2 - p0)
        centroid = (p0 + p1 + p2) / 3.0
        if np.dot(n, centroid) < 0:
            # käännä kolmion järjestys
            fixed_tris.append([tri[0], tri[2], tri[1]])
        else:
            fixed_tris.append(tri)
    return np.array(fixed_tris)


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



def main():
    print("=" * 60)
    print("LAATTATEKTONIIKKASIMULAATIO — KORJATTU VERSIO")
    print("=" * 60)
    print("\nAlustetaan simulaatio...")

    #world = World(radius=6367, resolution=32, plateNum=6,
    #              continentNum=3, continentSize=1250)
    world = World(radius=6367, resolution=100, plateNum=10,
                  continentNum=6, continentSize=3200)

    print(f"✓ Luotu {len(world.plates)} laattaa")
    print(f"✓ Yhteensä {sum(len(p.crusts) for p in world.plates)} maankuoren palaa")

    print("\nSimuloidaan laattatektoniikkaa...")
    for step in range(10):
        world.update(timestep=1.0)
        print(f"  Askel {step + 1:2d}/10 | Ikä: {world.age:5.1f} My | " +
              f"Laattoja: {len(world.plates)}")

    print("\n" + "=" * 60)


    export_world_mesh_by_plate(world, filename="earth_mesh_by_plate.ply")

    world_to_2d_raster_seamless(world, resolution_lon=2048, resolution_lat=1024, smoothing_sigma=2, filename="world_heightmap_seamless.png")
    world_to_2d_heightmap(world, resolution_lon=2048, resolution_lat=1024, smoothing_sigma=2, filename="world_heightmap_gray.png")
    world_to_2d_paletted(world, resolution_lon=2048, resolution_lat=1024, smoothing_sigma=2, filename="world_heightmap_paletted.png")

    print("Luodaan 3D-visualisointi...")
    visualize_world_pyvista_smooth(world)

    print("\n✓ Valmis! Selaimessa pitäisi avautua interaktiivinen 3D-näkymä.")
    print("\nKÄYTTÖOHJEET:")
    print("  • Vedä hiirellä pyörittääksesi planeettaa")
    print("  • Rullaa zoomataksesi")
    print("  • Vie hiiri laatan päälle nähdäksesi tietoja")
    print("=" * 60)

if __name__ == "__main__":
    main()

