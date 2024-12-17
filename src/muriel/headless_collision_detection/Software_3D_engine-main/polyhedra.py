import numpy as np

class Polyhedra:
    def __init__(self, vertices):
        self.vertices = np.array([[*[c/1000 for c in x],1.0] for x in np.reshape(vertices,(-1, 3))])
        self.edges = self.construct_geometry(vertices)
        

    class Plane:
        def __init__(self, *vertices):
            v1,v2,v3 = [np.array(v) for v in vertices]
            normal = np.cross(v2-v1, v3-v1)
            unit_denom = [np.square(v) for v in vertices]
            unit_normal = normal/unit_denom
            dot = normal @ v1

    def construct_geometry(self, _map):
        map_length = len(_map)
        edges = []
        count = 0
        for x in range(map_length):
            count+=1
            for y in range(map_length):
                count+=1
                for z in range(map_length):
                    count+=1
                    if x!=y!=z:
                        legal = True
                        _vertex = self.is_intersecting(Polyhedra.Plane(*_map[x]), Polyhedra.Plane(*_map[y]),Polyhedra.Plane(*_map[z])) 
                        _vertex = _vertex if _vertex is not None else np.array([x,y,z])

                        for m in range(map_length):
                            count+=1
                            m_plane = Polyhedra.Plane(*_map[m])
                            if (m_plane.norm @ _vertex ) + m_plane.dot > 0:
                                # legal = False
                                pass
                            # print(f"count: {count}/{map_length**4}")
                        if legal:
                            edges+=[(x,y),(x,z),(y,z)]
        return edges
    
    def is_intersecting(self,plane_1, plane_2, plane_3):
    
        m1, m2, m3 = np.array((plane_1.norm, plane_2.norm, plane_3.norm))

        u = np.cross(m2,m3)
        denom = m1 @ u

        if denom == 0:
            return None
    
        d = np.array([plane_1.dot,plane_2.dot,plane_3.dot])
        v = np.cross(m1,d)
        ood = 1.0/denom

        x = (d @ u) *ood
        y = (m3 @ v) *ood
        z = -(m2 @ v) *ood
        return np.array([x,y,z])
    

_object = Polyhedra(0
    ((( -1104,1596,-48 ), ( -1120,1596,-48 ), ( -1120,1560,-48 )),
(( -1120,1560,80 ), (-1120,1596, 80 ),(-1104 ,1596 ,80)),
(( -1120 ,1560,96), ( -1104,1560,96 ), ( -1104,1560,-48 )), 
(( -1104,1560,96 ), ( -1104,1596,96 ), ( -1104,1596,-48)), 
(( -1100,1704,96 ), ( -1116,1704,96 ), ( -1116,1704,-48 )), 
(( -1120,1596,96 ), ( -1120,1560,96 ), ( -1120,1560,-48 )) ))
        
edges = (
    (0, 1),
    (0, 3),
    (0, 4),
    (2, 1),
    (2, 3),
    (2, 7),
    (6, 3),
    (6, 4),
    (6, 7),
    (5, 1),
    (5, 4),
    (5, 7),
)       

test_tri_edges = [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14], [15,16,17]]
test_tri_vert = _object.vertices