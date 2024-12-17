import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

##Define the vertices. usually a cube contains 8 vertices
vertices = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1),
)

door = ((( -1104,1596,-48 ), ( -1120,1596,-48 ), ( -1120,1560,-48 )),
(( -1120,1560,80 ), (-1120,1596, 80 ),(-1104 ,1596 ,80)),
(( -1120 ,1560,96), ( -1104,1560,96 ), ( -1104,1560,-48 )), 
(( -1104,1560,96 ), ( -1104,1596,96 ), ( -1104,1596,-48)), 
(( -1100,1704,96 ), ( -1116,1704,96 ), ( -1116,1704,-48 )), 
(( -1120,1596,96 ), ( -1120,1560,96 ), ( -1120,1560,-48 )) )


def cross(m_1, m_2):
    #u × v = (v2(t1 − u3) − t4, u3v1 − t3, t4 − u2(v1 − t2))
    # where t1 = u1 − u2, t2 = v2 + v3, t3 = u1v3, and t4 = t1t2 − t3

    u1, u2, u3 = m_1

    v1, v2, v3 = m_2

    t1 = u1 - u2 
    t2 = v2 + v3
    t3 = u1*v3
    t4 = (t1*t2) - t3

    return (v2*(t1 - u3) - t4, (u3*v1) - t3, t4 - u2*(v1 - t2))
def dot(m_1,m_2):
    u_v = 0
    u = 0
    v = 0
    for i in range(3):
        u_v += m_1[i] * m_2[i]
        u += m_1[i]**2
        v += m_2[i]**2
        
    return u_v

print(dot(( -1104,1596,-48 ), ( -1120,1596,-48 )))
p1, p2, p3 = np.array(( -1104,1596,-48 )), np.array(( -1120,1596,-48 )), np.array(( -1120,1560,-48 ))

print(np.cross(p2-p1, p3-p1))

class Vector(np.ndarray): pass







#     Vector d(p1.d, p2.d, p3.d);
# Vector v = Cross(m1, d);
# float ood = 1.0f / denom;
# p.x = Dot(d, u) * ood;
# p.y = Dot(m3, v) * ood;
# p.z = -Dot(m2, v) * ood;
# return 1;

# // Test if quadrilateral (a, b, c, d) is convex
# int IsConvexQuad(Point a, Point b, Point c, Point d)
# {
# // Quad is nonconvex if Dot(Cross(bd, ba), Cross(bd, bc)) >= 0
# Vector bda = Cross(d - b, a - b);
# Vector bdc = Cross(d - b, c - b);
# if (Dot(bda, bdc) >= 0.0f) return 0;
# // Quad is now convex iff Dot(Cross(ac, ad), Cross(ac, ab)) < 0




class Polyhedra:
    def __init__(self, vertices):
        self.edges = self.construct_geometry(vertices)
        self.vertices = vertices

    class Plane:
        def __init__(self, *vertices):
            v1,v2,v3 = [np.array(v) for v in vertices]
            self.norm = np.cross(v2-v1, v3-v1)
            self.dot = self.norm @ v1

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


##define 12 edges for the body
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
import numpy as np
def read_map_file():
    curly= 0
    paren = 0
    temp_ = []
    p = []
    from pathlib import Path

    path = Path(__file__).parent / "test_file_input.txt"
    with open(path) as _f:
        for line in _f:
            if "{" in line:
                curly+=1
                continue
            if "}" in line:
                curly-=1
                continue
            if curly %2 == 0:
                t = line.split(" ",maxsplit=15)
                t = t[:-1]
                for x in range(3):
                    p1, n1,n2,n3,p2 = t[x*5:(x+1)*5]
                    p.append(eval("".join([p1,n1,",",n2,",",n3,p2])))
            elif not curly:
                break
    return p
        
static_raw = read_map_file()
# print(door_edges)
# static_edges = construct_geometry(static_raw[0:25])
# print(static_raw[2])
##define function to draw the cube
# import numpy as np

# num = np.matrix(static_raw)
# print(num.max(), num.min())

# def Cube():
#     glBegin(GL_LINES)
#     for i in edges:
#             for idx in i:
#                 glVertex3fv(vertices[idx])
#     glEnd()

# #Globals
# zoom = 1
# scale_up = 10
# scale_down = 0.5

# SCREEN_WIDTH = 600
# SCREEN_HEIGHT = 600

# def view_system_state(screen_):
#     _font = pygame.font.Font(size=100)
#     pygame.display(f'Zoom: {zoom}', False, (220, 0, 0))
#     screen_.blit(text_surface, (300,300))

# def main():
#     pygame.init()
#     pygame.font.init()

#     display=(600,600)
#     screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
#     gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
#     glTranslatef(0.0, 0.0, -5)
#     # new_screen = pygame.Surface((300,300))
#     # new_screen.blit(screen, (0, 0), (world_left, world_top, world_right, world_bottom))
#     while True:
        
#         # screen.blit(pygame.transform.scale(new_screen, (600, 600)), (0, 0))
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 quit()
#         click = pygame.mouse.get_pressed()
#         key = pygame.key.get_pressed()
        
#         # ZOOM IN/OUT
#         if click and key[pygame.K_z] and zoom < 10:
#             zoom *= scale_up
#         elif click and key[pygame.K_x] and zoom > 0.5:
#             zoom *= scale_down

#         glRotatef(1, 3, 1, 1)
#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#         Cube()
#         view_system_state(screen)
#         pygame.display.flip()
#         pygame.time.wait(10)


# main()
