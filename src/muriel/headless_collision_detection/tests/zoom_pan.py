from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame
from pygame.locals import *

class Polyhedra:
    def __init__(self, vertices = None, edges = None):
        self.vertices = getattr(self, "vertices", vertices)
        self.edges = getattr(self, "edges", edges)
    
    def draw(self):
        glTranslatef(0.0, 0.0, -5)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        for i in self.edges:
            for idx in i:
                glVertex3fv(self.vertices[idx])
        glEnd()

class Cube(Polyhedra):
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
    

def resize(width, height):
    if height==0:
        height=1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0*width/height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def init():
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 100.0, 255.0, 100.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

def draw():
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

def main():

    video_flags = OPENGL|DOUBLEBUF
    
    pygame.init()
    screen = pygame.display.set_mode((600,600), video_flags)

    cube_ = Cube()
    resize(600,600)
    init()
    
    frames = 0
    ticks = pygame.time.get_ticks()
    while 1:
        event = pygame.event.poll()
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            break
        
        m_x,m_y = pygame.mouse.get_pos()
        rote = m_x/600 * 360
        draw()
        glTranslatef(3, 1, -5)
        glRotatef(45, 3, 1, 1)
        
        # glScalef(m_y, m_y, m_y)
        cube_.draw()
        glBegin(GL_TRIANGLES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(-1.0, -1.0, 0)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(1.0, -1.0, 0)
        glEnd()
        
        # glColor3f( 0,0,1 )
        pygame.display.flip()
        pygame.time.wait(10)
        frames = frames+1
        
        
    # print (f"fps:  {((frames*1000)/(pygame.time.get_ticks()-ticks))}")


if __name__ == '__main__': main()
