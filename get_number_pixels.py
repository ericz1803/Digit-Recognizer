import pygame
import numpy

def draw(coords, scaling_factor):
    pygame.draw.rect(screen,[255,255,255],
                     [coords[0]-1,coords[1]-1,8,8])

def get_pixels():
    print("Draw a number on the screen. Press enter when done.")
    pygame.init()
    scaling_factor=5
    size=[28*scaling_factor,28*scaling_factor]
    global screen
    screen=pygame.display.set_mode(size)
    pygame.display.set_caption("Draw a number. Press enter when done.")
    screen.fill([0,0,0])
    pygame.display.update()

    done=False
    draw_mode=False
    while not done:
        
        events=pygame.event.get()
        for event in events:
            if event.type==pygame.MOUSEBUTTONDOWN:
                draw_mode=True
                draw(pygame.mouse.get_pos(),scaling_factor)
            if event.type==pygame.MOUSEMOTION and draw_mode:
                draw(pygame.mouse.get_pos(),scaling_factor)
            if event.type==pygame.MOUSEBUTTONUP:
                draw_mode=False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    pygame.image.save(screen, "screenshot.png")
                    done=True
                    pygame.display.quit()
                    pygame.quit()
                    return scaling_factor
        pygame.display.update()
                         
if __name__=='__main__':
    get_pixels()
