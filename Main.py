import pygame
import random
from time import time
from Matrix import *
from Network import *


network = Network(784, 32, 10)
# network = Network(784, 16, 16, 10)


# Display
pygame.init()

entering_command = False
command_string = ""
frame_rate = 60
width = 960
height = 640
font = pygame.font.SysFont('/Library/Fonts/Arial.ttf', 24)

training = False
batch_per_frame = 100
batch_size = 8

window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Neural Network")


def process_command(command, args):
    try:
        if command == "train":
            start = time()
            for i in range(int(args[0])):
                network.auto_train(batch_size)
                if i % 1 == 0:
                    print(i+1)
            print("Time Taken: %.3f seconds" % (time() - start))
        elif command == "test":
            start = time()
            correct = 0
            for j in range(int(args[0])):
                i, o = network.gen_training_data(random.randint(0, 59999))
                network.feedforward(i)
                correct += Matrix.collapse(network.values[-1]) == Matrix.collapse(o)
                if j % 100 == 0:
                    print(j)
            print("Accuracy:", correct/int(args[0])*100)
            print("Time Taken: %.3f seconds" % (time() - start))
        elif command == "save":
            network.save(args[0])
        elif command == "load":
            network.load(args[0])
    except (IndexError, ValueError) as e:
        print(e)


while True:
    pygame.time.delay(1000 // frame_rate)
    for event in pygame.event.get():
        if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN and entering_command:
            if event.key == pygame.K_RETURN:
                if command_string != "":
                    process_command(command_string.split()[0], command_string.split()[1:])
                command_string = ""
                entering_command = False
            elif event.key == pygame.K_BACKSPACE:
                command_string = command_string[:-1]
            elif event.key >= 32 and event.key <= 126:
                command_string += chr(event.key)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                network.selected += 1
                if network.selected >= network.layers[0]:
                    network.selected = -1
            elif event.key == pygame.K_UP:
                network.selected -= 1
                if network.selected < -1:
                    network.selected = network.layers[0]-1
            elif event.key >= 48 and event.key <= 57:
                if network.selected >= 0:
                    network.values[0].data[network.selected][0] = network.values[0].data[network.selected][0] * 10 + event.key - 48
            elif event.key == pygame.K_MINUS:
                if network.selected >= 0:
                    network.values[0].data[network.selected][0] *= -1
            elif event.key == pygame.K_SLASH:
                if network.selected >= 0:
                    network.values[0].data[network.selected][0] /= 10
            elif event.key == pygame.K_BACKSPACE:
                if network.selected >= 0:
                    network.values[0].data[network.selected][0] = 0
                else:
                    network.clear()
            elif event.key == pygame.K_RETURN:
                network.feedforward(None)
            elif event.key == pygame.K_TAB:
                training = not training
            elif event.key == pygame.K_SPACE:
                network.auto_train(batch_size)
            elif event.key == pygame.K_t:
                network.print()
            elif event.key == pygame.K_r:
                network.feedforward(network.gen_training_data(random.randint(0, 59999))[0])
                print("Random Test Done")
            elif event.key == pygame.K_LSHIFT:
                entering_command = True

    keys = pygame.key.get_pressed()

    if training:
        for i in range(batch_per_frame):
            network.auto_train(batch_size)

    window.fill((0, 0, 0))
    # network.display(window, font, width, height, 280, 160, 30)
    if entering_command:
        render = font.render(command_string + "_", False, (255, 255, 255))
        window.blit(render, (8, height - render.get_size()[1] - 8))
        pygame.draw.rect(window, (255, 255, 255), (0, height-render.get_size()[1]-16, width-1, render.get_size()[1]+15), 2)
    pygame.display.update()
