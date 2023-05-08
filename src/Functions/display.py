from IPython.display import display # to display images
from PIL import Image
def displayRGB(environment, action = -1):
    match action:
        case 0:
            print('LEFT !')
        case 1:
            print('DOWN !')
        case 2:
            print('RIGHT !')
        case 3:
            print('UP !')
        case _:
            print('Start')
    rgb_array = environment.render()
    image = Image.fromarray(rgb_array)
    display(image)