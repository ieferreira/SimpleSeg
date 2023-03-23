import ast

# Define a function to extract x,y coordinate pairs from path commands
def get_coordinates(path_commands):
    """
    Extracts x,y coordinates from path commands.
    :param path_commands: list of path commands
    :return: list of x,y coordinates
    """
    current_point = [0,0]
    coordinates = []
    for command in path_commands:
        if command[0] == 'M':
            current_point = command[1:]
            coordinates.append(current_point)
        elif command[0] == 'L':
            current_point = command[1:]
            coordinates.append(current_point)
        elif command[0] == 'Q':
            control_point = command[1:3]
            end_point = command[3:]
            for t in range(1, 11):
                t_normalized = t / 10
                x = (1 - t_normalized) ** 2 * current_point[0] + 2 * t_normalized * (1 - t_normalized) * control_point[0] + t_normalized ** 2 * end_point[0]
                y = (1 - t_normalized) ** 2 * current_point[1] + 2 * t_normalized * (1 - t_normalized) * control_point[1] + t_normalized ** 2 * end_point[1]
                coordinates.append([x, y])
            current_point = end_point
    return coordinates
