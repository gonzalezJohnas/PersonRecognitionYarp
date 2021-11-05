from PIL import Image

def format_face_coord(bottle_face_coord):
    import yarp
    """
    Process the face coordinates read by the yarp bottle
    :param bottle_face_coord: coordinates of detected face (yarp bottle)
    :return: list of boxes with box defined as  [top, left, bottom, right]
    """
    list_face_coord = []
    list_face_id = []

    for i in range(bottle_face_coord.size()):
        face_data = bottle_face_coord.get(i).asList()

        list_face_id.append(face_data.get(0).asString())
        face_coordinates = face_data.get(1).asList()

        list_face_coord.append([face_coordinates.get(0).asDouble(), face_coordinates.get(1).asDouble(),
                      face_coordinates.get(2).asDouble(), face_coordinates.get(3).asDouble()])

    return list_face_id, list_face_coord




def get_center_face(faces, width):
    """
    From a list of faces identify the most centered one according to the image width
    :param faces:
    :param width:
    :return: most centered face coordinate
    """
    min_x = 1e5
    min_index = -1

    for i, face_coord in enumerate(faces):
        center_face_x = face_coord[0] + ((face_coord[2] - face_coord[0]) / 2)
        if abs((width / 2)-center_face_x) < min_x:
            min_index = i
            min_x = abs((width / 2)-center_face_x)

    return [faces[min_index]]

def face_alignement(face_coord, frame, margin=15):
    """
    Preprocess tha face with standard face alignment preprocessing
    :param coord_faces: list of faces' coordinates
    :param frame: rgb image
    :return: list face aligned image
    """

    xmin = int(face_coord[0])
    ymin = int(face_coord[1])
    xmax = int(face_coord[2])
    ymax = int(face_coord[3])

    # heigth_face = (ymax - ymin)
    # width_face = (xmax - xmin)
    #
    # center_face = [ymin + (heigth_face / 2), xmin + (width_face / 2)]
    #
    # new_ymin = int(center_face[0] - margin)
    # new_xmin = int(center_face[1] - margin)
    #
    # new_ymax = int(center_face[0] + margin)
    # new_xmax = int(center_face[1] + margin)

    face = get_ROI(xmin, ymin, xmax, ymax, frame)
    return face


def get_ROI(x1, y1, x2, y2, frame):
    x1 = x1 if x1 > 0 else 0
    y1 = y1 if y1 > 0 else 0

    x2 = x2 if x2 <= frame.shape[1] else frame.shape[1]
    y2 = y2 if y2 <= frame.shape[0] else frame.shape[0]

    return frame[y1:y2, x1:x2]


def check_border_limit(coord, border_limit):
    if coord < 0:
        return 0
    elif coord > border_limit:
        return border_limit
    else:
        return coord


def format_names_to_bottle(list_names):
    import yarp

    list_objects_bottle = yarp.Bottle()
    list_objects_bottle.clear()

    for name in list_names:
        yarp_object_bottle = yarp.Bottle()
        yarp_object_bottle.addString(name[0])
        yarp_object_bottle.addDouble(round(float(name[1]), 2))
        list_objects_bottle.addList().read(yarp_object_bottle)

    return list_objects_bottle


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

def get_tensor_from_image( img_path, trans):
        frame = Image.open(img_path)
        tensor = trans(frame).unsqueeze(0).cuda(0)

        return tensor
