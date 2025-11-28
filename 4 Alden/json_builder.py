
#This file contains functions for adding symbols to an output Json file and connecting them
# The two important functions are Add_CSV(x,y) and Connect_CSV(x1,y1, x2,y2)

import json
#may need to import matrix to read it
#also need to know how much the original image is scaled down by

def find_symbol_by_coordinate(x, y, json_data):
    for symbol in json_data["symbols"]:
        x1, y1, x2, y2 = symbol["bbox"]

        # Check if point is inside bounding box
        if x1 <= x <= x2 and y1 <= y <= y2:
            return symbol  # Return full object

    return None  # If nothing found

def build_new_symbol(symbol, color):
    return {
        "name": symbol["name"],
        "id": symbol["id"],
        "cls_id": symbol["cls_id"],
        "color": color,  # [R, G, B]
        "connections": []
    }


def Add_CSV(x,y):
    #file we are writing to
    fileoutput = "diagram.json"
    #file we get from ai recongition
    imgRecJSON = "bs_connected.json"
    # scale of reduced image to original image
    scale = 16
    #create output file if it doesn't exist
    if not fileoutput.exists():
        fileoutput.write_text(json.dumps({"symbols": []}, indent=4))

    #open the output file
    with open(fileoutput, "r") as fileOutput:
        dataOutput = json.load(fileOutput)

    with open(imgRecJSON, "r") as fileInput:
        dataInput = json.load(fileInput)

    # get the color of the tile given
    # this currently won't function until the matrix is either passed as an arguement or made a global variable
    # if need be the color section can be removed and this can be commented out
    color = matrix[x][y]
    #get the data for the symbol at the given coordinates from the ai output file
    symbol = find_symbol_by_coordinate(x * scale ,y * scale, dataInput)

    if symbol is None:
        print("symbol not found")
        pass

    # create a new symbol with tailored info for the output file
    #has name, id, cls_id, color, and connections
    newSymbol = build_new_symbol(symbol, color)

    #add the new symbol to the output file
    dataOutput["symbols"].append(newSymbol)


    with fileoutput.open("w") as f:
        json.dump(dataOutput, f, indent=4)

    pass

def Connect_CSV(x1,y1, x2,y2):
    # file we are writing to
    fileoutput = "diagram.json"
    # file we get from ai recongition
    imgRecJSON = "bs_connected.json"
    # scale of reduced image to original image
    scale = 16

    #open the ai recognition file
    with open(imgRecJSON, "r") as fileInput:
        dataInput = json.load(fileInput)

    # get the data for the symbols at the given coordinates
    symbol1 = find_symbol_by_coordinate(x1 * scale ,y1 * scale, dataInput)
    symbol2 = find_symbol_by_coordinate(x2 * scale, y2 * scale, dataInput)


    if symbol1 is None or symbol2 is None:
        print("One or both symbols not found!")
        return

        # Add each other's id if not already connected
    if symbol2["id"] not in symbol1["connections"]:
        symbol1["connections"].append(symbol2["id"])
    if symbol1["id"] not in symbol2["connections"]:
        symbol2["connections"].append(symbol1["id"])

        # Save the updated JSON
    with open(fileoutput, "w") as f:
        json.dump(dataInput, f, indent=4)
    pass
