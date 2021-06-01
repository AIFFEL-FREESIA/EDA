import click
import os 
import pandas as pd
import cv2
import numpy as np
import re

def wkt_to_list(wkt):
    nums = re.findall(r'\d+(?:\.\d*)?', wkt)
    nums = [round(float(p)) for p in nums]
    coords = zip(*[iter(nums)] * 2)
    return np.array(list(coords), np.int32)

@click.command()
@click.option('--summary_path', help='Path to summary data')
@click.option('--list_path', help='Path to traget data')
@click.option('--save_path', help='Directory path to save label image')
@click.option('--width', default=1024)
@click.option('--height', default=1024)
def csv_to_label(summary_path, list_path, save_path, width, height):
    try:
        if not os.path.isfile(summary_path):
            raise Exception(summary_path)
        if not os.path.isfile(list_path):
            raise Exception(list_path)
        if not os.path.isdir(save_path):
            raise Exception(save_path)
    except Exception as e:
        print(f'[Error] No such file or directory: {e}')
        return
    

    CLASSES = {
        'building' : 0,
        'road' : 1
    }

    _color_map = [
        (165, 42, 42),
        (0, 192, 0),
        (255,255,255)
    ]

    df = pd.read_csv(summary_path)

    target_list = []
    with open(list_path, 'r') as f:
        target_list = f.readlines()

    target_list = [file.split('.')[0] for file in target_list]

    fill_char = click.style("#", fg="green")
    empty_char = click.style("-", fg="white", dim=True)
    label_text = "Creating labels... "
    with click.progressbar(iterable=target_list,
                           label=label_text,
                           fill_char=fill_char,
                           empty_char=empty_char) as items:
        for image_id in items:
            df_sub = df[df['image_id'] == image_id]
            polygons = df_sub['coordinates_pix'].array
            _type = df_sub['type'].array
            
            label = np.full((width, height, 3), 255, np.uint8)
            for polygon, _class in zip(polygons, _type):
                coords = wkt_to_list(polygon)
                color = _color_map[CLASSES[_class]]
                label = cv2.fillPoly(label, [coords], color=color)
                label = cv2.polylines(label, [coords], True, color=(255,255,255), thickness=1)
                
            cv2.imwrite(f'{save_path}/{image_id}.png', cv2.cvtColor(label, cv2.COLOR_BGR2RGB))

if __name__ == '__main__':
    csv_to_label()