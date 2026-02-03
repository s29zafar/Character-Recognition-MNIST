import json
import base64
import os

notebook_path = '/Users/saimzafar2002-apple.com/Desktop/Machine Learning/Character Recognition/ConAE.ipynb'
output_dir = '/Users/saimzafar2002-apple.com/Desktop/Machine Learning/Character Recognition/'

with open(notebook_path, 'r') as f:
    notebook = json.load(f)

image_count = 1
for cell in notebook['cells']:
    if 'outputs' in cell:
        for output in cell['outputs']:
            if 'data' in output and 'image/png' in output['data']:
                image_data = output['data']['image/png']
                # Sometimes it's a list of strings
                if isinstance(image_data, list):
                    image_data = "".join(image_data)
                
                image_bytes = base64.b64decode(image_data)
                
                output_filename = f'output_{image_count}.png'
                output_path = os.path.join(output_dir, output_filename)
                
                with open(output_path, 'wb') as img_file:
                    img_file.write(image_bytes)
                
                print(f'Saved {output_filename}')
                image_count += 1
