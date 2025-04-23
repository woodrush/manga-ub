import pandas as pd
import os
from tqdm import tqdm

d_csv_paths = {
    "character_count":                       "tasks/character_count.csv",
    "emotion_benchmark_split":               "tasks/emotion_benchmark_split.csv",
    "onomatopoeia_scene":                    "tasks/onomatopoeia_scene.csv",
    "panel_localization":                    "tasks/panel_localization.csv",
    "recognition_background":                "tasks/recognition_background.csv",
    "next_panel_inference":                  "tasks/next_panel_inference/test.csv",
}

d_filter_combinations_list = {
    "recognition_background": [
        {'task': 'Location',          'label': "Indoors"},
        {'task': 'Location',          'label': "Outdoors"},
        {'task': 'Time_of_day',       'label': "Day"},
        {'task': 'Time_of_day',       'label': "Night"},
        {'task': 'Weather',           'label': "Sunny"},
        {'task': 'Weather',           'label': "Rainy"},
        {'task': 'Weather_difficult', 'label': "Sunny"},
        {'task': 'Weather_difficult', 'label': "Rainy"},
        {'task': 'Weather_difficult', 'label': "Snowy"},
    ],
    "character_count": [
        {'label': 0},
        {'label': 1},
        {'label': 2},
        {'label': 3},
        {'label': 4},
    ],
    "onomatopoeia_scene": [
        {'text': "しーん"},
        {'text': "はっ"},
        {'text': "ガチャ"},
        {'text': "ザッ"},
        {'text': "キャー"},
        {'text': "ニコ"},
        {'text': "キョロ"},
        {'text': "クス"},
        {'text': "ザワ"},
        {'text': "もぐ"},
        {'text': "ビクッ"},
        {'text': "ワー"},
    ],
    "emotion_benchmark_split": [
        {'category': "face"},
        {'category': "body"},
    ],
    "panel_localization": [
        {'category': 'Location',          'positive_label': "Indoors"},
        {'category': 'Location',          'positive_label': "Outdoors"},
        {'category': 'Time_of_day',       'positive_label': "Day"},
        {'category': 'Time_of_day',       'positive_label': "Night"},
        {'category': 'Weather',           'positive_label': "Sunny"},
        {'category': 'Weather',           'positive_label': "Rainy"},
    ],
    "next_panel_inference": [
        {"category": "easy",      "image_order_type": "rightfirst", "cropped_type": "raw",            "is_with_transcription": False},
        {"category": "easy",      "image_order_type": "leftfirst",  "cropped_type": "raw",            "is_with_transcription": False},
        {"category": "easy",      "image_order_type": "rightfirst", "cropped_type": "speech_cropped", "is_with_transcription": False},
        {"category": "easy",      "image_order_type": "rightfirst", "cropped_type": "raw",            "is_with_transcription": True},
        {"category": "difficult", "image_order_type": "rightfirst", "cropped_type": "raw",            "is_with_transcription": False},
        {"category": "difficult", "image_order_type": "leftfirst",  "cropped_type": "raw",            "is_with_transcription": False},
        {"category": "difficult", "image_order_type": "rightfirst", "cropped_type": "speech_cropped", "is_with_transcription": False},
        {"category": "difficult", "image_order_type": "rightfirst", "cropped_type": "raw",            "is_with_transcription": True},
    ],
}

keys = list(d_filter_combinations_list.keys())
csv_files = [d_csv_paths[k] for k in keys]
filter_combinations_list = [d_filter_combinations_list[k] for k in keys]

# Ensure the output directory exists
output_dir = 'vis'
os.makedirs(output_dir, exist_ok=True)

# Function to filter and sample the data
def filter_and_sample_data(df, filter_combinations):
    filtered_sampled_df = pd.DataFrame()
    for combination in filter_combinations:
        filtered_df = df
        for column, value in combination.items():
            filtered_df = filtered_df[filtered_df[column] == value]
        sampled_df = filtered_df.head(30)  # Get the first 30 entries
        filtered_sampled_df = pd.concat([filtered_sampled_df, sampled_df])
    return filtered_sampled_df

# Function to reorder columns
def reorder_columns(df):
    columns = list(df.columns)
    if 'impath' in columns:
        columns.remove('impath')
        columns.append('impath')
    if 'prompt' in columns:
        columns.remove('prompt')
        columns.insert(-1, 'prompt')
    return df[columns]

# Function to convert the DataFrame to an HTML table with img tags
def dataframe_to_html_with_images(df):
    html = '<html><head><meta charset="utf-8"></head><body>'
    html += '<table border="1" class="dataframe">\n'
    html += '  <thead>\n'
    html += '    <tr style="text-align: right;">\n'
    
    for column in df.columns:
        html += f'      <th>{column}</th>\n'
    
    html += '    </tr>\n'
    html += '  </thead>\n'
    html += '  <tbody>\n'
    
    for index, row in df.iterrows():
        html += '    <tr>\n'
        for column in df.columns:
            value = row[column]
            if column == 'impath':
                value = f'<img src="{value}" style="max-width:500px;">'
            else:
                value = str(value).replace('\n', '<br>')
            html += f'      <td>{value}</td>\n'
        html += '    </tr>\n'
    
    html += '  </tbody>\n'
    html += '</table>'
    html += '</html>'
    
    return html

# Process each CSV file and filter combinations
for i, (csv_file, filter_combinations) in tqdm(enumerate(zip(csv_files, filter_combinations_list))):
    df = pd.read_csv(csv_file)
    
    for j, combination in enumerate(filter_combinations):
        filtered_sampled_df = filter_and_sample_data(df, [combination])

        # Reorder columns
        filtered_sampled_df = reorder_columns(filtered_sampled_df)
        
        # Prepend "../" to the impath column
        if 'impath' in filtered_sampled_df.columns:
            filtered_sampled_df['impath'] = "../" + filtered_sampled_df['impath'].astype(str)
        
        # Convert the DataFrame to an HTML table with images
        html_table = dataframe_to_html_with_images(filtered_sampled_df)
        
        # Save the HTML table to a file
        output_filename = os.path.join(output_dir, f'visualization_{i+1}_{j+1}.html')
        with open(output_filename, 'wt', encoding='utf-8') as f:
            f.write(html_table)

print("HTML files with images have been created in the 'vis' directory.")
